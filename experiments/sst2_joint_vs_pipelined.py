"""
SST-2에서 Joint DSP vs Pipelined DSP 비교 실험 스크립트.

- 모델: bert-base-uncased (12 layers, 12 heads -> 144 heads)
- 데이터: GLUE SST-2
- Pruning ratios: 10%, 30%, 50%
- Joint DSP: head mask를 학습 파라미터와 함께 end-to-end로 최적화
- Pipelined DSP: (1) 중요도 측정 → (2) 고정 mask로 fine-tuning
"""

import os
import json
import random
from typing import Dict, Tuple, List, Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    default_data_collator,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pruning_metrics import (
    gumbel_soft_top_k,
    STEFunction,
    TemperatureScheduler,
    SparsityMetric,
    PruningMetricTracker,
    convert_gate_to_mask,
    print_head_mask,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# 0. 유틸: Seed, 데이터, 모델 로딩
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """재현성을 위한 seed 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_sst2(batch_size: int = 32, max_samples: Optional[int] = None):
    """SST-2 데이터셋 준비"""
    print("\n" + "=" * 80)
    print("Preparing SST-2 dataset...")
    print("=" * 80)

    dataset = load_dataset("glue", "sst2")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    def preprocess(examples):
        return tokenizer(
            examples["sentence"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    encoded = dataset.map(preprocess, batched=True, desc="Tokenizing")
    encoded = encoded.remove_columns(["sentence", "idx"])
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Limit samples for faster experimentation
    train_dataset = encoded["train"]
    if max_samples and len(train_dataset) > max_samples:
        train_dataset = train_dataset.select(range(max_samples))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
    )
    eval_dataloader = DataLoader(
        encoded["validation"],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Eval samples: {len(encoded['validation'])}")
    print(f"Batch size: {batch_size}")

    return tokenizer, train_dataloader, eval_dataloader


def create_model():
    """BERT-base 모델 생성"""
    config = AutoConfig.from_pretrained("bert-base-uncased")
    config.num_labels = 2
    config.output_attentions = False
    config.output_hidden_states = False

    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        config=config,
    )
    return model.to(DEVICE)


# ---------------------------------------------------------------------------
# 1. 평가 함수
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_loader: DataLoader,
    head_mask: Optional[torch.Tensor] = None,
    desc: str = "Evaluating",
) -> Tuple[float, float]:
    """
    모델 평가

    Returns:
        accuracy, loss
    """
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0

    for batch in tqdm(eval_loader, desc=desc, leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        # Rename 'label' to 'labels' if necessary
        if 'label' in batch and 'labels' not in batch:
            batch['labels'] = batch.pop('label')

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            head_mask=head_mask,
        )

        logits = outputs.logits
        loss = outputs.loss

        preds = logits.argmax(dim=-1)
        labels = batch["labels"]

        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    model.train()
    accuracy = correct / total
    avg_loss = total_loss / total

    return accuracy, avg_loss


# ---------------------------------------------------------------------------
# 2. Head Importance 계산 (Pipelined DSP용)
# ---------------------------------------------------------------------------

def compute_head_importance(
    model: nn.Module,
    train_loader: DataLoader,
    n_layers: int,
    n_heads: int,
) -> torch.Tensor:
    """
    Gradient-based head importance (Michel et al., 2019)

    Returns:
        importance: (n_layers, n_heads) tensor
    """
    print("\nComputing head importance scores...")

    model.eval()
    head_importance = torch.zeros(n_layers, n_heads).to(DEVICE)
    head_mask = torch.ones(n_layers, n_heads).to(DEVICE)
    head_mask.requires_grad_(True)

    total_tokens = 0

    for batch in tqdm(train_loader, desc="Computing importance", leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        if 'label' in batch and 'labels' not in batch:
            batch['labels'] = batch.pop('label')

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            head_mask=head_mask,
        )

        loss = outputs.loss
        loss.backward()

        head_importance += head_mask.grad.abs().detach()
        total_tokens += batch["attention_mask"].float().sum().item()

        head_mask.grad = None

    # Normalize
    head_importance /= total_tokens

    # Layer-wise normalization
    norm_by_layer = torch.pow(torch.pow(head_importance, 2).sum(-1), 0.5)
    head_importance /= (norm_by_layer.unsqueeze(-1) + 1e-20)

    # Global normalization
    head_importance = (head_importance - head_importance.min()) / \
                     (head_importance.max() - head_importance.min() + 1e-20)

    model.train()
    return head_importance


# ---------------------------------------------------------------------------
# 3. Joint DSP 실험
# ---------------------------------------------------------------------------

def run_joint_dsp(
    pruning_ratio: float,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    num_epochs: int = 3,
    use_ste: bool = False,
    log_interval: int = 100,
) -> Dict:
    """
    Joint DSP:
    - Importance weights (w)와 BERT weights를 동시에 학습
    - Training 중 head_mask는 differentiable하게 생성
    """
    print("\n" + "=" * 80)
    print(f"Joint DSP - Pruning Ratio: {int(pruning_ratio*100)}%")
    print("=" * 80)

    set_seed(42)
    model = create_model()

    num_layers = model.config.num_hidden_layers  # 12
    num_heads = model.config.num_attention_heads  # 12
    total_heads = num_layers * num_heads  # 144
    keep_heads = int(round((1.0 - pruning_ratio) * total_heads))

    print(f"Total heads: {total_heads}")
    print(f"Keeping: {keep_heads} ({keep_heads/total_heads*100:.1f}%)")
    print(f"Pruning: {total_heads - keep_heads} ({pruning_ratio*100:.1f}%)")

    # Learnable importance weights
    w = nn.Parameter(torch.randn(num_layers, num_heads).double().to(DEVICE))
    nn.init.xavier_uniform_(w)

    # Temperature scheduler
    num_training_steps = len(train_loader) * num_epochs
    temp_scheduler = TemperatureScheduler(
        initial_temperature=1000.0,
        final_temperature=1e-2,
        cooldown_steps=num_training_steps,
    )

    # Optimizer: model + importance weights
    optimizer = AdamW([
        {'params': model.parameters(), 'lr': 2e-5},
        {'params': [w], 'lr': 0.5},  # Higher LR for importance weights
    ])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    tracker = PruningMetricTracker()
    global_step = 0
    best_acc = 0.0

    model.train()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Training")

        for batch in progress_bar:
            global_step += 1
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            if 'label' in batch and 'labels' not in batch:
                batch['labels'] = batch.pop('label')

            # Temperature annealing
            temperature = temp_scheduler.get_temperature(global_step)

            # Generate soft mask (differentiable)
            if use_ste:
                soft_mask = STEFunction.apply(w.view(-1), keep_heads).view_as(w)
            else:
                soft_mask = gumbel_soft_top_k(
                    w.view(-1).float(),
                    keep_heads,
                    temperature
                ).double().view_as(w)

            # Forward with soft mask
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                head_mask=soft_mask.float(),
            )

            loss = outputs.loss
            epoch_loss += loss.item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Logging
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'temp': f'{temperature:.2e}',
            })

            if global_step % log_interval == 0:
                with torch.no_grad():
                    sparsity_stats = SparsityMetric.get_stats(soft_mask.detach())
                    acc, eval_loss = evaluate(
                        model, eval_loader,
                        head_mask=soft_mask.float().detach(),
                        desc="Evaluating"
                    )

                    if acc > best_acc:
                        best_acc = acc

                    tracker.update(
                        step=global_step,
                        mask=soft_mask.float().detach(),
                        performance=acc,
                        temperature=temperature,
                    )

                    print(f"\n[Step {global_step}] "
                          f"loss={loss.item():.4f}, "
                          f"eval_loss={eval_loss:.4f}, "
                          f"acc={acc*100:.2f}%, "
                          f"best={best_acc*100:.2f}%, "
                          f"temp={temperature:.2e}, "
                          f"sparsity={sparsity_stats['sparsity_pct']:.1f}%")

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} avg loss: {avg_epoch_loss:.4f}")

    # Final evaluation with hard mask
    print("\nGenerating final hard mask...")
    final_mask = convert_gate_to_mask(w.float().detach(), keep_heads)
    print_head_mask(final_mask)

    final_acc, final_loss = evaluate(
        model, eval_loader,
        head_mask=final_mask,
        desc="Final evaluation"
    )

    results = {
        'method': 'joint_dsp',
        'pruning_ratio': pruning_ratio,
        'keep_heads': keep_heads,
        'total_heads': total_heads,
        'final_accuracy': final_acc,
        'final_loss': final_loss,
        'best_accuracy': best_acc,
        'use_ste': use_ste,
        'history': tracker.get_history(),
    }

    print(f"\n[Joint DSP] Final Results:")
    print(f"  Pruning ratio: {int(pruning_ratio*100)}%")
    print(f"  Keep heads: {keep_heads}/{total_heads}")
    print(f"  Final accuracy: {final_acc*100:.2f}%")
    print(f"  Best accuracy: {best_acc*100:.2f}%")

    return results


# ---------------------------------------------------------------------------
# 4. Pipelined DSP 실험
# ---------------------------------------------------------------------------

def run_pipelined_dsp(
    pruning_ratio: float,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    num_epochs_base: int = 1,
    num_epochs_ft: int = 3,
    log_interval: int = 100,
) -> Dict:
    """
    Pipelined DSP:
    - Stage 1: Base fine-tuning
    - Stage 2: Head importance 계산
    - Stage 3: Binary mask 생성 및 고정
    - Stage 4: Pruned model fine-tuning
    """
    print("\n" + "=" * 80)
    print(f"Pipelined DSP - Pruning Ratio: {int(pruning_ratio*100)}%")
    print("=" * 80)

    set_seed(42)
    model = create_model()

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    total_heads = num_layers * num_heads
    keep_heads = int(round((1.0 - pruning_ratio) * total_heads))

    print(f"Total heads: {total_heads}")
    print(f"Keeping: {keep_heads} ({keep_heads/total_heads*100:.1f}%)")
    print(f"Pruning: {total_heads - keep_heads} ({pruning_ratio*100:.1f}%)")

    # Stage 1: Base fine-tuning
    print("\n" + "-" * 80)
    print("Stage 1: Base fine-tuning")
    print("-" * 80)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = len(train_loader) * num_epochs_base
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    model.train()
    for epoch in range(num_epochs_base):
        print(f"\nBase training epoch {epoch + 1}/{num_epochs_base}")

        for batch in tqdm(train_loader, desc="Base training"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            if 'label' in batch and 'labels' not in batch:
                batch['labels'] = batch.pop('label')

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    base_acc, base_loss = evaluate(model, eval_loader, desc="Base model eval")
    print(f"Base model accuracy: {base_acc*100:.2f}%")

    # Stage 2: Compute head importance
    print("\n" + "-" * 80)
    print("Stage 2: Computing head importance")
    print("-" * 80)

    importance = compute_head_importance(
        model, train_loader, num_layers, num_heads
    )

    print("\nHead importance scores:")
    print_head_mask(importance)

    # Stage 3: Generate binary mask (top-k)
    print("\n" + "-" * 80)
    print("Stage 3: Generating binary mask")
    print("-" * 80)

    head_mask = convert_gate_to_mask(importance, keep_heads)
    print("\nBinary mask (keep top-k):")
    print_head_mask(head_mask)

    # Evaluate with mask before fine-tuning
    pruned_acc_before, _ = evaluate(
        model, eval_loader,
        head_mask=head_mask,
        desc="Pruned (before FT)"
    )
    print(f"Pruned accuracy (before fine-tuning): {pruned_acc_before*100:.2f}%")

    # Stage 4: Fine-tune with fixed mask
    print("\n" + "-" * 80)
    print("Stage 4: Fine-tuning with fixed mask")
    print("-" * 80)

    head_mask = head_mask.to(DEVICE)
    head_mask.requires_grad = False  # Mask is fixed

    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = len(train_loader) * num_epochs_ft
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    best_acc = pruned_acc_before
    global_step = 0

    model.train()

    for epoch in range(num_epochs_ft):
        print(f"\nFine-tuning epoch {epoch + 1}/{num_epochs_ft}")

        progress_bar = tqdm(train_loader, desc="Fine-tuning")

        for batch in progress_bar:
            global_step += 1
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            if 'label' in batch and 'labels' not in batch:
                batch['labels'] = batch.pop('label')

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                head_mask=head_mask,
            )

            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            if global_step % log_interval == 0:
                acc, eval_loss = evaluate(
                    model, eval_loader,
                    head_mask=head_mask,
                    desc="Evaluating"
                )

                if acc > best_acc:
                    best_acc = acc

                print(f"\n[Step {global_step}] "
                      f"loss={loss.item():.4f}, "
                      f"eval_loss={eval_loss:.4f}, "
                      f"acc={acc*100:.2f}%, "
                      f"best={best_acc*100:.2f}%")

    # Final evaluation
    final_acc, final_loss = evaluate(
        model, eval_loader,
        head_mask=head_mask,
        desc="Final evaluation"
    )

    results = {
        'method': 'pipelined_dsp',
        'pruning_ratio': pruning_ratio,
        'keep_heads': keep_heads,
        'total_heads': total_heads,
        'base_accuracy': base_acc,
        'pruned_accuracy_before_ft': pruned_acc_before,
        'final_accuracy': final_acc,
        'final_loss': final_loss,
        'best_accuracy': best_acc,
    }

    print(f"\n[Pipelined DSP] Final Results:")
    print(f"  Pruning ratio: {int(pruning_ratio*100)}%")
    print(f"  Keep heads: {keep_heads}/{total_heads}")
    print(f"  Base accuracy: {base_acc*100:.2f}%")
    print(f"  Pruned (before FT): {pruned_acc_before*100:.2f}%")
    print(f"  Final accuracy: {final_acc*100:.2f}%")
    print(f"  Best accuracy: {best_acc*100:.2f}%")

    return results


# ---------------------------------------------------------------------------
# 5. 메인: 실험 실행 및 결과 저장
# ---------------------------------------------------------------------------

def main():
    """메인 실험 실행"""
    set_seed(42)

    # Configuration
    pruning_ratios = [0.1, 0.3, 0.5]
    batch_size = 32
    max_train_samples = None  # None for full dataset, set to e.g. 5000 for quick test

    # Prepare data
    tokenizer, train_loader, eval_loader = prepare_sst2(
        batch_size=batch_size,
        max_samples=max_train_samples
    )

    # Results storage
    all_results = {
        'config': {
            'task': 'sst2',
            'model': 'bert-base-uncased',
            'batch_size': batch_size,
            'pruning_ratios': pruning_ratios,
            'max_train_samples': max_train_samples,
        },
        'joint_dsp': [],
        'pipelined_dsp': [],
    }

    # Run Joint DSP experiments
    print("\n\n" + "=" * 80)
    print("JOINT DSP EXPERIMENTS")
    print("=" * 80)

    for ratio in pruning_ratios:
        try:
            result = run_joint_dsp(
                pruning_ratio=ratio,
                train_loader=train_loader,
                eval_loader=eval_loader,
                num_epochs=3,
                use_ste=False,
                log_interval=200,
            )
            all_results['joint_dsp'].append(result)
        except Exception as e:
            print(f"Error in Joint DSP with ratio {ratio}: {e}")
            import traceback
            traceback.print_exc()

    # Run Pipelined DSP experiments
    print("\n\n" + "=" * 80)
    print("PIPELINED DSP EXPERIMENTS")
    print("=" * 80)

    for ratio in pruning_ratios:
        try:
            result = run_pipelined_dsp(
                pruning_ratio=ratio,
                train_loader=train_loader,
                eval_loader=eval_loader,
                num_epochs_base=1,
                num_epochs_ft=3,
                log_interval=200,
            )
            all_results['pipelined_dsp'].append(result)
        except Exception as e:
            print(f"Error in Pipelined DSP with ratio {ratio}: {e}")
            import traceback
            traceback.print_exc()

    # Print final summary
    print("\n\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nTask: SST-2")
    print(f"Model: BERT-base-uncased (12 layers × 12 heads = 144 total heads)")

    print("\n" + "-" * 80)
    print("Joint DSP Results:")
    print("-" * 80)
    print(f"{'Prune %':>10} {'Keep':>8} {'Final Acc':>12} {'Best Acc':>12}")
    print("-" * 80)
    for r in all_results['joint_dsp']:
        print(f"{int(r['pruning_ratio']*100):>10}% "
              f"{r['keep_heads']:>8}/{r['total_heads']:<3} "
              f"{r['final_accuracy']*100:>11.2f}% "
              f"{r['best_accuracy']*100:>11.2f}%")

    print("\n" + "-" * 80)
    print("Pipelined DSP Results:")
    print("-" * 80)
    print(f"{'Prune %':>10} {'Keep':>8} {'Base':>8} {'Before FT':>10} {'Final Acc':>12} {'Best Acc':>12}")
    print("-" * 80)
    for r in all_results['pipelined_dsp']:
        print(f"{int(r['pruning_ratio']*100):>10}% "
              f"{r['keep_heads']:>8}/{r['total_heads']:<3} "
              f"{r['base_accuracy']*100:>7.2f}% "
              f"{r['pruned_accuracy_before_ft']*100:>9.2f}% "
              f"{r['final_accuracy']*100:>11.2f}% "
              f"{r['best_accuracy']*100:>11.2f}%")

    # Save results
    output_dir = "experiments/results"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "sst2_joint_vs_pipelined.json")

    # Convert torch tensors to lists for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj

    all_results_json = convert_for_json(all_results)

    with open(output_file, 'w') as f:
        json.dump(all_results_json, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("\nExperiment completed!")


if __name__ == "__main__":
    main()
