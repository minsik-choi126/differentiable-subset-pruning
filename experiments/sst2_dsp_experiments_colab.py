"""
SST-2 Joint vs Pipelined DSP 비교 실험 - Colab Ready

Google Colab에서 바로 실행 가능한 완전한 스크립트입니다.

사용법:
1. Google Colab에서 이 파일을 업로드
2. Runtime > Run all 클릭
3. 결과 확인

또는:
1. 이 파일을 로컬에 저장
2. python sst2_dsp_experiments_colab.py 실행
"""

# ============================================================================
# Colab 환경 설정 (Colab에서만 실행됨)
# ============================================================================
try:
    import google.colab
    IN_COLAB = True
    print("Google Colab 환경 감지!")

    # 필요한 패키지 설치
    print("\n패키지 설치 중...")
    !pip install -q transformers datasets torch

except ImportError:
    IN_COLAB = False
    print("로컬 환경에서 실행 중")


# ============================================================================
# Import
# ============================================================================
import os
import json
import random
import math
from typing import Dict, Tuple, List, Optional
from tqdm.auto import tqdm

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n사용 디바이스: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


# ============================================================================
# DSP 핵심 함수들 (pruning_metrics.py 내용)
# ============================================================================

EPSILON = torch.finfo(torch.double).tiny


def gumbel_soft_top_k(w: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
    """
    Differentiable top-k selection using Gumbel-Softmax.

    Args:
        w: Importance weights (n_total_heads,)
        k: Number of heads to keep
        temperature: Temperature for softmax

    Returns:
        Soft mask (n_total_heads,) with sum ≈ k
    """
    # Gumbel noise
    u = torch.rand_like(w) * (1 - EPSILON) + EPSILON
    gumbel_noise = -torch.log(-torch.log(u))
    r = gumbel_noise + w
    epsilon = torch.ones_like(r) * EPSILON

    # Iterative soft top-k
    p = torch.zeros([k, w.size()[0]]).to(w.device).double()
    p[0] = torch.exp(nn.functional.log_softmax(r / temperature, 0))

    for j in range(1, k):
        r = r + torch.log(torch.max(1 - p[j - 1], epsilon))
        p[j] = torch.exp(nn.functional.log_softmax(r / temperature, 0))

    return p.sum(0)


class STEFunction(torch.autograd.Function):
    """Straight-Through Estimator for hard top-k selection."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, k: int) -> torch.Tensor:
        threshold = input.sort(descending=True)[0][k]
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output, None


class TemperatureScheduler:
    """Temperature annealing scheduler for DSP."""

    def __init__(
        self,
        initial_temperature: float = 1000.0,
        final_temperature: float = 1e-2,
        cooldown_steps: int = 25000
    ):
        self.initial_temp = initial_temperature
        self.final_temp = final_temperature
        self.cooldown_steps = cooldown_steps
        self.log_initial = np.log(initial_temperature)
        self.log_final = np.log(final_temperature)

    def get_temperature(self, step: int) -> float:
        if step >= self.cooldown_steps:
            return self.final_temp

        progress = step / self.cooldown_steps
        log_temp = self.log_initial - progress * (self.log_initial - self.log_final)
        return np.exp(log_temp)


def convert_gate_to_mask(
    gates: torch.Tensor,
    num_of_heads: int,
    threshold: float = 0.5
) -> torch.Tensor:
    """Convert gate values to binary masks (top-k selection)."""
    head_mask = torch.zeros_like(gates)
    flat_gates = gates.view(-1)
    top_k_indices = flat_gates.argsort(descending=True)[:num_of_heads]
    flat_mask = head_mask.view(-1)
    flat_mask[top_k_indices] = 1.0
    return flat_mask.view_as(gates)


def compute_sparsity(mask: torch.Tensor) -> Dict[str, float]:
    """Compute sparsity statistics."""
    total = mask.numel()
    remaining = mask.sum().item()
    pruned = total - remaining

    return {
        "total_heads": total,
        "remaining_heads": remaining,
        "pruned_heads": pruned,
        "sparsity_pct": 100.0 * pruned / total,
        "remaining_pct": 100.0 * remaining / total,
    }


def print_head_mask(mask: torch.Tensor):
    """Pretty-print a head mask tensor."""
    n_layers, n_heads = mask.shape

    print("Layer/Head >\t" + "\t".join(f"{i+1}" for i in range(min(n_heads, 12))))

    for layer_idx in range(n_layers):
        if mask.dtype in [torch.long, torch.int]:
            values = "\t".join(f"{int(x)}" for x in mask[layer_idx, :min(n_heads, 12)].cpu())
        else:
            values = "\t".join(f"{x:.2f}" for x in mask[layer_idx, :min(n_heads, 12)].cpu())

        print(f"Layer {layer_idx + 1}:\t{values}")

    stats = compute_sparsity(mask)
    print(f"\nRemaining: {stats['remaining_heads']}/{stats['total_heads']} "
          f"({stats['remaining_pct']:.1f}%)")


# ============================================================================
# 유틸리티 함수
# ============================================================================

def set_seed(seed: int = 42):
    """재현성을 위한 seed 고정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_sst2(batch_size: int = 32, max_samples: Optional[int] = None):
    """SST-2 데이터셋 준비"""
    print("\n" + "=" * 80)
    print("SST-2 데이터셋 준비 중...")
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

    encoded = dataset.map(preprocess, batched=True, desc="토큰화")
    encoded = encoded.remove_columns(["sentence", "idx"])
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

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

    print(f"훈련 샘플: {len(train_dataset)}")
    print(f"검증 샘플: {len(encoded['validation'])}")
    print(f"배치 크기: {batch_size}")

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


@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_loader: DataLoader,
    head_mask: Optional[torch.Tensor] = None,
    desc: str = "평가 중",
) -> Tuple[float, float]:
    """모델 평가"""
    model.eval()
    correct, total = 0, 0
    total_loss = 0.0

    for batch in tqdm(eval_loader, desc=desc, leave=False):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}

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
    return correct / total, total_loss / total


def compute_head_importance(
    model: nn.Module,
    train_loader: DataLoader,
    n_layers: int,
    n_heads: int,
    max_batches: int = 100,
) -> torch.Tensor:
    """Gradient-based head importance (Michel et al., 2019)"""
    print("\nHead importance 계산 중...")

    model.eval()
    head_importance = torch.zeros(n_layers, n_heads).to(DEVICE)
    head_mask = torch.ones(n_layers, n_heads).to(DEVICE)
    head_mask.requires_grad_(True)

    total_tokens = 0

    for i, batch in enumerate(tqdm(train_loader, desc="Importance 계산", leave=False)):
        if i >= max_batches:
            break

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


# ============================================================================
# Joint DSP
# ============================================================================

def run_joint_dsp(
    pruning_ratio: float,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    num_epochs: int = 3,
    use_ste: bool = False,
    log_interval: int = 200,
) -> Dict:
    """Joint DSP: End-to-end learnable head selection"""

    print("\n" + "=" * 80)
    print(f"Joint DSP - Pruning Ratio: {int(pruning_ratio*100)}%")
    print("=" * 80)

    set_seed(42)
    model = create_model()

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    total_heads = num_layers * num_heads
    keep_heads = int(round((1.0 - pruning_ratio) * total_heads))

    print(f"전체 heads: {total_heads}")
    print(f"유지: {keep_heads} ({keep_heads/total_heads*100:.1f}%)")
    print(f"제거: {total_heads - keep_heads} ({pruning_ratio*100:.1f}%)")

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

    # Optimizer
    optimizer = AdamW([
        {'params': model.parameters(), 'lr': 2e-5},
        {'params': [w], 'lr': 0.5},
    ])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    global_step = 0
    best_acc = 0.0
    model.train()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"훈련")

        for batch in progress_bar:
            global_step += 1
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            if 'label' in batch and 'labels' not in batch:
                batch['labels'] = batch.pop('label')

            # Temperature
            temperature = temp_scheduler.get_temperature(global_step)

            # Generate mask
            if use_ste:
                soft_mask = STEFunction.apply(w.view(-1), keep_heads).view_as(w)
            else:
                soft_mask = gumbel_soft_top_k(
                    w.view(-1).float(), keep_heads, temperature
                ).double().view_as(w)

            # Forward
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

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'temp': f'{temperature:.2e}',
            })

            if global_step % log_interval == 0:
                acc, eval_loss = evaluate(
                    model, eval_loader,
                    head_mask=soft_mask.float().detach(),
                    desc="평가"
                )
                best_acc = max(best_acc, acc)

                print(f"\n[Step {global_step}] "
                      f"loss={loss.item():.4f}, "
                      f"eval_acc={acc*100:.2f}%, "
                      f"best={best_acc*100:.2f}%")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} 평균 loss: {avg_loss:.4f}")

    # Final mask
    print("\n최종 hard mask 생성...")
    final_mask = convert_gate_to_mask(w.float().detach(), keep_heads)
    print_head_mask(final_mask)

    final_acc, final_loss = evaluate(
        model, eval_loader, head_mask=final_mask, desc="최종 평가"
    )

    print(f"\n[Joint DSP] 최종 결과:")
    print(f"  Pruning ratio: {int(pruning_ratio*100)}%")
    print(f"  최종 accuracy: {final_acc*100:.2f}%")
    print(f"  최고 accuracy: {best_acc*100:.2f}%")

    return {
        'method': 'joint_dsp',
        'pruning_ratio': pruning_ratio,
        'final_accuracy': final_acc,
        'best_accuracy': best_acc,
    }


# ============================================================================
# Pipelined DSP
# ============================================================================

def run_pipelined_dsp(
    pruning_ratio: float,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    num_epochs_base: int = 1,
    num_epochs_ft: int = 3,
    log_interval: int = 200,
) -> Dict:
    """Pipelined DSP: Two-stage pruning"""

    print("\n" + "=" * 80)
    print(f"Pipelined DSP - Pruning Ratio: {int(pruning_ratio*100)}%")
    print("=" * 80)

    set_seed(42)
    model = create_model()

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    total_heads = num_layers * num_heads
    keep_heads = int(round((1.0 - pruning_ratio) * total_heads))

    print(f"전체 heads: {total_heads}")
    print(f"유지: {keep_heads} ({keep_heads/total_heads*100:.1f}%)")

    # Stage 1: Base fine-tuning
    print("\n" + "-" * 80)
    print("Stage 1: Base 모델 fine-tuning")
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
        print(f"\nBase 훈련 epoch {epoch + 1}/{num_epochs_base}")

        for batch in tqdm(train_loader, desc="Base 훈련"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            if 'label' in batch and 'labels' not in batch:
                batch['labels'] = batch.pop('label')

            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    base_acc, _ = evaluate(model, eval_loader, desc="Base 평가")
    print(f"Base accuracy: {base_acc*100:.2f}%")

    # Stage 2: Compute importance
    print("\n" + "-" * 80)
    print("Stage 2: Head importance 계산")
    print("-" * 80)

    importance = compute_head_importance(
        model, train_loader, num_layers, num_heads, max_batches=100
    )
    print("\nImportance scores:")
    print_head_mask(importance)

    # Stage 3: Generate mask
    print("\n" + "-" * 80)
    print("Stage 3: Binary mask 생성")
    print("-" * 80)

    head_mask = convert_gate_to_mask(importance, keep_heads).to(DEVICE)
    print_head_mask(head_mask)

    pruned_acc_before, _ = evaluate(
        model, eval_loader, head_mask=head_mask, desc="Pruning 후 (FT 전)"
    )
    print(f"Pruned accuracy (FT 전): {pruned_acc_before*100:.2f}%")

    # Stage 4: Fine-tune
    print("\n" + "-" * 80)
    print("Stage 4: Pruned 모델 fine-tuning")
    print("-" * 80)

    head_mask.requires_grad = False

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

        for batch in tqdm(train_loader, desc="Fine-tuning"):
            global_step += 1
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            if 'label' in batch and 'labels' not in batch:
                batch['labels'] = batch.pop('label')

            outputs = model(**batch, head_mask=head_mask)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if global_step % log_interval == 0:
                acc, _ = evaluate(
                    model, eval_loader, head_mask=head_mask, desc="평가"
                )
                best_acc = max(best_acc, acc)
                print(f"\n[Step {global_step}] acc={acc*100:.2f}%, best={best_acc*100:.2f}%")

    final_acc, _ = evaluate(model, eval_loader, head_mask=head_mask, desc="최종 평가")

    print(f"\n[Pipelined DSP] 최종 결과:")
    print(f"  Pruning ratio: {int(pruning_ratio*100)}%")
    print(f"  Base: {base_acc*100:.2f}%")
    print(f"  Pruned (FT 전): {pruned_acc_before*100:.2f}%")
    print(f"  최종 accuracy: {final_acc*100:.2f}%")

    return {
        'method': 'pipelined_dsp',
        'pruning_ratio': pruning_ratio,
        'base_accuracy': base_acc,
        'pruned_before_ft': pruned_acc_before,
        'final_accuracy': final_acc,
        'best_accuracy': best_acc,
    }


# ============================================================================
# 메인 실험
# ============================================================================

def main():
    """메인 실험 실행"""
    print("\n" + "=" * 80)
    print("SST-2 Joint vs Pipelined DSP 비교 실험")
    print("=" * 80)

    set_seed(42)

    # Configuration
    pruning_ratios = [0.1, 0.3, 0.5]  # 10%, 30%, 50%
    batch_size = 32

    # Colab에서는 샘플 수를 제한 (빠른 실험)
    max_train_samples = 5000 if IN_COLAB else None

    if max_train_samples:
        print(f"\n⚠️ 빠른 실험을 위해 {max_train_samples}개 샘플만 사용")
        print("전체 데이터로 실행하려면 max_train_samples = None으로 설정하세요\n")

    # 데이터 준비
    tokenizer, train_loader, eval_loader = prepare_sst2(
        batch_size=batch_size,
        max_samples=max_train_samples
    )

    # 결과 저장
    all_results = {
        'config': {
            'task': 'sst2',
            'model': 'bert-base-uncased',
            'device': str(DEVICE),
            'pruning_ratios': pruning_ratios,
        },
        'joint_dsp': [],
        'pipelined_dsp': [],
    }

    # Joint DSP 실험
    print("\n\n" + "=" * 80)
    print("JOINT DSP 실험")
    print("=" * 80)

    for ratio in pruning_ratios:
        try:
            result = run_joint_dsp(
                pruning_ratio=ratio,
                train_loader=train_loader,
                eval_loader=eval_loader,
                num_epochs=2 if IN_COLAB else 3,  # Colab에서는 epoch 줄임
                use_ste=False,
                log_interval=200,
            )
            all_results['joint_dsp'].append(result)
        except Exception as e:
            print(f"❌ Joint DSP 에러 (ratio={ratio}): {e}")

    # Pipelined DSP 실험
    print("\n\n" + "=" * 80)
    print("PIPELINED DSP 실험")
    print("=" * 80)

    for ratio in pruning_ratios:
        try:
            result = run_pipelined_dsp(
                pruning_ratio=ratio,
                train_loader=train_loader,
                eval_loader=eval_loader,
                num_epochs_base=1,
                num_epochs_ft=2 if IN_COLAB else 3,
                log_interval=200,
            )
            all_results['pipelined_dsp'].append(result)
        except Exception as e:
            print(f"❌ Pipelined DSP 에러 (ratio={ratio}): {e}")

    # 최종 요약
    print("\n\n" + "=" * 80)
    print("📊 최종 결과 요약")
    print("=" * 80)

    print(f"\nTask: SST-2")
    print(f"Model: BERT-base (144 heads)")

    print("\n" + "-" * 80)
    print("Joint DSP:")
    print("-" * 80)
    print(f"{'Prune %':>10} {'Final Acc':>12} {'Best Acc':>12}")
    print("-" * 80)
    for r in all_results['joint_dsp']:
        print(f"{int(r['pruning_ratio']*100):>10}% "
              f"{r['final_accuracy']*100:>11.2f}% "
              f"{r['best_accuracy']*100:>11.2f}%")

    print("\n" + "-" * 80)
    print("Pipelined DSP:")
    print("-" * 80)
    print(f"{'Prune %':>10} {'Base':>8} {'Pruned':>10} {'Final':>10}")
    print("-" * 80)
    for r in all_results['pipelined_dsp']:
        print(f"{int(r['pruning_ratio']*100):>10}% "
              f"{r['base_accuracy']*100:>7.2f}% "
              f"{r['pruned_before_ft']*100:>9.2f}% "
              f"{r['final_accuracy']*100:>9.2f}%")

    # 결과 저장
    if IN_COLAB:
        output_file = "sst2_results.json"
    else:
        os.makedirs("results", exist_ok=True)
        output_file = "results/sst2_results.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ 결과 저장됨: {output_file}")
    print("\n실험 완료! 🎉")


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    main()
