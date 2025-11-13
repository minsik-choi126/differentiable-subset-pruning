"""
DSP (Differentiable Subset Pruning) - 실제 동작 시연

이 스크립트는 DSP가 어떻게 작동하는지 단계별로 보여줍니다.
"""

import torch
import torch.nn as nn
import numpy as np
from pruning_metrics import (
    gumbel_soft_top_k,
    STEFunction,
    TemperatureScheduler,
    convert_gate_to_mask,
    print_head_mask,
    SparsityMetric,
)


def demo_1_mask_generation():
    """
    Demo 1: Mask 생성 과정 (Gumbel vs STE)
    """
    print("=" * 80)
    print("Demo 1: Mask 생성 - Gumbel-Softmax vs STE")
    print("=" * 80)

    # Setup
    n_layers = 12
    n_heads = 12
    total_heads = n_layers * n_heads  # 144
    num_to_keep = 48

    # Importance weights (학습 중이라고 가정)
    # 일부러 불균등하게 초기화
    torch.manual_seed(42)
    w = torch.randn(n_layers, n_heads).double()

    print(f"\n현재 importance weights (w):")
    print(f"Shape: {w.shape}")
    print(f"Min: {w.min():.3f}, Max: {w.max():.3f}, Mean: {w.mean():.3f}")
    print(f"\nLayer 0 weights: {w[0].numpy()}")

    # Method 1: Gumbel-Softmax with different temperatures
    print("\n" + "-" * 80)
    print("Method 1: Gumbel-Softmax (다양한 temperature)")
    print("-" * 80)

    for temp in [1000.0, 10.0, 1.0, 0.01]:
        mask = gumbel_soft_top_k(w.view(-1), num_to_keep, temp).view_as(w)

        print(f"\nTemperature = {temp:.2e}:")
        print(f"  Mask sum: {mask.sum():.2f} (target: {num_to_keep})")
        print(f"  Mask range: [{mask.min():.4f}, {mask.max():.4f}]")
        print(f"  Layer 0 mask: {mask[0][:6].numpy()}")  # 처음 6개만

        # Soft vs Hard 측정
        binary_mask = (mask > 0.5).float()
        print(f"  Binarized (>0.5) sum: {binary_mask.sum():.0f}")

    # Method 2: STE
    print("\n" + "-" * 80)
    print("Method 2: Straight-Through Estimator")
    print("-" * 80)

    mask_ste = STEFunction.apply(w.view(-1), num_to_keep).view_as(w)

    print(f"\nMask sum: {mask_ste.sum():.0f} (target: {num_to_keep})")
    print(f"Unique values: {mask_ste.unique().tolist()}")
    print(f"Layer 0 mask: {mask_ste[0][:6].numpy()}")


def demo_2_attention_pruning():
    """
    Demo 2: Attention에 mask 적용하여 pruning
    """
    print("\n\n" + "=" * 80)
    print("Demo 2: Attention Pruning 시뮬레이션")
    print("=" * 80)

    batch_size = 2
    seq_len = 4
    n_heads = 4
    d_k = 8

    # Dummy attention probabilities
    torch.manual_seed(42)
    attention_probs = torch.softmax(
        torch.randn(batch_size, n_heads, seq_len, seq_len),
        dim=-1
    )

    print(f"\nOriginal attention_probs shape: {attention_probs.shape}")
    print(f"\nHead 0 attention (batch 0):")
    print(attention_probs[0, 0].numpy())

    # Mask (head 0, 2만 keep, head 1, 3은 prune)
    head_mask = torch.tensor([1.0, 0.0, 1.0, 0.0]).view(1, 4, 1, 1)

    print(f"\nHead mask: {head_mask.view(-1).tolist()}")
    print(f"  → Head 0, 2: KEEP (×1)")
    print(f"  → Head 1, 3: PRUNE (×0)")

    # Apply mask
    masked_attention_probs = attention_probs * head_mask

    print(f"\nAfter masking:")
    print(f"Head 0 (kept):")
    print(masked_attention_probs[0, 0].numpy())
    print(f"\nHead 1 (pruned):")
    print(masked_attention_probs[0, 1].numpy())

    # Context 계산 (value는 dummy)
    value = torch.randn(batch_size, n_heads, seq_len, d_k)

    context_original = torch.matmul(attention_probs, value)
    context_masked = torch.matmul(masked_attention_probs, value)

    print(f"\nContext shape: {context_original.shape}")
    print(f"Context difference (head 1이 pruned됨):")
    diff = (context_original - context_masked).abs()
    print(f"  Head 0 (kept): {diff[0, 0].mean():.6f} (거의 0)")
    print(f"  Head 1 (pruned): {diff[0, 1].mean():.6f} (큰 차이)")


def demo_3_gradient_flow():
    """
    Demo 3: Gradient가 w로 흐르는 과정
    """
    print("\n\n" + "=" * 80)
    print("Demo 3: Gradient Flow (w 학습 과정)")
    print("=" * 80)

    # Setup
    n_heads = 4
    num_to_keep = 2

    # Importance weights (learnable!)
    w = nn.Parameter(torch.randn(n_heads).double())
    print(f"\nInitial w: {w.data.numpy()}")

    # Optimizer
    optimizer = torch.optim.SGD([w], lr=0.1)

    # Simulate training
    print("\n" + "-" * 80)
    print("Simulated training (5 steps)")
    print("-" * 80)

    for step in range(5):
        # Generate mask
        temperature = 1.0
        mask = gumbel_soft_top_k(w, num_to_keep, temperature)

        # Dummy forward pass
        # 가정: head 0, 1은 task에 중요, head 2, 3은 불필요
        # Task loss는 중요한 head가 pruning되면 증가
        task_importance = torch.tensor([1.0, 1.0, 0.0, 0.0])
        dummy_loss = -torch.sum(mask * task_importance)  # 중요한 head를 keep해야 loss 감소

        # Backward
        optimizer.zero_grad()
        dummy_loss.backward()

        print(f"\nStep {step}:")
        print(f"  w: {w.data.numpy()}")
        print(f"  mask: {mask.detach().numpy()}")
        print(f"  loss: {dummy_loss.item():.4f}")
        print(f"  grad: {w.grad.numpy()}")

        # Update
        optimizer.step()

    print(f"\nFinal w: {w.data.numpy()}")
    print(f"→ Head 0, 1의 importance↑ (중요)")
    print(f"→ Head 2, 3의 importance↓ (불필요)")


def demo_4_temperature_annealing():
    """
    Demo 4: Temperature annealing 효과
    """
    print("\n\n" + "=" * 80)
    print("Demo 4: Temperature Annealing")
    print("=" * 80)

    # Setup
    n_heads = 12
    num_to_keep = 4

    w = torch.randn(n_heads).double()
    scheduler = TemperatureScheduler(
        initial_temperature=100.0,
        final_temperature=0.01,
        cooldown_steps=1000
    )

    print(f"\nImportance weights: {w.numpy()}")
    print(f"Top-{num_to_keep} indices: {w.argsort(descending=True)[:num_to_keep].tolist()}")

    print("\n" + "-" * 80)
    print("Mask evolution with temperature annealing:")
    print("-" * 80)

    for step in [0, 100, 300, 500, 800, 1000]:
        temp = scheduler.get_temperature(step)
        mask = gumbel_soft_top_k(w, num_to_keep, temp)

        print(f"\nStep {step:4d}, T={temp:7.4f}:")
        print(f"  Mask: {mask.numpy()}")
        print(f"  Sum: {mask.sum():.2f}")

        # Entropy (얼마나 uncertain한지)
        probs = mask / mask.sum()
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        print(f"  Entropy: {entropy:.4f} (낮을수록 hard)")


def demo_5_full_training_simulation():
    """
    Demo 5: 전체 Training 시뮬레이션
    """
    print("\n\n" + "=" * 80)
    print("Demo 5: Full Training Simulation")
    print("=" * 80)

    # Setup
    n_layers = 3
    n_heads = 4
    total_heads = n_layers * n_heads  # 12
    num_to_keep = 6  # 50% pruning

    # Model의 importance weights
    w = nn.Parameter(torch.randn(n_layers, n_heads).double())

    # Optimizer
    optimizer = torch.optim.Adam([w], lr=0.5)

    # Temperature scheduler
    temp_scheduler = TemperatureScheduler(
        initial_temperature=10.0,
        final_temperature=0.01,
        cooldown_steps=50
    )

    print(f"\nSetup:")
    print(f"  Total heads: {total_heads}")
    print(f"  Keep: {num_to_keep} ({num_to_keep/total_heads*100:.0f}%)")
    print(f"  Prune: {total_heads - num_to_keep} ({(1-num_to_keep/total_heads)*100:.0f}%)")

    # Simulate task importance (ground truth)
    # 가정: 각 layer의 첫 2개 head만 중요
    true_importance = torch.zeros(n_layers, n_heads)
    true_importance[:, :2] = 1.0

    print(f"\nGround truth important heads:")
    print_head_mask(true_importance)

    # Training loop
    print("\n" + "-" * 80)
    print("Training...")
    print("-" * 80)

    history = []

    for step in range(100):
        # Get temperature
        temp = temp_scheduler.get_temperature(step)

        # Generate mask
        mask = gumbel_soft_top_k(w.view(-1), num_to_keep, temp).view_as(w)

        # Dummy loss: task에 중요한 head를 keep해야 낮아짐
        loss = -torch.sum(mask * true_importance)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log
        if step % 20 == 0 or step == 99:
            print(f"\nStep {step:3d}:")
            print(f"  Temperature: {temp:.4f}")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Mask sum: {mask.sum():.2f}")

            # Check if learning correct heads
            hard_mask = convert_gate_to_mask(w, num_to_keep)
            overlap = (hard_mask * true_importance).sum()
            print(f"  Overlap with ground truth: {overlap:.0f}/{true_importance.sum():.0f}")

        history.append({
            'step': step,
            'loss': loss.item(),
            'temperature': temp,
            'mask_sum': mask.sum().item()
        })

    # Final result
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    final_mask = convert_gate_to_mask(w, num_to_keep)

    print(f"\nLearned importance weights:")
    print(w.data.numpy())

    print(f"\nFinal mask:")
    print_head_mask(final_mask)

    print(f"\nGround truth:")
    print_head_mask(true_importance)

    # Accuracy
    overlap = (final_mask * true_importance).sum()
    precision = overlap / final_mask.sum()
    recall = overlap / true_importance.sum()

    print(f"\nResults:")
    print(f"  Precision: {precision:.2%} ({overlap:.0f}/{final_mask.sum():.0f})")
    print(f"  Recall: {recall:.2%} ({overlap:.0f}/{true_importance.sum():.0f})")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DSP 작동 원리 - 실전 데모" + " " * 22 + "║")
    print("╚" + "=" * 78 + "╝")

    demo_1_mask_generation()
    demo_2_attention_pruning()
    demo_3_gradient_flow()
    demo_4_temperature_annealing()
    demo_5_full_training_simulation()

    print("\n\n" + "=" * 80)
    print("모든 데모 완료!")
    print("=" * 80)
    print("\n핵심 요약:")
    print("1. w (importance weights)는 학습 가능한 파라미터")
    print("2. 매 forward마다 w로부터 mask 생성 (Gumbel 또는 STE)")
    print("3. Mask를 attention probabilities에 곱해서 pruning")
    print("4. Task loss로 w를 학습 → 중요한 head는 w↑, 불필요한 head는 w↓")
    print("5. Temperature annealing으로 soft → hard selection")
    print("6. 최종적으로 top-k head만 선택하는 hard mask 획득")
    print("=" * 80)
