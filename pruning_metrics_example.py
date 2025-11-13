"""
Usage examples for pruning metrics module.

This demonstrates how to use the core DSP techniques.
"""

import torch
from pruning_metrics import (
    # DSP core
    gumbel_soft_top_k,
    STEFunction,
    TemperatureScheduler,
    DSPHeadSelector,
    # L0 regularization
    ConcreteGate,
    # Head importance
    HeadImportanceMetric,
    # Utilities
    SparsityMetric,
    PruningMetricTracker,
    convert_gate_to_mask,
    print_head_mask,
)


def example_dsp_basic():
    """
    Example 1: Basic DSP usage with Gumbel-Softmax
    """
    print("=" * 80)
    print("Example 1: Differentiable Subset Pruning (DSP)")
    print("=" * 80)

    # Setup
    n_layers = 12
    n_heads = 12
    total_heads = n_layers * n_heads  # 144 heads
    num_to_keep = 48  # Keep 48/144 = 33% of heads

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DSP head selector
    selector = DSPHeadSelector(
        n_layers=n_layers,
        n_heads=n_heads,
        num_heads_to_keep=num_to_keep,
        device=device,
        use_ste=False  # Use Gumbel-Softmax
    )

    # Create temperature scheduler
    temp_scheduler = TemperatureScheduler(
        initial_temperature=1000.0,
        final_temperature=1e-8,
        cooldown_steps=25000
    )

    # Simulate training loop
    print(f"\nTraining with DSP (keeping {num_to_keep}/{total_heads} heads)...")

    for step in [0, 1000, 5000, 10000, 20000, 25000]:
        temperature = temp_scheduler.get_temperature(step)
        mask = selector.get_mask(temperature=temperature)

        stats = SparsityMetric.get_stats(mask)

        print(f"\nStep {step:5d}:")
        print(f"  Temperature: {temperature:.2e}")
        print(f"  Mask sum: {mask.sum():.2f} (target: {num_to_keep})")
        print(f"  Sparsity: {stats['sparsity_pct']:.1f}%")

    # Final mask
    print("\n" + "=" * 80)
    print("Final learned mask:")
    print("=" * 80)
    final_mask = convert_gate_to_mask(selector.get_importance_weights(), num_to_keep)
    print_head_mask(final_mask)


def example_ste():
    """
    Example 2: Using Straight-Through Estimator
    """
    print("\n\n" + "=" * 80)
    print("Example 2: Straight-Through Estimator (STE)")
    print("=" * 80)

    # Setup
    n_layers = 12
    n_heads = 12
    num_to_keep = 48

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DSP with STE
    selector = DSPHeadSelector(
        n_layers=n_layers,
        n_heads=n_heads,
        num_heads_to_keep=num_to_keep,
        device=device,
        use_ste=True  # Use STE instead of Gumbel
    )

    print(f"\nUsing STE (hard selection in forward, soft in backward)...")

    # Get mask (no temperature needed for STE)
    mask = selector.get_mask()

    print(f"\nMask statistics:")
    print(f"  Unique values: {mask.unique().tolist()}")
    print(f"  Mask sum: {mask.sum():.0f} (target: {num_to_keep})")

    stats = SparsityMetric.get_stats(mask)
    print(f"  Sparsity: {stats['sparsity_pct']:.1f}%")


def example_l0_regularization():
    """
    Example 3: L0 regularization (Voita et al.)
    """
    print("\n\n" + "=" * 80)
    print("Example 3: L0 Regularization")
    print("=" * 80)

    # Create concrete gate for one layer
    n_heads = 12
    gate = ConcreteGate(
        shape=[1, n_heads, 1, 1],
        temperature=0.33,
        stretch_limits=(-0.1, 1.1),
        l0_penalty=0.5,
    )

    print(f"\nL0 penalty coefficient: {gate.l0_penalty}")

    # Simulate training
    print("\nSimulated training steps:")

    for step in [0, 100, 500, 1000]:
        # Get gates (training mode - with noise)
        gates = gate.get_gates(is_train=True)

        # Compute penalty
        penalty = gate.get_penalty()

        # Sparsity
        sparsity = gate.get_sparsity_rate()

        print(f"\nStep {step:4d}:")
        print(f"  Gate values (mean): {gates.mean():.3f}")
        print(f"  L0 penalty: {penalty:.3f}")
        print(f"  Sparsity rate: {sparsity * 100:.1f}%")

        # Simulate parameter update (just for demo)
        with torch.no_grad():
            gate.log_a -= 0.1  # Decrease log_a to increase sparsity


def example_head_importance():
    """
    Example 4: Computing head importance (Michel et al.)

    Note: This requires an actual model, so we'll just show the setup.
    """
    print("\n\n" + "=" * 80)
    print("Example 4: Head Importance Computation")
    print("=" * 80)

    n_layers = 12
    n_heads = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create importance metric computer
    importance_metric = HeadImportanceMetric(
        n_layers=n_layers,
        n_heads=n_heads,
        device=device,
        normalize_by_layer=True,
        normalize_global=True,
    )

    print("\nSetup for head importance computation:")
    print(f"  Layers: {n_layers}")
    print(f"  Heads per layer: {n_heads}")
    print(f"  Total heads: {n_layers * n_heads}")
    print(f"  Layer normalization: {importance_metric.normalize_by_layer}")
    print(f"  Global normalization: {importance_metric.normalize_global}")

    print("\nUsage:")
    print("  importance = importance_metric.compute(model, dataloader)")
    print("  # Returns tensor of shape (n_layers, n_heads)")


def example_metric_tracking():
    """
    Example 5: Tracking metrics during pruning
    """
    print("\n\n" + "=" * 80)
    print("Example 5: Metric Tracking")
    print("=" * 80)

    # Create tracker
    tracker = PruningMetricTracker()

    # Simulate pruning process
    n_layers = 12
    n_heads = 12

    print("\nSimulating pruning process...")

    for step, (keep_ratio, perf) in enumerate([
        (1.0, 0.92),   # No pruning
        (0.8, 0.91),   # 20% pruned
        (0.6, 0.90),   # 40% pruned
        (0.4, 0.87),   # 60% pruned
        (0.2, 0.80),   # 80% pruned
    ]):
        # Create mask
        num_to_keep = int(n_layers * n_heads * keep_ratio)
        mask = torch.zeros(n_layers, n_heads)
        mask.view(-1)[:num_to_keep] = 1.0

        # Track metrics
        tracker.update(
            step=step * 1000,
            mask=mask,
            performance=perf,
            temperature=1000.0 / (step + 1)
        )

    # Show history
    print("\nPruning history:")
    history = tracker.get_history()

    for i in range(len(history["step"])):
        print(f"\nStep {history['step'][i]:5d}:")
        print(f"  Remaining heads: {history['remaining_heads'][i]:.0f}")
        print(f"  Sparsity: {history['sparsity'][i]:.1f}%")
        print(f"  Performance: {history['performance'][i]:.1f}%")
        print(f"  Temperature: {history['temperature'][i]:.2e}")

    # Best result
    print("\n" + "=" * 80)
    print("Best result:")
    print("=" * 80)
    best = tracker.get_best()
    for key, value in best.items():
        print(f"  {key}: {value}")


def example_comparison():
    """
    Example 6: Comparing DSP methods
    """
    print("\n\n" + "=" * 80)
    print("Example 6: Comparing DSP vs STE vs L0")
    print("=" * 80)

    n_layers = 12
    n_heads = 12
    num_to_keep = 48

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Method 1: DSP with Gumbel-Softmax
    print("\n1. DSP with Gumbel-Softmax:")
    dsp_selector = DSPHeadSelector(
        n_layers=n_layers,
        n_heads=n_heads,
        num_heads_to_keep=num_to_keep,
        device=device,
        use_ste=False
    )
    dsp_mask = dsp_selector.get_mask(temperature=1.0)
    print(f"   Mask sum: {dsp_mask.sum():.2f}")
    print(f"   Differentiable: Yes (soft)")

    # Method 2: DSP with STE
    print("\n2. DSP with STE:")
    ste_selector = DSPHeadSelector(
        n_layers=n_layers,
        n_heads=n_heads,
        num_heads_to_keep=num_to_keep,
        device=device,
        use_ste=True
    )
    ste_mask = ste_selector.get_mask()
    print(f"   Mask sum: {ste_mask.sum():.0f}")
    print(f"   Differentiable: Yes (straight-through)")
    print(f"   Binary: {len(ste_mask.unique()) == 2}")

    # Method 3: L0 regularization
    print("\n3. L0 Regularization:")
    print(f"   Uses soft gates with penalty on expected L0 norm")
    print(f"   No hard constraint on number of heads")
    print(f"   Sparsity controlled by penalty coefficient")


if __name__ == "__main__":
    # Run all examples
    example_dsp_basic()
    example_ste()
    example_l0_regularization()
    example_head_importance()
    example_metric_tracking()
    example_comparison()

    print("\n\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
