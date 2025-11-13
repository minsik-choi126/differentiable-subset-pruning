# Pruning Metrics for Differentiable Subset Pruning

This module provides a clean, well-documented implementation of core pruning techniques from recent research papers.

## Overview

The `pruning_metrics.py` module implements three major pruning approaches:

1. **Differentiable Subset Pruning (DSP)** - Li et al., TACL 2021
2. **L0 Regularization** - Voita et al., ACL 2019
3. **Gradient-based Head Importance** - Michel et al., NeurIPS 2019

## Differentiable Subset Pruning (DSP)

DSP allows end-to-end learning of which attention heads to keep by making the top-k operation differentiable.

### Key Components

#### 1. Gumbel-Softmax Top-K Selection

```python
from pruning_metrics import gumbel_soft_top_k

# Importance weights for all heads
w = torch.randn(144)  # e.g., 12 layers × 12 heads

# Select top 48 heads with soft selection
temperature = 1.0
mask = gumbel_soft_top_k(w, k=48, temperature=temperature)

# mask.sum() ≈ 48 (soft, differentiable)
```

**How it works:**
- Adds Gumbel noise to importance weights
- Applies sequential softmax to select top-k
- Temperature controls soft vs hard selection
  - High temp (e.g., 1000) → very soft
  - Low temp (e.g., 1e-8) → nearly hard

#### 2. Straight-Through Estimator (STE)

Alternative to Gumbel-Softmax with hard forward, soft backward:

```python
from pruning_metrics import STEFunction

# Forward: hard top-k selection
mask = STEFunction.apply(w, k=48)

# mask contains exactly 48 ones (hard, binary)
# Backward: gradients pass through unchanged
```

#### 3. Temperature Annealing

```python
from pruning_metrics import TemperatureScheduler

scheduler = TemperatureScheduler(
    initial_temperature=1000.0,
    final_temperature=1e-8,
    cooldown_steps=25000
)

for step in range(training_steps):
    temperature = scheduler.get_temperature(step)
    mask = gumbel_soft_top_k(w, k=48, temperature=temperature)
    # ... training ...
```

Exponentially decays: `T(t) = T_0 * exp(-t/τ * log(T_0/T_f))`

#### 4. Complete DSP Head Selector

High-level interface combining all DSP components:

```python
from pruning_metrics import DSPHeadSelector

selector = DSPHeadSelector(
    n_layers=12,
    n_heads=12,
    num_heads_to_keep=48,
    device=device,
    use_ste=False  # or True for STE
)

# Get mask (soft or hard depending on use_ste)
mask = selector.get_mask(temperature=temperature)

# Access learned importance weights
importance = selector.get_importance_weights()
```

## L0 Regularization

Uses concrete distribution for soft binary gates with L0 penalty.

```python
from pruning_metrics import ConcreteGate

# Create gate for one layer's heads
gate = ConcreteGate(
    shape=[1, 12, 1, 1],  # Broadcastable shape
    temperature=0.33,
    stretch_limits=(-0.1, 1.1),
    l0_penalty=0.5
)

# Apply gate to attention probabilities
gated_attention = gate(attention_probs, is_train=True)

# Get L0 penalty for loss
penalty = gate.get_penalty()
loss = task_loss + penalty

# Check sparsity
sparsity_rate = gate.get_sparsity_rate()
```

**Key difference from DSP:**
- No hard constraint on number of heads
- Sparsity controlled by penalty coefficient
- Each head has independent gate

## Head Importance Estimation

Gradient-based importance from Michel et al.:

```python
from pruning_metrics import HeadImportanceMetric

metric = HeadImportanceMetric(
    n_layers=12,
    n_heads=12,
    device=device,
    normalize_by_layer=True,
    normalize_global=True
)

# Compute importance scores
importance = metric.compute(model, dataloader)
# Returns: tensor of shape (12, 12)

# Use for pruning decision
threshold = importance.quantile(0.5)
mask = (importance > threshold).float()
```

**How it works:**
- Measures `|∂L/∂m_h|` where `m_h` is head mask
- Higher importance = bigger impact on loss
- Typically used for one-shot pruning

## Evaluation Metrics

### Sparsity Metrics

```python
from pruning_metrics import SparsityMetric

mask = torch.randint(0, 2, (12, 12))

# Simple metrics
sparsity = SparsityMetric.compute_sparsity(mask)  # % pruned
remaining = SparsityMetric.compute_remaining_ratio(mask)  # % kept

# Detailed stats
stats = SparsityMetric.get_stats(mask)
# Returns: {
#   'total_heads': 144,
#   'remaining_heads': 72,
#   'pruned_heads': 72,
#   'sparsity_pct': 50.0,
#   'remaining_pct': 50.0
# }
```

### Metric Tracking

Track metrics throughout training:

```python
from pruning_metrics import PruningMetricTracker

tracker = PruningMetricTracker()

for step in training_loop:
    mask = get_current_mask()
    perf = evaluate(model)
    temp = temperature_scheduler.get_temperature(step)

    tracker.update(step, mask, perf, temp)

# Retrieve history
history = tracker.get_history()
# Returns: {'step': [...], 'sparsity': [...], 'performance': [...], ...}

# Find best result
best = tracker.get_best()
# Returns: {'best_step': ..., 'best_performance': ..., 'best_sparsity': ...}
```

## Utilities

### Convert Gates to Masks

```python
from pruning_metrics import convert_gate_to_mask

# Top-k conversion
gates = torch.rand(12, 12)
mask = convert_gate_to_mask(gates, num_of_heads=48)

# Threshold conversion
mask = convert_gate_to_mask(gates, threshold=0.5)
```

### Print Head Mask

```python
from pruning_metrics import print_head_mask

mask = torch.randint(0, 2, (12, 12))
print_head_mask(mask)

# Output:
# Layer/Head >    1       2       3       ...
# Layer 1:        1.0000  0.0000  1.0000  ...
# Layer 2:        0.0000  1.0000  1.0000  ...
# ...
# Remaining heads: 72/144 (50.0%)
```

## Method Comparison

| Method | Hard Constraint | Differentiable | Forward Pass | Best For |
|--------|----------------|----------------|--------------|----------|
| **DSP (Gumbel)** | Yes (k heads) | Yes (soft) | Soft selection | End-to-end training with exact k constraint |
| **DSP (STE)** | Yes (k heads) | Yes (straight-through) | Hard selection | Faster convergence, binary masks |
| **L0 Regularization** | No | Yes | Soft gates | Flexible sparsity, continuous relaxation |
| **Head Importance** | No | N/A | Not applicable | One-shot pruning, analysis |

## Usage Examples

See `pruning_metrics_example.py` for complete examples:

```bash
python pruning_metrics_example.py
```

Examples include:
1. Basic DSP with Gumbel-Softmax
2. Using Straight-Through Estimator
3. L0 regularization
4. Head importance computation
5. Metric tracking
6. Method comparison

## References

1. **Differentiable Subset Pruning of Transformer Heads**
   Jiaoda Li, Ryan Cotterell, Mrinmaya Sachan
   TACL 2021
   https://arxiv.org/abs/2108.04657

2. **Are Sixteen Heads Really Better than One?**
   Paul Michel, Omer Levy, Graham Neubig
   NeurIPS 2019
   https://arxiv.org/abs/1905.10650

3. **Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned**
   Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, Ivan Titov
   ACL 2019
   https://arxiv.org/abs/1905.09418

4. **Learning Sparse Neural Networks through L0 Regularization**
   Christos Louizos, Max Welling, Diederik P. Kingma
   ICLR 2018
   https://arxiv.org/abs/1712.01312

## Implementation Notes

- All methods support both CPU and GPU
- Uses `torch.double` for numerical stability in Gumbel operations
- Temperature annealing uses exponential decay
- L0 gates use stretched concrete distribution
- Head importance uses gradient accumulation over dataset
