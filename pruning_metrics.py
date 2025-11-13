"""
Pruning Metrics and Techniques for Differentiable Subset Pruning

This module implements the core techniques from:
- "Differentiable Subset Pruning of Transformer Heads" (Li et al., TACL 2021)
- "Are Sixteen Heads Really Better than One?" (Michel et al., 2019)
- "Analyzing Multi-Head Self-Attention" (Voita et al., 2019)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm


# ============================================================================
# DSP: Differentiable Subset Pruning (Li et al., TACL 2021)
# ============================================================================

EPSILON = torch.finfo(torch.double).tiny


def gumbel_soft_top_k(w: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
    """
    Differentiable top-k selection using Gumbel-Softmax trick.

    This is the core of Differentiable Subset Pruning (DSP).
    It allows end-to-end learning of which k heads to keep by making
    the top-k operation differentiable.

    Args:
        w: Importance weights of shape (n_total_heads,)
        k: Number of heads to select (keep)
        temperature: Temperature parameter for Gumbel-Softmax
                    Lower temperature -> closer to hard selection

    Returns:
        Soft mask of shape (n_total_heads,) with sum ≈ k

    Reference:
        Li et al., "Differentiable Subset Pruning of Transformer Heads", TACL 2021
        Section 3.2: Differentiable Subset Pruning
    """
    # Sample Gumbel noise
    u = torch.rand_like(w) * (1 - EPSILON) + EPSILON
    gumbel_noise = -torch.log(-torch.log(u))

    # Add Gumbel noise to importance weights
    r = gumbel_noise + w
    epsilon = torch.ones_like(r) * EPSILON

    # Iteratively compute soft top-k using sequential softmax
    p = torch.zeros([k, w.size()[0]]).to(w.device).double()

    # First selection
    p[0] = torch.exp(nn.functional.log_softmax(r / temperature, 0))

    # Subsequent selections (remove probability mass from already selected)
    for j in range(1, k):
        r = r + torch.log(torch.max(1 - p[j - 1], epsilon))
        p[j] = torch.exp(nn.functional.log_softmax(r / temperature, 0))

    # Sum probabilities across k selections
    return p.sum(0)


class STEFunction(torch.autograd.Function):
    """
    Straight-Through Estimator for hard top-k selection.

    Forward: Hard selection (discrete)
    Backward: Straight-through gradient (identity)

    This provides an alternative to Gumbel-Softmax that uses
    hard selection in forward pass but allows gradients to flow.

    Reference:
        Bengio et al., "Estimating or Propagating Gradients Through
        Stochastic Neurons for Conditional Computation", 2013
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, k: int) -> torch.Tensor:
        """
        Select top-k elements (hard selection).

        Args:
            input: Importance weights
            k: Number of elements to select

        Returns:
            Binary mask with k ones
        """
        threshold = input.sort(descending=True)[0][k]
        return (input > threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Pass gradients through unchanged (straight-through).
        """
        return grad_output, None


class TemperatureScheduler:
    """
    Temperature annealing scheduler for DSP.

    Exponentially decays temperature from initial to final value
    over a specified number of steps.

    Reference:
        Li et al., TACL 2021, Section 4.1: Training Details
    """

    def __init__(
        self,
        initial_temperature: float = 1000.0,
        final_temperature: float = 1e-8,
        cooldown_steps: int = 25000
    ):
        """
        Args:
            initial_temperature: Starting temperature (high for soft selection)
            final_temperature: Ending temperature (low for hard selection)
            cooldown_steps: Number of steps to anneal over
        """
        self.initial_temp = initial_temperature
        self.final_temp = final_temperature
        self.cooldown_steps = cooldown_steps

        # Precompute log values for efficiency
        self.log_initial = np.log(initial_temperature)
        self.log_final = np.log(final_temperature)

    def get_temperature(self, step: int) -> float:
        """
        Get temperature at given training step.

        Args:
            step: Current training step

        Returns:
            Temperature value
        """
        if step >= self.cooldown_steps:
            return self.final_temp

        # Exponential decay: T(t) = T_0 * exp(-t/τ * log(T_0/T_f))
        progress = step / self.cooldown_steps
        log_temp = self.log_initial - progress * (self.log_initial - self.log_final)

        return np.exp(log_temp)


class DSPHeadSelector:
    """
    Differentiable Subset Pruning head selector.

    Maintains learnable importance weights and applies differentiable
    top-k selection to determine which heads to keep.
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        num_heads_to_keep: int,
        device: torch.device,
        use_ste: bool = False
    ):
        """
        Args:
            n_layers: Number of transformer layers
            n_heads: Number of attention heads per layer
            num_heads_to_keep: Number of heads to keep (k)
            device: Device for computation
            use_ste: Whether to use Straight-Through Estimator
        """
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.num_heads_to_keep = num_heads_to_keep
        self.device = device
        self.use_ste = use_ste

        # Learnable importance weights (w in the paper)
        self.w = nn.Parameter(
            torch.empty([n_layers, n_heads], device=device).double()
        )
        nn.init.xavier_uniform_(self.w)

    def get_mask(self, temperature: Optional[float] = None) -> torch.Tensor:
        """
        Compute head selection mask using DSP.

        Args:
            temperature: Temperature for Gumbel-Softmax (ignored if use_ste=True)

        Returns:
            Mask of shape (n_layers, n_heads)
        """
        if self.use_ste:
            # Straight-through estimator (hard selection)
            mask = STEFunction.apply(
                self.w.view(-1),
                self.num_heads_to_keep
            ).view_as(self.w)
        else:
            # Gumbel-Softmax (soft selection)
            if temperature is None:
                raise ValueError("Temperature required for Gumbel-Softmax")

            mask = gumbel_soft_top_k(
                self.w.view(-1),
                self.num_heads_to_keep,
                temperature
            ).view_as(self.w)

        return mask

    def get_importance_weights(self) -> torch.Tensor:
        """Get current importance weights."""
        return self.w.detach()


# ============================================================================
# L0 Regularization (Voita et al., ACL 2019)
# ============================================================================

class ConcreteGate(nn.Module):
    """
    Concrete (continuous relaxation) gate for L0 regularization.

    Uses stretched concrete distribution to create differentiable gates
    that can be trained to be approximately binary (0 or 1).

    Reference:
        Voita et al., "Analyzing Multi-Head Self-Attention", ACL 2019
        Louizos et al., "Learning Sparse Neural Networks through L0 Regularization", ICLR 2018
    """

    def __init__(
        self,
        shape: list,
        temperature: float = 0.33,
        stretch_limits: Tuple[float, float] = (-0.1, 1.1),
        l0_penalty: float = 1.0,
        eps: float = 1e-6
    ):
        """
        Args:
            shape: Shape of gate variable (can be broadcasted)
            temperature: Concrete sigmoid temperature (lower = more discrete)
            stretch_limits: (min, max) value before clipping to [0, 1]
            l0_penalty: Coefficient for L0 regularization
            eps: Small value to avoid NaNs
        """
        super().__init__()
        self.temperature = temperature
        self.stretch_limits = stretch_limits
        self.l0_penalty = l0_penalty
        self.eps = eps

        # Learnable gate parameters (log α in the paper)
        self.log_a = nn.Parameter(torch.empty(shape))
        nn.init.xavier_uniform_(self.log_a)

    def forward(self, values: torch.Tensor, is_train: Optional[bool] = None):
        """
        Apply gate to values.

        Args:
            values: Tensor to gate
            is_train: Whether in training mode (for sampling)

        Returns:
            Gated values
        """
        is_train = self.training if is_train is None else is_train
        gates = self.get_gates(is_train)
        return values * gates

    def get_gates(self, is_train: bool) -> torch.Tensor:
        """
        Sample gate activations in [0, 1] interval.

        Args:
            is_train: Whether to add noise (training) or use expectation

        Returns:
            Gate values
        """
        low, high = self.stretch_limits

        if is_train:
            # Sample from concrete distribution
            shape = self.log_a.size()
            noise = (1 - 2 * self.eps) * torch.rand(shape).to(self.log_a.device) + self.eps

            # Gumbel-Softmax trick for binary concrete
            logit = (torch.log(noise) - torch.log(1 - noise) + self.log_a) / self.temperature
            concrete = torch.sigmoid(logit)
        else:
            # Use expectation (no noise)
            concrete = torch.sigmoid(self.log_a)

        # Stretch and clip to [0, 1]
        stretched = concrete * (high - low) + low
        clipped = torch.clamp(stretched, 0, 1)

        return clipped

    def get_penalty(self) -> torch.Tensor:
        """
        Compute L0 regularization penalty.

        Returns:
            Penalty value to minimize
        """
        low, high = self.stretch_limits
        assert low < 0.0, "Lower stretch limit must be negative for L0 penalty"

        # Compute P(gate is open) = P(stretched sigmoid > 0)
        p_open = torch.sigmoid(self.log_a - self.temperature * np.log(-low / high))
        p_open = torch.clamp(p_open, self.eps, 1.0 - self.eps)

        # L0 penalty: sum of opening probabilities
        return self.l0_penalty * torch.sum(p_open)

    def get_sparsity_rate(self) -> float:
        """
        Compute fraction of gates that are closed (zero).

        Returns:
            Sparsity rate in [0, 1]
        """
        gates = self.get_gates(is_train=False)
        is_closed = (gates == 0.0)
        return torch.mean(is_closed.float()).item()


# ============================================================================
# Head Importance (Michel et al., NeurIPS 2019)
# ============================================================================

class HeadImportanceMetric:
    """
    Gradient-based head importance estimation.

    Computes importance scores by measuring how much the loss changes
    when a head is masked (using gradient of loss w.r.t. mask).

    Reference:
        Michel et al., "Are Sixteen Heads Really Better than One?", NeurIPS 2019
    """

    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        device: torch.device,
        normalize_by_layer: bool = True,
        normalize_global: bool = True
    ):
        """
        Args:
            n_layers: Number of transformer layers
            n_heads: Number of attention heads per layer
            device: Device for computation
            normalize_by_layer: Apply layer-wise L2 normalization
            normalize_global: Apply global min-max normalization
        """
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.device = device
        self.normalize_by_layer = normalize_by_layer
        self.normalize_global = normalize_global

    def compute(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        head_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute head importance scores.

        Importance is measured as: I_h = |∂L/∂m_h|
        where L is the loss and m_h is the mask for head h.

        Args:
            model: Model to evaluate
            dataloader: DataLoader for computing importance
            head_mask: Optional initial mask (default: all ones)

        Returns:
            Importance tensor of shape (n_layers, n_heads)
        """
        # Initialize
        head_importance = torch.zeros(self.n_layers, self.n_heads).to(self.device)

        if head_mask is None:
            head_mask = torch.ones(self.n_layers, self.n_heads).to(self.device)

        head_mask.requires_grad_(True)
        model.apply_masks(head_mask)

        total_tokens = 0.0

        # Accumulate gradients over dataset
        for inputs in tqdm(dataloader, desc="Computing head importance"):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs)
            loss = outputs[0]

            # Compute gradient w.r.t. head mask
            grad = torch.autograd.grad(loss, head_mask)[0]
            head_importance += grad.abs().detach()

            # Track number of tokens for normalization
            total_tokens += inputs["attention_mask"].float().sum().item()

        # Normalize by number of tokens
        head_importance /= total_tokens

        # Apply normalizations
        if self.normalize_by_layer:
            head_importance = self._normalize_by_layer(head_importance)

        if self.normalize_global:
            head_importance = self._normalize_global(head_importance)

        return head_importance

    def _normalize_by_layer(
        self,
        importance: torch.Tensor,
        exponent: float = 2
    ) -> torch.Tensor:
        """Apply L2 normalization within each layer."""
        norm_by_layer = torch.pow(
            torch.pow(importance, exponent).sum(-1),
            1 / exponent
        )
        return importance / (norm_by_layer.unsqueeze(-1) + 1e-20)

    def _normalize_global(self, importance: torch.Tensor) -> torch.Tensor:
        """Apply min-max normalization globally."""
        min_val = importance.min()
        max_val = importance.max()
        return (importance - min_val) / (max_val - min_val + 1e-20)


# ============================================================================
# Evaluation Metrics
# ============================================================================

class SparsityMetric:
    """Metrics for measuring sparsity of pruned models."""

    @staticmethod
    def compute_sparsity(mask: torch.Tensor) -> float:
        """
        Compute sparsity percentage.

        Args:
            mask: Binary mask (1 = kept, 0 = pruned)

        Returns:
            Sparsity percentage (0-100)
        """
        num_pruned = (mask == 0).sum().item()
        total = mask.numel()
        return 100.0 * num_pruned / total

    @staticmethod
    def compute_remaining_ratio(mask: torch.Tensor) -> float:
        """
        Compute ratio of remaining (unpruned) elements.

        Args:
            mask: Binary mask (1 = kept, 0 = pruned)

        Returns:
            Remaining ratio (0-100)
        """
        num_remaining = mask.sum().item()
        total = mask.numel()
        return 100.0 * num_remaining / total

    @staticmethod
    def get_stats(mask: torch.Tensor) -> Dict[str, float]:
        """
        Get comprehensive sparsity statistics.

        Returns:
            Dictionary with sparsity metrics
        """
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


class PruningMetricTracker:
    """
    Tracks metrics during the pruning process.
    """

    def __init__(self):
        self.history = {
            "step": [],
            "sparsity": [],
            "performance": [],
            "remaining_heads": [],
            "temperature": [],
        }

    def update(
        self,
        step: int,
        mask: torch.Tensor,
        performance: float,
        temperature: Optional[float] = None
    ):
        """
        Record metrics at current step.

        Args:
            step: Training step
            mask: Current head mask
            performance: Task performance metric
            temperature: Current temperature (for DSP)
        """
        stats = SparsityMetric.get_stats(mask)

        self.history["step"].append(step)
        self.history["sparsity"].append(stats["sparsity_pct"])
        self.history["performance"].append(performance * 100)
        self.history["remaining_heads"].append(stats["remaining_heads"])

        if temperature is not None:
            self.history["temperature"].append(temperature)

    def get_history(self) -> Dict[str, list]:
        """Get full tracking history."""
        return self.history

    def get_best(self) -> Dict[str, float]:
        """Get best performance and corresponding metrics."""
        if not self.history["performance"]:
            return {}

        best_idx = np.argmax(self.history["performance"])

        result = {
            "best_step": self.history["step"][best_idx],
            "best_performance": self.history["performance"][best_idx],
            "best_sparsity": self.history["sparsity"][best_idx],
            "best_remaining_heads": self.history["remaining_heads"][best_idx],
        }

        if self.history["temperature"]:
            result["best_temperature"] = self.history["temperature"][best_idx]

        return result


# ============================================================================
# Utility Functions
# ============================================================================

def convert_gate_to_mask(
    gates: torch.Tensor,
    num_of_heads: Optional[int] = None,
    threshold: float = 0.5
) -> torch.Tensor:
    """
    Convert gate values to binary masks.

    Args:
        gates: Gate values (continuous or binary)
        num_of_heads: If specified, keep top-k heads
        threshold: Threshold for binarization (if num_of_heads is None)

    Returns:
        Binary mask tensor
    """
    head_mask = torch.zeros_like(gates)

    if num_of_heads is not None:
        # Top-k selection
        flat_gates = gates.view(-1)
        top_k_indices = flat_gates.argsort(descending=True)[:num_of_heads]

        flat_mask = head_mask.view(-1)
        flat_mask[top_k_indices] = 1.0
        head_mask = flat_mask.view_as(gates)
    else:
        # Threshold-based selection
        head_mask = (gates > threshold).float()

    return head_mask


def print_head_mask(mask: torch.Tensor, logger=None):
    """
    Pretty-print a head mask tensor.

    Args:
        mask: Head mask of shape (n_layers, n_heads)
        logger: Optional logger (uses print if None)
    """
    n_layers, n_heads = mask.shape

    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    # Header
    log("Layer/Head >\t" + "\t".join(f"{i+1}" for i in range(n_heads)))

    # Each layer
    for layer_idx in range(n_layers):
        if mask.dtype in [torch.long, torch.int]:
            values = "\t".join(f"{int(x)}" for x in mask[layer_idx].cpu())
        else:
            values = "\t".join(f"{x:.4f}" for x in mask[layer_idx].cpu())

        log(f"Layer {layer_idx + 1}:\t{values}")

    # Summary
    stats = SparsityMetric.get_stats(mask)
    log(f"\nRemaining heads: {stats['remaining_heads']}/{stats['total_heads']} "
        f"({stats['remaining_pct']:.1f}%)")
