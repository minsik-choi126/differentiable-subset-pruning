"""
Pruning Metrics for Attention Head Pruning

This module contains refined metric calculations used in differentiable subset pruning.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm


class HeadImportanceMetric:
    """
    Computes attention head importance scores using gradient-based methods.

    Based on "Are Sixteen Heads Really Better than One?" (Michel et al., 2019)
    http://arxiv.org/abs/1905.10650
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
            device: Device to run computations on
            normalize_by_layer: Whether to apply layer-wise normalization
            normalize_global: Whether to apply global min-max normalization
        """
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.device = device
        self.normalize_by_layer = normalize_by_layer
        self.normalize_global = normalize_global

    def compute(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        head_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute head importance scores.

        Args:
            model: The model to evaluate
            dataloader: DataLoader for computing importance
            head_mask: Optional mask to apply (default: all ones)

        Returns:
            Tensor of shape (n_layers, n_heads) with importance scores
        """
        # Initialize head importance and mask
        head_importance = torch.zeros(self.n_layers, self.n_heads).to(self.device)

        if head_mask is None:
            head_mask = torch.ones(self.n_layers, self.n_heads).to(self.device)

        head_mask.requires_grad_(requires_grad=True)
        model.apply_masks(head_mask)

        # Accumulate gradients
        tot_tokens = 0.0

        for inputs in tqdm(dataloader, desc="Computing head importance"):
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs)
            loss = outputs[0]

            # Compute gradient with respect to head mask
            grad = torch.autograd.grad(loss, head_mask)[0]
            head_importance += grad.abs().detach()

            # Track number of tokens
            tot_tokens += inputs["attention_mask"].float().detach().sum().item()

        # Normalize by number of tokens
        head_importance /= tot_tokens

        # Apply layer-wise normalization
        if self.normalize_by_layer:
            head_importance = self._normalize_by_layer(head_importance)

        # Apply global normalization
        if self.normalize_global:
            head_importance = self._normalize_global(head_importance)

        return head_importance

    def _normalize_by_layer(self, importance: torch.Tensor, exponent: float = 2) -> torch.Tensor:
        """Apply L2 normalization across heads within each layer."""
        norm_by_layer = torch.pow(torch.pow(importance, exponent).sum(-1), 1 / exponent)
        return importance / (norm_by_layer.unsqueeze(-1) + 1e-20)

    def _normalize_global(self, importance: torch.Tensor) -> torch.Tensor:
        """Apply min-max normalization globally."""
        min_val = importance.min()
        max_val = importance.max()
        return (importance - min_val) / (max_val - min_val + 1e-20)


class SparsityMetric:
    """Computes sparsity metrics for pruned models."""

    @staticmethod
    def compute_sparsity(mask: torch.Tensor) -> float:
        """
        Compute sparsity percentage.

        Args:
            mask: Binary mask tensor (1 = kept, 0 = pruned)

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
            mask: Binary mask tensor (1 = kept, 0 = pruned)

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


class TaskPerformanceMetric:
    """Computes task-specific performance metrics."""

    def __init__(self, task_name: str, output_mode: str = "classification"):
        """
        Args:
            task_name: Name of the task (e.g., 'mnli', 'sst-2')
            output_mode: 'classification' or 'regression'
        """
        self.task_name = task_name
        self.output_mode = output_mode

    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        head_mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Evaluate model performance.

        Args:
            model: Model to evaluate
            dataloader: Evaluation dataloader
            device: Device to run on
            head_mask: Optional head mask to apply

        Returns:
            Task performance score
        """
        if head_mask is not None:
            model.apply_masks(head_mask)

        model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs in tqdm(dataloader, desc="Evaluating"):
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs)
                logits = outputs[1]
                labels = inputs["labels"]

                all_preds.append(logits.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

        # Concatenate all predictions and labels
        preds = np.concatenate(all_preds, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        # Compute final predictions
        if self.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        else:
            preds = np.squeeze(preds)

        # Compute metric (accuracy for classification)
        if self.output_mode == "classification":
            accuracy = (preds == labels).mean()
            return accuracy
        else:
            # For regression, could use MSE or correlation
            from scipy.stats import pearsonr, spearmanr
            pearson_corr = pearsonr(preds, labels)[0]
            return pearson_corr


class NERMetric:
    """Metrics for Named Entity Recognition tasks."""

    def __init__(self, label_map: Dict[int, str]):
        """
        Args:
            label_map: Mapping from label IDs to label names
        """
        self.label_map = label_map

    def compute(
        self,
        predictions: np.ndarray,
        label_ids: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute NER metrics (precision, recall, F1).

        Args:
            predictions: Model predictions (batch_size, seq_len, num_labels)
            label_ids: Ground truth labels (batch_size, seq_len)

        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        from seqeval.metrics import f1_score, precision_score, recall_score

        preds_list, labels_list = self._align_predictions(predictions, label_ids)

        return {
            "precision": precision_score(labels_list, preds_list),
            "recall": recall_score(labels_list, preds_list),
            "f1": f1_score(labels_list, preds_list),
        }

    def _align_predictions(
        self,
        predictions: np.ndarray,
        label_ids: np.ndarray
    ) -> Tuple[list, list]:
        """Align predictions with labels, filtering padding tokens."""
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape

        preds_list = [[] for _ in range(batch_size)]
        labels_list = [[] for _ in range(batch_size)]

        ignore_index = -100  # CrossEntropyLoss default ignore_index

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != ignore_index:
                    labels_list[i].append(self.label_map[label_ids[i][j]])
                    preds_list[i].append(self.label_map[preds[i][j]])

        return preds_list, labels_list


class PruningMetricTracker:
    """
    Tracks multiple metrics during the pruning process.
    """

    def __init__(self):
        self.history = {
            "sparsity": [],
            "performance": [],
            "remaining_heads": [],
        }

    def update(
        self,
        mask: torch.Tensor,
        performance: float
    ):
        """
        Record metrics at current pruning step.

        Args:
            mask: Current head mask
            performance: Current task performance
        """
        sparsity_metric = SparsityMetric()
        stats = sparsity_metric.get_stats(mask)

        self.history["sparsity"].append(stats["sparsity_pct"])
        self.history["performance"].append(performance * 100)
        self.history["remaining_heads"].append(stats["remaining_heads"])

    def get_history(self) -> Dict[str, list]:
        """Get full tracking history."""
        return self.history

    def get_best(self) -> Dict[str, float]:
        """Get best performance and corresponding sparsity."""
        if not self.history["performance"]:
            return {}

        best_idx = np.argmax(self.history["performance"])

        return {
            "best_performance": self.history["performance"][best_idx],
            "best_sparsity": self.history["sparsity"][best_idx],
            "best_remaining_heads": self.history["remaining_heads"][best_idx],
        }


def convert_gate_to_mask(
    gates: torch.Tensor,
    num_of_heads: Optional[int] = None
) -> torch.Tensor:
    """
    Convert gate values to binary masks.

    Args:
        gates: Gate values (continuous or binary)
        num_of_heads: Number of heads to keep (top-k selection)
                     If None, use threshold of 0.5

    Returns:
        Binary mask tensor
    """
    head_mask = torch.zeros_like(gates)

    if num_of_heads is not None:
        # Select top-k heads
        flat_gates = gates.view(-1)
        top_k_indices = flat_gates.argsort(descending=True)[:num_of_heads]

        flat_mask = head_mask.view(-1)
        flat_mask[top_k_indices] = 1.0
        head_mask = flat_mask.view_as(gates)
    else:
        # Use threshold
        head_mask = (gates > 0.5).float()

    return head_mask
