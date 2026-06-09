"""
PDCL Cross-Dimension Graph Formation — Raw Math
================================================
The D processing dimensions are not truly independent.
Some dimensions, by chance of their data partition, end up
learning correlated features. When we detect this correlation,
we can weight the gradient aggregation to give MORE influence
to highly-correlated dimension pairs (since they have learned
complementary structure from related data regions).

Architecture:
    Each dimension d tracks its mean feature activations per step:
        activation_stats[d] = mean(|h_doc_d|, axis=(batch, seq))  → (d_model,)

    At epoch boundaries, compute Pearson correlation between all pairs:
        ρ(d1, d2) = Cov(stats_d1, stats_d2) / (σ_d1 * σ_d2)

    Build weighted adjacency matrix:
        adj[d1, d2] = max(0, ρ(d1, d2))   ← only positive correlation

    Compute dimension weights for gradient aggregation:
        w[d] = 1 + λ * mean(adj[d, :])    ← more connected → more weight
        w = w / sum(w)                     ← normalize to sum to 1

    Apply in gradient aggregation:
        grad_master = sum_d( w[d] * grad_d )

Math:
    Pearson correlation:
        ρ(X, Y) = (sum((X - μ_X)(Y - μ_Y))) / (N * σ_X * σ_Y)

This replaces uniform averaging (1/D for all) with
data-driven weighted averaging that reflects which dimensions
learned complementary vs redundant patterns.
"""

from pdcl_backend import xp as np, to_cpu, cpu_np
from typing import Dict, List, Optional
import numpy as _np


class CrossDimensionGraph:
    """
    Tracks cross-dimension feature correlations and computes
    weighted gradient aggregation based on learned graph structure.
    """

    def __init__(self,
                 n_dimensions: int,
                 d_model: int,
                 correlation_lambda: float = 0.5,
                 min_epochs_before_graph: int = 2,
                 ema_decay: float = 0.9):
        """
        Args:
            n_dimensions            : number of processing dimensions
            d_model                 : feature dimension size
            correlation_lambda      : weight of graph influence on aggregation (0=uniform, 1=full graph)
            min_epochs_before_graph : epochs before graph is built
            ema_decay               : decay for EMA of activation statistics
        """
        self.n_dims = n_dimensions
        self.d_model = d_model
        self.lam = correlation_lambda
        self.min_epochs = min_epochs_before_graph
        self.ema_decay = ema_decay

        # EMA of mean absolute activation per dimension: (n_dims, d_model)
        self.activation_ema: Dict[int, _np.ndarray] = {}

        # Correlation matrix: (n_dims, n_dims)
        self.correlation_matrix = _np.eye(n_dimensions, dtype=_np.float32)

        # Weighted aggregation weights: (n_dims,) — starts uniform
        self.aggregation_weights = _np.ones(n_dimensions, dtype=_np.float32) / n_dimensions

        # Graph adjacency (positive correlations only)
        self.adjacency = _np.zeros((n_dimensions, n_dimensions), dtype=_np.float32)

        # Step counter per dimension
        self.step_counts: Dict[int, int] = {d: 0 for d in range(n_dimensions)}

        print(f"CrossDimensionGraph initialized:")
        print(f"  n_dimensions     : {n_dimensions}")
        print(f"  d_model          : {d_model}")
        print(f"  correlation λ    : {correlation_lambda}")
        print(f"  Graph active at  : epoch {min_epochs_before_graph}")

    def update_activations(self, dim_id: int, doc_repr: 'np.ndarray') -> None:
        """
        Called after each forward pass for a dimension.
        Updates EMA of mean absolute activation for this dimension.

        Args:
            dim_id   : which processing dimension (0 to n_dims-1)
            doc_repr : (B, T_doc, d_model) — doc representation from attention output
        """
        # Mean absolute activation per feature: (d_model,)
        current_stats = _np.abs(to_cpu(doc_repr)).mean(axis=(0, 1))  # (d_model,)

        if dim_id not in self.activation_ema:
            self.activation_ema[dim_id] = current_stats.copy()
        else:
            self.activation_ema[dim_id] = (
                self.ema_decay * self.activation_ema[dim_id] +
                (1.0 - self.ema_decay) * current_stats
            )

        self.step_counts[dim_id] = self.step_counts.get(dim_id, 0) + 1

    def compute_graph(self, epoch: int) -> _np.ndarray:
        """
        Called at epoch boundaries.
        Computes Pearson correlation matrix and updates aggregation weights.

        Returns:
            correlation_matrix : (n_dims, n_dims)
        """
        if epoch < self.min_epochs:
            return self.correlation_matrix

        # Need activation stats for all dimensions
        available_dims = [d for d in range(self.n_dims) if d in self.activation_ema]
        if len(available_dims) < 2:
            return self.correlation_matrix

        # Build activation matrix: (n_dims, d_model)
        stats_matrix = _np.zeros((self.n_dims, self.d_model), dtype=_np.float32)
        for d in available_dims:
            stats_matrix[d] = self.activation_ema[d]

        # Pearson correlation matrix
        # Subtract mean per dimension
        means = stats_matrix.mean(axis=1, keepdims=True)  # (n_dims, 1)
        stds  = stats_matrix.std(axis=1, keepdims=True) + 1e-8

        z_scored = (stats_matrix - means) / stds  # (n_dims, d_model)

        # ρ = (Z @ Z.T) / (d_model - 1)
        corr = (z_scored @ z_scored.T) / (self.d_model - 1)  # (n_dims, n_dims)

        # Clip to [-1, 1]
        corr = _np.clip(corr, -1.0, 1.0)
        self.correlation_matrix = corr

        # Adjacency: only positive correlations, no self-loops
        adj = _np.maximum(0.0, corr)
        _np.fill_diagonal(adj, 0.0)
        self.adjacency = adj

        # Aggregation weights:
        # w[d] = 1 + λ * mean_connectivity[d]
        # mean_connectivity[d] = mean of adj[d, :] (how correlated dim d is with others)
        mean_conn = adj.mean(axis=1)  # (n_dims,)
        raw_weights = 1.0 + self.lam * mean_conn
        self.aggregation_weights = raw_weights / (raw_weights.sum() + 1e-8)

        return corr

    def get_aggregation_weights(self) -> _np.ndarray:
        """
        Returns current aggregation weights for gradient averaging.
        Shape: (n_dims,) — sums to 1.
        """
        return self.aggregation_weights.copy()

    def get_graph_summary(self) -> Dict:
        """Returns summary statistics about the current graph."""
        n_edges = int((self.adjacency > 0.1).sum()) // 2
        avg_correlation = float(self.adjacency.mean())
        max_correlation = float(self.adjacency.max())
        weight_entropy = float(
            -_np.sum(self.aggregation_weights * _np.log(self.aggregation_weights + 1e-8))
        )
        return {
            'n_edges': n_edges,
            'avg_correlation': avg_correlation,
            'max_correlation': max_correlation,
            'weight_entropy': weight_entropy,
            'aggregation_weights': self.aggregation_weights.tolist(),
            'correlation_matrix': self.correlation_matrix.tolist(),
        }
