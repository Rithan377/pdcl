"""
PDCL Group Relationship Detection — Raw Math
=============================================
Pairwise attention only captures A→B relationships.
Some patterns only emerge when THREE or more tokens
are considered together as a unit.

Example:
    "cat" attends to "sat" (pairwise — strong)
    "sat" attends to "mat" (pairwise — strong)
    "cat"+"sat"+"mat" together encode a scene (group — only visible jointly)

Algorithm (efficient — avoids O(T³) brute force):
    1. From the attention matrix, find strong pairs:
       (i, j) is a strong pair if avg_attention[i, j] > threshold

    2. Build candidate triplets from strong pairs:
       (i, j, k) is a candidate if (i,j), (j,k), (i,k) are ALL strong pairs
       This dramatically reduces search space

    3. Compute three-way cosine similarity for each candidate triplet:
       Φ(i,j,k) = (h_i · h_j)(h_j · h_k)(h_i · h_k) / (||h_i|| ||h_j|| ||h_k||²)

    4. If Φ(i,j,k) >= γ, create group embedding:
       z_ijk = MLP([h_i || h_j || h_k])

    5. Pool all group embeddings and add as residual to doc representation

Math for backward:
    d_z_pool → d_z_ijk (via pooling backward)
    d_z_ijk → d_h_concat (via MLP backward)
    d_h_concat → d_h_i, d_h_j, d_h_k (via concat backward)
    Scatter back to d_out_doc via original token positions
"""

from pdcl_backend import xp as np, to_cpu
from typing import Dict, List, Tuple, Optional
import numpy as _cpu_np


def relu(x):
    return np.maximum(0, x)


def relu_backward(x, d_out):
    return d_out * (x > 0)


def xavier_uniform(shape):
    fan_in = shape[0]
    fan_out = shape[1] if len(shape) > 1 else shape[0]
    bound = _cpu_np.sqrt(6.0 / (fan_in + fan_out))
    return np.array(_cpu_np.random.uniform(-bound, bound, shape).astype(_cpu_np.float32))


class GroupRelationDetector:
    """
    Detects 3-way group relationships among document tokens.
    Produces group embeddings that are added as residuals to the doc representation.
    """

    def __init__(self,
                 d_model: int = 256,
                 pair_threshold: float = 0.15,
                 group_threshold: float = 0.10,
                 max_triplets_per_sample: int = 32,
                 active_from_epoch: int = 3):
        """
        Args:
            d_model                  : embedding dimension
            pair_threshold           : min attention weight to consider a strong pair
            group_threshold          : min three-way similarity to confirm a group
            max_triplets_per_sample  : cap on triplets per batch sample (efficiency)
            active_from_epoch        : don't compute groups until this epoch
                                       (let pairwise attention stabilize first)
        """
        self.d_model = d_model
        self.pair_threshold = pair_threshold
        self.group_threshold = group_threshold
        self.max_triplets = max_triplets_per_sample
        self.active_from_epoch = active_from_epoch

        # MLP: [h_i || h_j || h_k] → group embedding
        # Input: 3 * d_model → hidden: d_model → output: d_model
        self.W1 = xavier_uniform((3 * d_model, d_model))
        self.b1 = np.zeros(d_model)
        self.W2 = xavier_uniform((d_model, d_model))
        self.b2 = np.zeros(d_model)

        # Gradients
        self.dW1 = np.zeros_like(self.W1)
        self.db1 = np.zeros_like(self.b1)
        self.dW2 = np.zeros_like(self.W2)
        self.db2 = np.zeros_like(self.b2)

        # Cache
        self._cache = {}
        self._active = False

        print(f"GroupRelationDetector initialized:")
        print(f"  d_model          : {d_model}")
        print(f"  pair_threshold   : {pair_threshold}")
        print(f"  group_threshold  : {group_threshold}")
        print(f"  Active from epoch: {active_from_epoch}")

    def _find_strong_pairs(self,
                           attn_matrix: _cpu_np.ndarray,
                           T: int) -> List[Tuple[int, int]]:
        """
        Find token pairs with attention weight above threshold.
        attn_matrix: (T, T) averaged attention weights (CPU numpy)
        Returns list of (i, j) strong pairs.
        """
        # Average across heads if needed
        if attn_matrix.ndim == 3:
            avg_attn = attn_matrix.mean(axis=0)  # (T, T)
        else:
            avg_attn = attn_matrix

        # Symmetrize: pair (i,j) is strong if either direction is strong
        sym_attn = (avg_attn + avg_attn.T) / 2.0

        rows, cols = _cpu_np.where(sym_attn > self.pair_threshold)
        pairs = [(int(r), int(c)) for r, c in zip(rows, cols) if r < c]
        return pairs

    def _find_triplet_candidates(self,
                                  pairs: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
        """
        From strong pairs, find candidate triplets where ALL 3 pairwise links are strong.
        If (i,j), (j,k), (i,k) all exist in pairs → (i,j,k) is a candidate.
        """
        pair_set = set(pairs)
        triplets = []
        checked = set()

        # Build adjacency: for each node, who are its neighbors?
        adjacency = {}
        for (i, j) in pairs:
            adjacency.setdefault(i, set()).add(j)
            adjacency.setdefault(j, set()).add(i)

        for (i, j) in pairs:
            # Find common neighbors of i and j — these form triplets
            neighbors_i = adjacency.get(i, set())
            neighbors_j = adjacency.get(j, set())
            common = neighbors_i & neighbors_j

            for k in common:
                triplet = tuple(sorted([i, j, k]))
                if triplet not in checked:
                    checked.add(triplet)
                    triplets.append(triplet)

        return triplets[:self.max_triplets]  # cap for efficiency

    def _three_way_similarity(self,
                               h_i: _cpu_np.ndarray,
                               h_j: _cpu_np.ndarray,
                               h_k: _cpu_np.ndarray) -> float:
        """
        Three-way cosine similarity.
        Φ(i,j,k) = (h_i·h_j)(h_j·h_k)(h_i·h_k) / (||h_i|| ||h_j|| ||h_k||²)
        """
        norm_i = _cpu_np.linalg.norm(h_i) + 1e-8
        norm_j = _cpu_np.linalg.norm(h_j) + 1e-8
        norm_k = _cpu_np.linalg.norm(h_k) + 1e-8

        cos_ij = _cpu_np.dot(h_i, h_j) / (norm_i * norm_j)
        cos_jk = _cpu_np.dot(h_j, h_k) / (norm_j * norm_k)
        cos_ik = _cpu_np.dot(h_i, h_k) / (norm_i * norm_k)

        return float(cos_ij * cos_jk * cos_ik)

    def forward(self,
                out_doc: 'np.ndarray',
                attn_weights: 'np.ndarray',
                epoch: int = 0) -> 'np.ndarray':
        """
        Forward pass.

        Inputs:
            out_doc      : (B, T_doc, d_model) — doc representations from attention
            attn_weights : (B, n_heads, T_doc, T_doc) — attention weights for pair detection
            epoch        : current epoch (used to decide if module is active)

        Output:
            out_doc_enriched : (B, T_doc, d_model) — same shape, with group residuals added
        """
        self._active = (epoch >= self.active_from_epoch)
        B, T_doc, d = out_doc.shape

        if not self._active or T_doc < 3:
            # Not yet active — pass through unchanged
            self._cache = {'active': False, 'out_doc': out_doc}
            return out_doc

        # Work on CPU for the combinatorial search
        out_doc_cpu = to_cpu(out_doc)   # (B, T_doc, d)
        attn_cpu = to_cpu(attn_weights) # (B, H, T_doc, T_doc)

        # Per-sample group embedding accumulator
        group_residuals_cpu = _cpu_np.zeros_like(out_doc_cpu)  # (B, T_doc, d)

        # Cache for backward
        triplet_records = []

        for b in range(B):
            # Average attention across heads for this sample
            avg_attn_b = attn_cpu[b].mean(axis=0)  # (T_doc, T_doc)

            # Find strong pairs
            strong_pairs = self._find_strong_pairs(avg_attn_b, T_doc)
            if len(strong_pairs) < 3:
                triplet_records.append([])
                continue

            # Find triplet candidates
            triplets = self._find_triplet_candidates(strong_pairs)

            sample_records = []
            for triplet in triplets:
                i, j, k = triplet
                h_i = out_doc_cpu[b, i]   # (d,)
                h_j = out_doc_cpu[b, j]   # (d,)
                h_k = out_doc_cpu[b, k]   # (d,)

                # Check three-way similarity
                phi = self._three_way_similarity(h_i, h_j, h_k)
                if phi < self.group_threshold:
                    continue

                # Build group embedding via MLP
                h_concat = _cpu_np.concatenate([h_i, h_j, h_k])  # (3d,)
                z1 = h_concat @ to_cpu(self.W1) + to_cpu(self.b1)  # (d,)
                a1 = _cpu_np.maximum(0, z1)                          # ReLU
                z2 = a1 @ to_cpu(self.W2) + to_cpu(self.b2)         # (d,)

                # Add group embedding as residual to all three token positions
                group_residuals_cpu[b, i] += z2
                group_residuals_cpu[b, j] += z2
                group_residuals_cpu[b, k] += z2

                # Record for backward
                sample_records.append({
                    'triplet': (i, j, k),
                    'h_concat': h_concat,
                    'z1': z1,
                    'a1': a1,
                    'z2': z2,
                    'phi': phi,
                })

            triplet_records.append(sample_records)

        # Move group residuals to device and add to out_doc
        group_residuals = np.array(group_residuals_cpu.astype(_cpu_np.float32))
        out_doc_enriched = out_doc + group_residuals

        self._cache = {
            'active': True,
            'out_doc': out_doc,
            'group_residuals': group_residuals,
            'triplet_records': triplet_records,
            'B': B,
            'T_doc': T_doc,
            'd': d,
        }

        return out_doc_enriched

    def backward(self, d_out_enriched: 'np.ndarray') -> 'np.ndarray':
        """
        Backward pass.

        Input:
            d_out_enriched : (B, T_doc, d_model)

        Output:
            d_out_doc : (B, T_doc, d_model) — gradient w.r.t original out_doc
        """
        if not self._cache.get('active', False):
            return d_out_enriched

        cache = self._cache
        B, T_doc, d = cache['B'], cache['T_doc'], cache['d']
        triplet_records = cache['triplet_records']

        # Gradient passes through the residual addition to out_doc
        d_out_doc = d_out_enriched.copy()

        # Also backprop through the MLP for each triplet
        d_out_cpu = to_cpu(d_out_enriched)  # (B, T_doc, d)

        for b in range(B):
            for record in triplet_records[b]:
                i, j, k = record['triplet']
                h_concat = record['h_concat']
                z1 = record['z1']
                a1 = record['a1']

                # Gradient reaching z2 from the three positions it contributed to
                d_z2 = d_out_cpu[b, i] + d_out_cpu[b, j] + d_out_cpu[b, k]

                # Backward through W2
                d_a1 = d_z2 @ to_cpu(self.W2).T        # (d,)
                W1_cpu = to_cpu(self.W1)
                W2_cpu = to_cpu(self.W2)
                self.dW2 += np.array(
                    _cpu_np.outer(a1, d_z2).astype(_cpu_np.float32)
                )
                self.db2 += np.array(d_z2.astype(_cpu_np.float32))

                # Backward through ReLU
                d_z1 = d_a1 * (z1 > 0)

                # Backward through W1
                self.dW1 += np.array(
                    _cpu_np.outer(h_concat, d_z1).astype(_cpu_np.float32)
                )
                self.db1 += np.array(d_z1.astype(_cpu_np.float32))

        return d_out_doc

    def zero_grad(self):
        self.dW1[:] = 0
        self.db1[:] = 0
        self.dW2[:] = 0
        self.db2[:] = 0

    def update(self, lr: float, clip_norm: float = 1.0):
        grads = [self.dW1, self.db1, self.dW2, self.db2]
        params = [self.W1, self.b1, self.W2, self.b2]

        grad_norm = float(to_cpu(np.sqrt(sum(np.sum(g**2) for g in grads))))
        clip_scale = min(1.0, clip_norm / (grad_norm + 1e-8))

        for p, g in zip(params, grads):
            p -= lr * g * clip_scale
