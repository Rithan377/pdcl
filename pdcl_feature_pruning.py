"""
PDCL Feature Dimension Pruning — Raw Math
==========================================
Core PDCL concept: not all d_model features inside each dimension
are equally useful. Low-importance features waste computation and
add noise. This module tracks gradient magnitude per feature across
training and permanently masks out low-importance ones.

Key distinction from token pruning:
    Token pruning  = which document tokens are relevant to the question
    Feature pruning = which learned features (dimensions) inside weight
                      matrices are actually carrying meaningful signal

Math:
    Importance of feature j in weight matrix W:
        s[j] = running_mean( ||∂L/∂W[:, j]||_2 )

    Adaptive threshold (grows each epoch so pruning gets more aggressive):
        ε_e = percentile(s, prune_pct_e)
        prune_pct_e = base_pct + (max_pct - base_pct) * (e / E)

    Binary mask:
        mask[j] = 1  if s[j] >= ε_e
        mask[j] = 0  if s[j] <  ε_e   ← permanently zero, never restored

    Applied to gradient before update:
        ∂L/∂W[:, j] *= mask[j]

    And zeroes the weight itself (dead feature stays dead):
        W[:, j] *= mask[j]

This runs at EPOCH boundaries, not every step — giving the model
time to learn before features are judged and pruned.
"""

from pdcl_backend import xp as np, to_cpu, cpu_np
from typing import Dict, Optional


class FeatureDimensionPruner:
    """
    Tracks per-feature gradient importance across training steps.
    Applies progressive feature-level pruning at epoch boundaries.
    """

    def __init__(self,
                 d_model: int = 256,
                 base_prune_pct: float = 10.0,
                 max_prune_pct: float = 40.0,
                 total_epochs: int = 20,
                 min_epochs_before_prune: int = 3,
                 ema_decay: float = 0.95):
        """
        Args:
            d_model             : feature dimension size
            base_prune_pct      : starting pruning percentile (lower = keep more)
            max_prune_pct       : maximum pruning percentile reached at final epoch
            total_epochs        : total training epochs (for schedule)
            min_epochs_before_prune: warm-up before any pruning starts
            ema_decay           : decay for exponential moving average of importance
        """
        self.d_model = d_model
        self.base_prune_pct = base_prune_pct
        self.max_prune_pct = max_prune_pct
        self.total_epochs = total_epochs
        self.min_epochs_before_prune = min_epochs_before_prune
        self.ema_decay = ema_decay

        # EMA importance scores per weight matrix (keyed by matrix name)
        # shape per matrix: (fan_out,) — one score per output feature
        self.importance: Dict[str, cpu_np.ndarray] = {}

        # Permanent binary masks — once a feature is pruned it stays pruned
        # shape per matrix: (fan_out,) — 1=active, 0=permanently pruned
        self.masks: Dict[str, cpu_np.ndarray] = {}

        # Cumulative step count for EMA
        self.step = 0

        print(f"FeatureDimensionPruner initialized:")
        print(f"  d_model              : {d_model}")
        print(f"  Prune range          : {base_prune_pct}% → {max_prune_pct}%")
        print(f"  Warm-up epochs       : {min_epochs_before_prune}")
        print(f"  EMA decay            : {ema_decay}")

    def _get_weight_grad_pairs(self, model) -> Dict[str, tuple]:
        """
        Extract all (weight_matrix, gradient_matrix) pairs from the model.
        Only includes 2D matrices — not 1D biases/gammas.
        Returns dict keyed by descriptive name.
        """
        pairs = {}
        attn = model.attention
        prune = model.prune
        embed = model.embed

        # Attention matrices
        for prefix, module in [
            ('doc_attn', attn.doc_self_attn),
            ('que_attn', attn.que_self_attn),
            ('cross_attn', attn.que_doc_cross_attn),
        ]:
            pairs[f'{prefix}_Wq'] = (module.W_q, module.dW_q)
            pairs[f'{prefix}_Wk'] = (module.W_k, module.dW_k)
            pairs[f'{prefix}_Wv'] = (module.W_v, module.dW_v)
            pairs[f'{prefix}_Wo'] = (module.W_o, module.dW_o)

        pairs['fusion_W'] = (attn.W_fusion, attn.dW_fusion)
        pairs['prune_Wd'] = (prune.W_d, prune.dW_d)
        pairs['prune_Wq'] = (prune.W_q, prune.dW_q)

        # Embedding matrices (large — prune conservatively)
        pairs['embed_Etoken'] = (embed.E_token, embed.dE_token)

        return pairs

    def update_importance(self, model) -> None:
        """
        Called AFTER each backward pass.
        Updates EMA of gradient magnitude per output feature for every weight matrix.

        Uses unscaled gradient norms (before burst scaling) when available,
        and respects blast range active masks — frozen columns have their
        EMA importance scores preserved (not decayed to zero).

        Math:
            current_importance[j] = ||∂L/∂W[:, j]||_2  (unscaled, pre-burst)
            For active columns (active_mask[j] = 1):
                ema[j] = decay * ema[j] + (1 - decay) * current_importance[j]
            For frozen columns (active_mask[j] = 0):
                ema[j] unchanged  (importance score held constant)
        """
        self.step += 1
        has_unscaled = hasattr(model, '_unscaled_grad_norms') and model._unscaled_grad_norms
        has_active = hasattr(model, '_active_masks') and model._active_masks
        pairs = self._get_weight_grad_pairs(model)

        for name, (W, dW) in pairs.items():
            # Use unscaled gradient norms if available (pre-burst scaling)
            if has_unscaled and name in model._unscaled_grad_norms:
                current_imp = model._unscaled_grad_norms[name]
            else:
                dW_cpu = to_cpu(dW)  # (fan_in, fan_out)
                current_imp = cpu_np.linalg.norm(dW_cpu, axis=0)  # (fan_out,)

            # Get active mask (1=active, 0=frozen by blast range)
            if has_active and name in model._active_masks:
                active_mask = model._active_masks[name]
            else:
                active_mask = cpu_np.ones(current_imp.shape, dtype=cpu_np.float32)

            if name not in self.importance:
                # Initialize EMA
                self.importance[name] = current_imp.copy()
                self.masks[name] = cpu_np.ones(current_imp.shape, dtype=cpu_np.float32)
            else:
                # Selective EMA update: only update active columns, preserve frozen ones
                updated = (
                    self.ema_decay * self.importance[name] +
                    (1.0 - self.ema_decay) * current_imp
                )
                # Where active_mask=1: use updated EMA
                # Where active_mask=0: keep previous importance (frozen, preserved)
                self.importance[name] = (
                    active_mask * updated +
                    (1.0 - active_mask) * self.importance[name]
                )

    def apply_masks(self, epoch: int, model) -> Dict[str, float]:
        """
        Called at EPOCH boundaries (only after maturity is reached).
        
        CONSERVATIVE PRUNING: Only removes truly dead features — those whose
        gradient importance is near zero (below 1% of mean importance).
        No aggressive ramping schedule. No compounding.
        
        Returns dict of sparsity stats per matrix.
        """
        pairs = self._get_weight_grad_pairs(model)
        stats = {}

        for name, (W, dW) in pairs.items():
            if name not in self.importance:
                continue

            imp = self.importance[name]       # (fan_out,)
            existing_mask = self.masks[name]  # (fan_out,) — permanent mask

            # Only consider currently active features
            active_imp = imp[existing_mask > 0.5]
            if len(active_imp) < 4:
                continue  # keep at least a few features

            # Conservative threshold: only prune features with importance
            # below 1% of the mean importance — truly dead channels
            mean_imp = float(cpu_np.mean(active_imp))
            threshold = mean_imp * 0.01  # 1% of mean = essentially zero

            # New mask: prune only near-zero features (AND respect previous mask)
            new_mask = ((imp >= threshold) & (existing_mask > 0.5)).astype(cpu_np.float32)

            # Safety: never prune more than 40% of features total
            if new_mask.sum() < max(4, imp.shape[0] * 0.6):
                new_mask = existing_mask.copy()

            self.masks[name] = new_mask

            # Apply mask to weight matrix on device
            mask_device = np.array(new_mask)
            W_cpu = to_cpu(W)
            dW_cpu = to_cpu(dW)

            # Zero out pruned feature columns in both weight and gradient
            W_cpu[:, new_mask < 0.5] = 0.0
            dW_cpu[:, new_mask < 0.5] = 0.0

            # Write back to device
            if hasattr(W, 'set'):  # CuPy
                W.set(W_cpu)
                dW.set(dW_cpu)
            else:
                W[:] = W_cpu
                dW[:] = dW_cpu

            sparsity = 1.0 - new_mask.mean()
            stats[name] = float(sparsity)

        return stats

    def apply_gradient_masks(self, model) -> None:
        """
        Called DURING update — masks gradient of pruned features to zero.
        Prevents pruned features from being accidentally restored by gradient updates.
        """
        pairs = self._get_weight_grad_pairs(model)
        for name, (W, dW) in pairs.items():
            if name not in self.masks:
                continue
            mask_cpu = self.masks[name]  # (fan_out,)
            pruned_cols = cpu_np.where(mask_cpu < 0.5)[0]
            if len(pruned_cols) == 0:
                continue
            # Zero pruned columns in gradient
            if hasattr(dW, 'get'):  # CuPy
                dW_cpu = dW.get()
                dW_cpu[:, pruned_cols] = 0.0
                dW.set(dW_cpu)
            else:
                dW[:, pruned_cols] = 0.0

    def get_sparsity_report(self) -> Dict[str, float]:
        """Returns current sparsity ratio per weight matrix."""
        return {
            name: float(1.0 - mask.mean())
            for name, mask in self.masks.items()
        }
