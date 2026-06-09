"""
PDCL Parallel Multi-Head Attention Layer — Raw Math + Connection Pruning
=========================================================================
No libraries. Pure NumPy tensor and matrix operations.

ORIGINAL: Verified correct multi-head attention (unchanged)
ADDED:    Connection pruning over time — tracking which token-to-token
          attention connections are consistently weak and permanently
          zeroing them to reduce computation and enforce sparsity.

Connection Pruning Math:
    After each forward pass, update EMA of attention weights:
        Ā_{ij}^(t) = α * Ā_{ij}^(t-1) + (1-α) * A_{ij}^(t)

    At epoch boundaries, build sparse mask:
        S_{ij} = 1  if Ā_{ij} >= δ
        S_{ij} = 0  if Ā_{ij} <  δ   ← permanently zeroed

    Future forward passes apply mask BEFORE softmax:
        S_masked = S + (-∞ * (1 - S_ij))   ← masked positions → -∞ → softmax → 0
        A = softmax(S_masked)

This means:
    - Connections that are consistently ignored are permanently disabled
    - The model becomes sparser over training
    - Remaining connections carry stronger, more focused signal
    - Nothing is pruned based on a single forward pass — requires persistent
      low attention over many steps before pruning triggers

The group relationship detector needs the raw attention weights (before masking)
to find strong pairs. We expose get_last_attn_weights() for this purpose.
"""

from pdcl_backend import xp as np, to_cpu, to_device, GPU_AVAILABLE
from typing import Dict, Tuple, Optional
import numpy as _cpu_np


# ─────────────────────────────────────────────
# SOFTMAX UTILITIES WITH GRADIENTS
# ─────────────────────────────────────────────

def softmax(x, axis=-1):
    max_val = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - max_val)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def softmax_backward(out, d_out, axis=-1):
    sum_d_out_out = np.sum(d_out * out, axis=axis, keepdims=True)
    return out * (d_out - sum_d_out_out)


def xavier_uniform(shape):
    import numpy as _np
    fan_in = shape[0]
    fan_out = shape[1] if len(shape) > 1 else shape[0]
    bound = _np.sqrt(6.0 / (fan_in + fan_out))
    return np.array(_np.random.uniform(-bound, bound, shape).astype(_np.float32))


# ─────────────────────────────────────────────
# MULTI-HEAD ATTENTION — UNCHANGED CORE MATH
# ─────────────────────────────────────────────

class MultiHeadAttention:
    """
    Multi-Head Attention with optional sparse connection mask.
    Core math unchanged from verified version.
    Connection pruning adds a pre-softmax mask applied over time.
    """

    def __init__(self, d_model: int, n_heads: int = 4,
                 conn_ema_decay: float = 0.95,
                 conn_prune_threshold: float = 0.02):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.scale   = 1.0 / _cpu_np.sqrt(self.d_head)

        # Learnable projections
        self.W_q = xavier_uniform((d_model, d_model))
        self.W_k = xavier_uniform((d_model, d_model))
        self.W_v = xavier_uniform((d_model, d_model))
        self.W_o = xavier_uniform((d_model, d_model))

        # Gradients
        self.dW_q = np.zeros_like(self.W_q)
        self.dW_k = np.zeros_like(self.W_k)
        self.dW_v = np.zeros_like(self.W_v)
        self.dW_o = np.zeros_like(self.W_o)

        # ── Connection Pruning State ──
        self.conn_ema_decay   = conn_ema_decay
        self.conn_threshold   = conn_prune_threshold
        self.attn_ema         = None   # (H, T_q, T_k) running average — initialized lazily
        self.conn_mask        = None   # (H, T_q, T_k) binary — None = all connections active
        self.conn_step        = 0

        self._cache = {}

    def forward(self, q_in, k_in, v_in):
        B, T_q, _ = q_in.shape
        _, T_k, _ = k_in.shape

        Q_proj = q_in @ self.W_q
        K_proj = k_in @ self.W_k
        V_proj = v_in @ self.W_v

        Q = Q_proj.reshape(B, T_q, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        K = K_proj.reshape(B, T_k, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        V = V_proj.reshape(B, T_k, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        S = (Q @ K.transpose(0, 1, 3, 2)) * self.scale  # (B, H, T_q, T_k)

        # ── Apply connection mask if active ──
        if self.conn_mask is not None:
            # conn_mask: (H, T_q, T_k) — broadcast over batch
            # Set masked positions to large negative → softmax → 0
            mask_expanded = self.conn_mask[np.newaxis, :, :, :]  # (1, H, T_q, T_k)
            # -1e9 where mask=0 (pruned), 0 where mask=1 (active)
            mask_val = (1.0 - mask_expanded) * (-1e9)
            S = S + mask_val

        A = softmax(S, axis=-1)   # (B, H, T_q, T_k)
        O = A @ V                  # (B, H, T_q, d_head)

        O_concat = O.transpose(0, 2, 1, 3).reshape(B, T_q, self.d_model)
        out = O_concat @ self.W_o

        # ── Update EMA of attention weights for connection pruning ──
        # Average over batch dimension first: (H, T_q, T_k)
        A_mean_cpu = _cpu_np.array(to_cpu(A)).mean(axis=0)

        if self.attn_ema is None:
            self.attn_ema = A_mean_cpu.copy()
        else:
            # EMA update — only for same-sized sequences
            if self.attn_ema.shape == A_mean_cpu.shape:
                self.attn_ema = (
                    self.conn_ema_decay * self.attn_ema +
                    (1.0 - self.conn_ema_decay) * A_mean_cpu
                )
            else:
                # Sequence length changed — reset
                self.attn_ema = A_mean_cpu.copy()

        self.conn_step += 1

        self._cache = {
            'q_in': q_in, 'k_in': k_in, 'v_in': v_in,
            'Q': Q, 'K': K, 'V': V,
            'S': S, 'A': A, 'O': O, 'O_concat': O_concat,
        }

        return out

    def get_last_attn_weights(self):
        """
        Returns the most recent attention weights: (B, H, T_q, T_k).
        Used by GroupRelationDetector to find strong pairs.
        """
        return self._cache.get('A', None)

    def update_connection_mask(self, threshold: Optional[float] = None) -> int:
        """
        Called at epoch boundaries.
        Builds a sparse mask by permanently zeroing connections with EMA < threshold.

        Returns:
            n_pruned : number of connections pruned
        """
        if self.attn_ema is None:
            return 0

        delta = threshold if threshold is not None else self.conn_threshold

        # Build mask: 1 where connection is active, 0 where pruned
        new_mask_cpu = (self.attn_ema >= delta).astype(_cpu_np.float32)

        # Respect existing mask (once pruned, always pruned)
        if self.conn_mask is not None:
            existing_mask_cpu = _cpu_np.array(to_cpu(self.conn_mask))
            new_mask_cpu = new_mask_cpu * existing_mask_cpu

        # Safety: never prune more than 70% of connections
        active_frac = new_mask_cpu.mean()
        if active_frac < 0.30:
            # Too aggressive — only prune the bottom 30%
            flat = self.attn_ema.flatten()
            safe_threshold = _cpu_np.percentile(flat, 30)
            new_mask_cpu = (self.attn_ema >= safe_threshold).astype(_cpu_np.float32)

        H, T_q, T_k = new_mask_cpu.shape
        n_pruned = int((new_mask_cpu < 0.5).sum())

        self.conn_mask = np.array(new_mask_cpu)
        return n_pruned

    def backward(self, d_out):
        cache = self._cache
        q_in, k_in, v_in = cache['q_in'], cache['k_in'], cache['v_in']
        Q, K, V = cache['Q'], cache['K'], cache['V']
        A, O    = cache['A'], cache['O']
        O_concat = cache['O_concat']

        B, T_q, _ = q_in.shape
        _, T_k, _ = k_in.shape

        self.dW_o += np.sum(O_concat.transpose(0, 2, 1) @ d_out, axis=0)
        d_O_concat = d_out @ self.W_o.T

        d_O = d_O_concat.reshape(B, T_q, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        d_V = A.transpose(0, 1, 3, 2) @ d_O
        d_A = d_O @ V.transpose(0, 1, 3, 2)

        d_S = softmax_backward(A, d_A, axis=-1)
        d_S = d_S * self.scale

        # Gradient does not flow through masked positions (they got -1e9 → zero in softmax)
        # The softmax_backward handles this correctly already via dA → dS

        d_Q = d_S @ K
        d_K = d_S.transpose(0, 1, 3, 2) @ Q

        d_Q_proj = d_Q.transpose(0, 2, 1, 3).reshape(B, T_q, self.d_model)
        d_K_proj = d_K.transpose(0, 2, 1, 3).reshape(B, T_k, self.d_model)
        d_V_proj = d_V.transpose(0, 2, 1, 3).reshape(B, T_k, self.d_model)

        self.dW_q += np.sum(q_in.transpose(0, 2, 1) @ d_Q_proj, axis=0)
        self.dW_k += np.sum(k_in.transpose(0, 2, 1) @ d_K_proj, axis=0)
        self.dW_v += np.sum(v_in.transpose(0, 2, 1) @ d_V_proj, axis=0)

        d_q_in = d_Q_proj @ self.W_q.T
        d_k_in = d_K_proj @ self.W_k.T
        d_v_in = d_V_proj @ self.W_v.T

        return d_q_in, d_k_in, d_v_in

    def zero_grad(self):
        self.dW_q[:] = 0
        self.dW_k[:] = 0
        self.dW_v[:] = 0
        self.dW_o[:] = 0

    def update(self, lr, clip_norm=1.0):
        grads  = [self.dW_q, self.dW_k, self.dW_v, self.dW_o]
        params = [self.W_q,  self.W_k,  self.W_v,  self.W_o]
        grad_norm = float(to_cpu(np.sqrt(sum(np.sum(g**2) for g in grads))))
        clip_scale = min(1.0, clip_norm / (grad_norm + 1e-8))
        for p, g in zip(params, grads):
            p -= lr * g * clip_scale


# ─────────────────────────────────────────────
# PARALLEL ATTENTION CONTROLLER
# ─────────────────────────────────────────────

class ParallelAttention:
    """
    PDCL Parallel Attention layer.
    Runs doc self-attention, question self-attention, and cross-attention
    simultaneously. Exposes attention weights for group relation detection.
    Connection pruning is managed per-head within each sub-module.
    """

    def __init__(self, d_model=256, n_heads=4,
                 conn_ema_decay=0.95, conn_prune_threshold=0.02):
        self.d_model = d_model

        self.doc_self_attn      = MultiHeadAttention(d_model, n_heads, conn_ema_decay, conn_prune_threshold)
        self.que_self_attn      = MultiHeadAttention(d_model, n_heads, conn_ema_decay, conn_prune_threshold)
        self.que_doc_cross_attn = MultiHeadAttention(d_model, n_heads, conn_ema_decay, conn_prune_threshold)

        self.W_fusion  = xavier_uniform((d_model * 2, d_model))
        self.dW_fusion = np.zeros_like(self.W_fusion)

        self._cache = {}

        print(f"ParallelAttention initialized:")
        print(f"  d_model  : {d_model}")
        print(f"  n_heads  : {n_heads}")
        print(f"  Conn EMA : {conn_ema_decay} | prune δ: {conn_prune_threshold}")

    def forward(self, seq_doc, seq_que):
        doc_self  = self.doc_self_attn.forward(seq_doc, seq_doc, seq_doc)
        que_self  = self.que_self_attn.forward(seq_que, seq_que, seq_que)
        que_cross = self.que_doc_cross_attn.forward(que_self, doc_self, doc_self)

        que_concat = np.concatenate([que_self, que_cross], axis=-1)
        out_que = que_concat @ self.W_fusion
        out_doc = doc_self

        self._cache = {
            'que_self': que_self, 'que_cross': que_cross,
            'que_concat': que_concat,
        }

        return {'out_doc': out_doc, 'out_que': out_que}

    def get_doc_attn_weights(self):
        """
        Returns doc self-attention weights (B, H, T_doc, T_doc).
        Used by GroupRelationDetector.
        """
        return self.doc_self_attn.get_last_attn_weights()

    def update_connection_masks(self, threshold=None) -> Dict:
        """
        Update sparse connection masks for all three attention modules.
        Called at epoch boundaries.
        Returns count of pruned connections per module.
        """
        return {
            'doc_self'  : self.doc_self_attn.update_connection_mask(threshold),
            'que_self'  : self.que_self_attn.update_connection_mask(threshold),
            'cross_attn': self.que_doc_cross_attn.update_connection_mask(threshold),
        }

    def backward(self, d_out_doc, d_out_que):
        cache = self._cache
        que_concat = cache['que_concat']

        self.dW_fusion += np.sum(que_concat.transpose(0, 2, 1) @ d_out_que, axis=0)
        d_que_concat = d_out_que @ self.W_fusion.T

        d_que_self_from_fusion = d_que_concat[:, :, :self.d_model]
        d_que_cross            = d_que_concat[:, :, self.d_model:]

        d_que_self_from_cross, d_doc_self_k, d_doc_self_v = \
            self.que_doc_cross_attn.backward(d_que_cross)

        d_doc_self = d_out_doc + d_doc_self_k + d_doc_self_v
        d_que_self = d_que_self_from_fusion + d_que_self_from_cross

        d_q_doc, d_k_doc, d_v_doc = self.doc_self_attn.backward(d_doc_self)
        d_seq_doc = d_q_doc + d_k_doc + d_v_doc

        d_q_que, d_k_que, d_v_que = self.que_self_attn.backward(d_que_self)
        d_seq_que = d_q_que + d_k_que + d_v_que

        return d_seq_doc, d_seq_que

    def zero_grad(self):
        self.doc_self_attn.zero_grad()
        self.que_self_attn.zero_grad()
        self.que_doc_cross_attn.zero_grad()
        self.dW_fusion[:] = 0

    def update(self, lr, clip_norm=1.0):
        self.doc_self_attn.update(lr, clip_norm)
        self.que_self_attn.update(lr, clip_norm)
        self.que_doc_cross_attn.update(lr, clip_norm)

        grad_norm = float(to_cpu(np.linalg.norm(self.dW_fusion)))
        clip_scale = min(1.0, clip_norm / (grad_norm + 1e-8))
        self.W_fusion -= lr * self.dW_fusion * clip_scale
