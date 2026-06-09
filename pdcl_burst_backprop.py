"""
PDCL Core Burst Backpropagation — Full Integration
====================================================
This is the central model class. It integrates ALL PDCL components:

    1. EmbeddingLayer          — token + position + segment embeddings
    2. DimensionSegmentation   — split into doc/que subspaces and sequences
    3. ParallelAttention       — doc self, que self, cross-attention (+ connection pruning)
    4. GroupRelationDetector   — 3-way relationship detection (new)
    5. AdaptiveSoftPruning     — meaningful learned gating (replaces hard top-K)
    6. QASpanHead              — span prediction over full doc sequence
    7. CoreBurstScaling        — distance-weighted gradient decay from core point (new)

Core Burst Backprop Math:
    After graph_backward() collects all gradients, BEFORE update():
    For each weight matrix W with gradient dW:

    1. Core score per output feature j:
        c[j] = ||dW[:, j]||_2  *  ||W[:, j]||_2
        (combines gradient signal with weight magnitude)

    2. Core point:
        j* = argmax_j( c[j] )

    3. Distance from core (normalized):
        dist[j] = |j - j*| / max(fan_out - 1, 1)

    4. Exponential decay:
        λ[j] = exp(-β * dist[j])
        β controls sharpness: higher β = more focused burst

    5. Scale gradient:
        dW[:, j] *= λ[j]

This concentrates learning energy at the most informative feature
and smoothly tapers off toward less important features.
Features far from the core still learn but very slowly — they are
not frozen (that is feature pruning's job), just de-emphasized.
"""

from pdcl_backend import xp as np, to_cpu, to_device, GPU_AVAILABLE
from pdcl_embedding import EmbeddingLayer
from pdcl_segmentation import DimensionSegmentation
from pdcl_attention import ParallelAttention
from pdcl_pruning import AdaptiveSoftPruning
from pdcl_group_relations import GroupRelationDetector
from typing import Tuple, Optional
import numpy as _cpu_np


def xavier_uniform(shape):
    fan_in = shape[0]
    fan_out = shape[1] if len(shape) > 1 else shape[0]
    bound = _cpu_np.sqrt(6.0 / (fan_in + fan_out))
    return np.array(_cpu_np.random.uniform(-bound, bound, shape).astype(_cpu_np.float32))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-8)


# ─────────────────────────────────────────────
# QA SPAN HEAD — now operates over full T_doc
# ─────────────────────────────────────────────

class QASpanHead:
    """
    Span extraction head.
    Operates over ALL doc tokens (soft-gated, not hard top-K).
    Outputs start/end probability distribution over T_doc positions.

    Math:
        logit_start[b, i] = (gated_doc[b, i] @ v_start)[0]
        logit_end[b, i]   = (gated_doc[b, i] @ v_end)[0]
        p_start = softmax(logit_start)
        p_end   = softmax(logit_end)
        loss    = -log(p_start[ans_start]) - log(p_end[ans_end])
    """

    def __init__(self, d_model: int):
        self.d_model = d_model
        self.v_start = xavier_uniform((d_model, 1))
        self.v_end   = xavier_uniform((d_model, 1))

        self.dv_start = np.zeros_like(self.v_start)
        self.dv_end   = np.zeros_like(self.v_end)

        self._cache = {}

    def forward(self,
                gated_doc: 'np.ndarray',
                start_targets: 'np.ndarray',
                end_targets: 'np.ndarray') -> Tuple['np.ndarray', 'np.ndarray', float]:
        """
        Inputs:
            gated_doc     : (B, T_doc, d_model) — soft-gated doc tokens
            start_targets : (B,) — index of answer start in doc sequence
            end_targets   : (B,) — index of answer end in doc sequence

        Outputs:
            p_start : (B, T_doc) start probabilities
            p_end   : (B, T_doc) end probabilities
            loss    : scalar cross-entropy loss
        """
        B, T_doc, d = gated_doc.shape

        # Compute span logits
        logits_start = (gated_doc @ self.v_start).squeeze(-1)  # (B, T_doc)
        logits_end   = (gated_doc @ self.v_end).squeeze(-1)    # (B, T_doc)

        p_start = softmax(logits_start, axis=-1)  # (B, T_doc)
        p_end   = softmax(logits_end,   axis=-1)  # (B, T_doc)

        # Cross-entropy loss
        loss = 0.0
        for b in range(B):
            s = int(to_cpu(start_targets[b]))
            e = int(to_cpu(end_targets[b]))
            s = max(0, min(s, T_doc - 1))
            e = max(0, min(e, T_doc - 1))
            loss -= float(to_cpu(np.log(p_start[b, s] + 1e-9)))
            loss -= float(to_cpu(np.log(p_end[b, e] + 1e-9)))

        loss /= (2.0 * B)

        self._cache = {
            'gated_doc': gated_doc,
            'logits_start': logits_start,
            'logits_end': logits_end,
            'p_start': p_start,
            'p_end': p_end,
            'start_targets': start_targets,
            'end_targets': end_targets,
            'B': B, 'T_doc': T_doc,
        }

        return p_start, p_end, loss

    def backward(self) -> 'np.ndarray':
        """Returns gradient w.r.t gated_doc."""
        cache = self._cache
        B, T_doc = cache['B'], cache['T_doc']
        gated_doc = cache['gated_doc']
        p_start   = cache['p_start']
        p_end     = cache['p_end']
        start_t   = cache['start_targets']
        end_t     = cache['end_targets']

        # Gradient of cross-entropy loss w.r.t logits
        d_logits_start = p_start.copy()  # (B, T_doc)
        d_logits_end   = p_end.copy()

        for b in range(B):
            s = max(0, min(int(to_cpu(start_t[b])), T_doc - 1))
            e = max(0, min(int(to_cpu(end_t[b])),   T_doc - 1))
            d_logits_start[b, s] -= 1.0
            d_logits_end[b, e]   -= 1.0

        d_logits_start /= (2.0 * B)
        d_logits_end   /= (2.0 * B)

        # Gradient w.r.t v_start and v_end
        self.dv_start += np.sum(
            gated_doc.transpose(0, 2, 1) @ d_logits_start[:, :, np.newaxis],
            axis=0
        )
        self.dv_end += np.sum(
            gated_doc.transpose(0, 2, 1) @ d_logits_end[:, :, np.newaxis],
            axis=0
        )

        # Gradient w.r.t gated_doc
        d_gated_doc = (
            d_logits_start[:, :, np.newaxis] @ self.v_start.T +
            d_logits_end[:, :, np.newaxis]   @ self.v_end.T
        )   # (B, T_doc, d_model)

        return d_gated_doc

    def zero_grad(self):
        self.dv_start[:] = 0
        self.dv_end[:]   = 0

    def update(self, lr, clip_norm=1.0):
        for p, g in [(self.v_start, self.dv_start), (self.v_end, self.dv_end)]:
            gn = float(to_cpu(np.linalg.norm(g)))
            scale = min(1.0, clip_norm / (gn + 1e-8))
            p -= lr * g * scale


# ─────────────────────────────────────────────
# CORE BURST SCALING WITH BLAST RANGE FREEZING
# ─────────────────────────────────────────────

def apply_core_burst_scaling(model: 'CoreBurstBackprop',
                              beta: float = 2.0,
                              freeze_threshold: float = 0.15) -> None:
    """
    Core point identification + blast range freezing.

    Applied AFTER graph_backward() collects all gradients,
    BEFORE update() applies them.

    For each 2D weight matrix W with gradient dW:
        1. Compute core score per output feature j:
               c[j] = ||dW[:, j]||_2  *  ||W[:, j]||_2
        2. Find core feature:
               j* = argmax(c)
        3. Compute distance (normalized to [0,1]):
               dist[j] = |j - j*| / max(fan_out - 1, 1)
        4. Exponential decay:
               λ[j] = exp(-β * dist[j])
        5. Blast range freezing:
               If λ[j] >= freeze_threshold → scale gradient by λ[j] (active)
               If λ[j] <  freeze_threshold → zero gradient entirely  (frozen)
        6. Record active_mask and unscaled_grad_norms for the pruner.

    Args:
        model           : CoreBurstBackprop instance
        beta            : decay sharpness (higher = more focused, less spread)
        freeze_threshold: minimum decay value to keep a column active (below = frozen)
    """
    weight_grad_pairs = _collect_weight_grad_pairs(model)
    model._unscaled_grad_norms = {}
    model._active_masks = {}

    for name, (W, dW) in weight_grad_pairs.items():
        if dW is None or dW.ndim < 2:
            continue

        W_cpu  = to_cpu(W)
        dW_cpu = to_cpu(dW)

        fan_in, fan_out = W_cpu.shape[0], W_cpu.shape[1]
        if fan_out < 2:
            continue

        # Record unscaled gradient norms BEFORE any scaling
        grad_mag = _cpu_np.linalg.norm(dW_cpu, axis=0)   # (fan_out,)
        model._unscaled_grad_norms[name] = grad_mag.copy()

        weight_mag = _cpu_np.linalg.norm(W_cpu, axis=0)   # (fan_out,)
        core_score = grad_mag * weight_mag                  # (fan_out,)

        if core_score.sum() < 1e-12:
            # No signal — mark all as active (no freezing)
            model._active_masks[name] = _cpu_np.ones(fan_out, dtype=_cpu_np.float32)
            continue

        # Core point
        j_star = int(_cpu_np.argmax(core_score))

        # Distance from core (normalized)
        js   = _cpu_np.arange(fan_out, dtype=_cpu_np.float32)
        dist = _cpu_np.abs(js - j_star) / max(fan_out - 1, 1)

        # Exponential decay
        decay = _cpu_np.exp(-beta * dist)  # (fan_out,)

        # Blast range: active if decay >= threshold, frozen otherwise
        active_mask = (decay >= freeze_threshold).astype(_cpu_np.float32)
        model._active_masks[name] = active_mask

        # Apply: scale active columns by decay, zero frozen columns
        scaled_dW = dW_cpu * decay[_cpu_np.newaxis, :] * active_mask[_cpu_np.newaxis, :]

        # Write back
        if hasattr(dW, 'set'):  # CuPy
            dW.set(scaled_dW.astype(dW.dtype))
        else:
            dW[:] = scaled_dW


def _collect_weight_grad_pairs(model: 'CoreBurstBackprop') -> dict:
    """Extract all (W, dW) pairs from model components."""
    pairs = {}
    attn  = model.attention
    prune = model.prune
    embed = model.embed
    span  = model.span_head
    grp   = model.group_rel

    for prefix, module in [
        ('doc_attn', attn.doc_self_attn),
        ('que_attn', attn.que_self_attn),
        ('cross',    attn.que_doc_cross_attn),
    ]:
        pairs[f'{prefix}_Wq'] = (module.W_q, module.dW_q)
        pairs[f'{prefix}_Wk'] = (module.W_k, module.dW_k)
        pairs[f'{prefix}_Wv'] = (module.W_v, module.dW_v)
        pairs[f'{prefix}_Wo'] = (module.W_o, module.dW_o)

    pairs['fusion']   = (attn.W_fusion,  attn.dW_fusion)
    pairs['prune_Wd'] = (prune.W_d,      prune.dW_d)
    pairs['prune_Wq'] = (prune.W_q,      prune.dW_q)
    pairs['embed_tok']= (embed.E_token,  embed.dE_token)
    pairs['grp_W1']   = (grp.W1,         grp.dW1)
    pairs['grp_W2']   = (grp.W2,         grp.dW2)

    return pairs


# ─────────────────────────────────────────────
# CORE MODEL
# ─────────────────────────────────────────────

class CoreBurstBackprop:
    """
    PDCL Full Model — integrates all components.
    """

    def __init__(self,
                 vocab_size: int,
                 max_positions: int = 1024,
                 d_model: int = 256,
                 n_heads: int = 4,
                 burst_beta: float = 2.0,
                 group_pair_threshold: float = 0.15,
                 group_active_from_epoch: int = 3,
                 k_factor: float = 0.5,
                 gate_sharpness: float = 3.0,
                 burst_freeze_threshold: float = 0.15):

        self.d_model    = d_model
        self.burst_beta = burst_beta
        self.burst_freeze_threshold = burst_freeze_threshold
        self.current_epoch = 0
        self.maturity_reached = False

        # Blast range state (populated by apply_core_burst_scaling)
        self._unscaled_grad_norms = {}
        self._active_masks = {}

        # ── Components ──
        self.embed     = EmbeddingLayer(
            vocab_size=vocab_size, d_model=d_model,
            max_pos=max_positions, num_segments=2
        )
        self.segment   = DimensionSegmentation(d_model=d_model)
        self.attention = ParallelAttention(d_model=d_model, n_heads=n_heads)
        self.group_rel = GroupRelationDetector(
            d_model=d_model,
            pair_threshold=group_pair_threshold,
            active_from_epoch=group_active_from_epoch
        )
        self.prune     = AdaptiveSoftPruning(
            d_model=d_model,
            k_factor=k_factor,
            gate_sharpness=gate_sharpness
        )
        self.span_head = QASpanHead(d_model=d_model)

        print(f"\nCoreBurstBackprop initialized:")
        print(f"  vocab_size   : {vocab_size}")
        print(f"  d_model      : {d_model}")
        print(f"  burst β      : {burst_beta}")
        print(f"  k_factor     : {k_factor} (adaptive soft pruning)")

    def forward(self,
                token_ids:    'np.ndarray',
                positions:    'np.ndarray',
                segment_mask: 'np.ndarray',
                start_targets:'np.ndarray',
                end_targets:  'np.ndarray',
                training:     bool = True) -> Tuple['np.ndarray', 'np.ndarray', float]:
        """
        Full forward pass.

        Inputs:
            token_ids    : (B, T)
            positions    : (B, T)
            segment_mask : (B, T) — 0=doc, 1=que
            start_targets: (B,)  — answer start index in doc sequence
            end_targets  : (B,)  — answer end index in doc sequence
            training     : bool

        Returns:
            p_start : (B, T_doc)
            p_end   : (B, T_doc)
            loss    : float
        """
        # 1. Embedding
        embeddings = self.embed.forward(token_ids, positions, segment_mask, training)

        # 2. Segmentation → doc/que sequences
        seg_out = self.segment.forward(embeddings, segment_mask)
        seq_doc = seg_out['seq_doc']   # (B, T_doc, d)
        seq_que = seg_out['seq_que']   # (B, T_que, d)

        # 3. Parallel attention
        attn_out = self.attention.forward(seq_doc, seq_que)
        out_doc  = attn_out['out_doc']  # (B, T_doc, d)
        out_que  = attn_out['out_que']  # (B, T_que, d)

        # 4. Group relationship detection (active from epoch 3 onwards)
        attn_weights = self.attention.get_doc_attn_weights()
        if attn_weights is not None:
            out_doc = self.group_rel.forward(out_doc, attn_weights,
                                              epoch=self.current_epoch)

        # 5. Adaptive soft pruning (replaces hard top-K)
        gated_doc, gates = self.prune.forward(out_doc, out_que)

        # 6. Span prediction over full doc sequence
        p_start, p_end, loss = self.span_head.forward(
            gated_doc, start_targets, end_targets
        )

        return p_start, p_end, loss

    def burst_backward(self) -> None:
        """
        Full backward pass with optional core burst gradient scaling.

        Phase 1 (maturity_reached=False):
            Standard backward — no burst scaling, no freezing.
            All gradients flow freely so the model can learn balanced representations.

        Phase 2 (maturity_reached=True):
            SpanHead → SoftPruning → GroupRelations → Attention → Segmentation → Embedding
            Then: Core Burst Scaling with blast range freezing applied to ALL gradients.
        """
        from pdcl_graph_backprop import GraphParallelBackward
        gbp = GraphParallelBackward(self)
        gbp.graph_backward()

        if self.maturity_reached:
            # Phase 2: Apply core burst scaling with blast range freezing
            apply_core_burst_scaling(self, beta=self.burst_beta, freeze_threshold=self.burst_freeze_threshold)
        else:
            # Phase 1: No burst scaling — free gradient flow
            self._unscaled_grad_norms = {}
            self._active_masks = {}

    def zero_grad(self) -> None:
        self.embed.zero_grad()
        self.attention.zero_grad()
        self.prune.zero_grad()
        self.group_rel.zero_grad()
        self.span_head.zero_grad()

    def update(self, lr: float, clip_norm: float = 1.0) -> None:
        self.embed.update(lr, clip_norm)
        self.attention.update(lr, clip_norm)
        self.prune.update(lr, clip_norm)
        self.group_rel.update(lr, clip_norm)
        self.span_head.update(lr, clip_norm)

    def save_checkpoint(self, path: str) -> None:
        import os
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        _cpu_np.savez(
            path,
            E_token  = to_cpu(self.embed.E_token),
            E_pos    = to_cpu(self.embed.E_pos),
            E_seg    = to_cpu(self.embed.E_seg),
            gamma    = to_cpu(self.embed.gamma),
            beta     = to_cpu(self.embed.beta),
            W_q_doc  = to_cpu(self.attention.doc_self_attn.W_q),
            W_k_doc  = to_cpu(self.attention.doc_self_attn.W_k),
            W_v_doc  = to_cpu(self.attention.doc_self_attn.W_v),
            W_o_doc  = to_cpu(self.attention.doc_self_attn.W_o),
            W_q_que  = to_cpu(self.attention.que_self_attn.W_q),
            W_k_que  = to_cpu(self.attention.que_self_attn.W_k),
            W_v_que  = to_cpu(self.attention.que_self_attn.W_v),
            W_o_que  = to_cpu(self.attention.que_self_attn.W_o),
            W_q_cross= to_cpu(self.attention.que_doc_cross_attn.W_q),
            W_k_cross= to_cpu(self.attention.que_doc_cross_attn.W_k),
            W_v_cross= to_cpu(self.attention.que_doc_cross_attn.W_v),
            W_o_cross= to_cpu(self.attention.que_doc_cross_attn.W_o),
            W_fusion = to_cpu(self.attention.W_fusion),
            W_d_prune= to_cpu(self.prune.W_d),
            W_q_prune= to_cpu(self.prune.W_q),
            log_k    = to_cpu(self.prune.log_k),
            v_start  = to_cpu(self.span_head.v_start),
            v_end    = to_cpu(self.span_head.v_end),
            grp_W1   = to_cpu(self.group_rel.W1),
            grp_b1   = to_cpu(self.group_rel.b1),
            grp_W2   = to_cpu(self.group_rel.W2),
            grp_b2   = to_cpu(self.group_rel.b2),
        )
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        import os
        actual_path = path
        if not path.endswith('.npz'):
            if os.path.exists(path + '.npz'):
                actual_path = path + '.npz'
            elif not os.path.exists(path):
                actual_path = path + '.npz'

        data = _cpu_np.load(actual_path)

        def _set(param, name):
            if name in data:
                param[:] = to_device(data[name])

        _set(self.embed.E_token, 'E_token')
        _set(self.embed.E_pos, 'E_pos')
        _set(self.embed.E_seg, 'E_seg')
        _set(self.embed.gamma, 'gamma')
        _set(self.embed.beta, 'beta')
        _set(self.attention.doc_self_attn.W_q, 'W_q_doc')
        _set(self.attention.doc_self_attn.W_k, 'W_k_doc')
        _set(self.attention.doc_self_attn.W_v, 'W_v_doc')
        _set(self.attention.doc_self_attn.W_o, 'W_o_doc')
        _set(self.attention.que_self_attn.W_q, 'W_q_que')
        _set(self.attention.que_self_attn.W_k, 'W_k_que')
        _set(self.attention.que_self_attn.W_v, 'W_v_que')
        _set(self.attention.que_self_attn.W_o, 'W_o_que')
        _set(self.attention.que_doc_cross_attn.W_q, 'W_q_cross')
        _set(self.attention.que_doc_cross_attn.W_k, 'W_k_cross')
        _set(self.attention.que_doc_cross_attn.W_v, 'W_v_cross')
        _set(self.attention.que_doc_cross_attn.W_o, 'W_o_cross')
        _set(self.attention.W_fusion, 'W_fusion')
        _set(self.prune.W_d, 'W_d_prune')
        _set(self.prune.W_q, 'W_q_prune')
        _set(self.prune.log_k, 'log_k')
        _set(self.span_head.v_start, 'v_start')
        _set(self.span_head.v_end, 'v_end')
        _set(self.group_rel.W1, 'grp_W1')
        _set(self.group_rel.b1, 'grp_b1')
        _set(self.group_rel.W2, 'grp_W2')
        _set(self.group_rel.b2, 'grp_b2')
        print(f"Checkpoint loaded from {actual_path}")
