"""
PDCL Adaptive Soft Pruning — Raw Math
=======================================
Replaces the hard top-K token selection with meaningful,
learned, adaptive soft gating.

THE PROBLEM WITH HARD TOP-K:
    Keeping exactly K=16 or K=32 tokens is arbitrary and destructive.
    - If the answer is in token 33, it gets silently dropped
    - K is fixed regardless of document complexity
    - Tokens just above/below the cutoff are treated completely differently
    - Context is abruptly severed

THE PDCL SOLUTION — ADAPTIVE SOFT GATING:
    Every token gets a continuous relevance gate in [0, 1].
    Low-relevance tokens → gate ≈ 0 → near-zero contribution
    High-relevance tokens → gate ≈ 1 → full contribution

    The threshold adapts per document based on the actual score distribution:
        threshold = mean(scores) + k_factor * std(scores)

    This means:
    - A document with one clearly relevant section: threshold is high,
      most tokens get low gates, a few get high gates
    - A document where relevance is spread out: threshold is lower,
      more tokens contribute meaningfully
    - NOTHING IS ABRUPTLY DROPPED — context is fully preserved via weighting

    The gate is differentiable everywhere, so gradients flow correctly
    through what the model learns to "ignore."

Math:
    c_que[b] = mean(out_que[b, :, :], axis=0)         (d,) — question context
    p_doc[b] = out_doc[b] @ W_d                        (T_doc, d) — projected doc
    p_que[b] = c_que[b] @ W_q                          (d,) — projected que

    scores[b, i] = dot(p_doc[b, i], p_que[b])         scalar per token
    mean_s = mean(scores, axis=1, keepdims=True)       (B, 1)
    std_s  = std(scores,  axis=1, keepdims=True)       (B, 1)

    threshold = mean_s + k_factor * std_s              (B, 1) adaptive!
    gate_logit[b, i] = (scores[b, i] - threshold[b]) * sharpness
    gate[b, i] = sigmoid(gate_logit[b, i])             soft 0→1

    gated_doc[b, i] = out_doc[b, i] * gate[b, i]      all tokens weighted

The span head then operates over ALL T_doc positions with gated representations.
High-gate tokens → high logit → predicted as answer start/end
Low-gate tokens → near-zero representation → naturally low logit

No abrupt K cutoff. No dropped context. Fully differentiable.
"""

from pdcl_backend import xp as np, to_cpu
from typing import Dict, Tuple, Optional
import numpy as _cpu_np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def xavier_uniform(shape):
    fan_in = shape[0]
    fan_out = shape[1] if len(shape) > 1 else shape[0]
    bound = _cpu_np.sqrt(6.0 / (fan_in + fan_out))
    return np.array(_cpu_np.random.uniform(-bound, bound, shape).astype(_cpu_np.float32))


class AdaptiveSoftPruning:
    """
    PDCL Adaptive Soft Pruning Layer.
    Replaces hard top-K with meaningful continuous relevance gating.
    All tokens are preserved — just weighted by their learned relevance.
    """

    def __init__(self,
                 d_model: int = 256,
                 k_factor: float = 0.5,
                 gate_sharpness: float = 3.0):
        """
        Args:
            d_model        : hidden dimension size
            k_factor       : how many std above mean sets the threshold
                             k_factor=0   → half the tokens above threshold
                             k_factor=0.5 → top ~30% above threshold
                             k_factor=1.0 → top ~16% above threshold
                             k_factor=1.5 → top ~7% above threshold
                             Adapts per document — so context is never arbitrarily cut
            gate_sharpness : how sharply the gate transitions (higher = harder boundary)
                             Low sharpness (1.0)  → very smooth transition
                             High sharpness (5.0) → near-hard boundary
        """
        self.d_model = d_model
        self.k_factor = k_factor
        self.gate_sharpness = gate_sharpness

        # Learnable relevance projection matrices
        bound = _cpu_np.sqrt(6.0 / (d_model + d_model))
        self.W_d = np.array(_cpu_np.random.uniform(-bound, bound, (d_model, d_model)).astype(_cpu_np.float32))
        self.W_q = np.array(_cpu_np.random.uniform(-bound, bound, (d_model, d_model)).astype(_cpu_np.float32))

        # Learnable k_factor (so model can adapt its own pruning aggressiveness)
        self.log_k = np.array([_cpu_np.log(max(k_factor, 0.1))], dtype=_cpu_np.float32)

        # Gradients
        self.dW_d = np.zeros_like(self.W_d)
        self.dW_q = np.zeros_like(self.W_q)
        self.dlog_k = np.zeros(1, dtype=_cpu_np.float32)

        # Cache for backward
        self._cache = {}

        print(f"AdaptiveSoftPruning initialized:")
        print(f"  d_model        : {d_model}")
        print(f"  k_factor       : {k_factor} (adaptive threshold = mean + k*std)")
        print(f"  gate_sharpness : {gate_sharpness}")
        print(f"  Mode           : SOFT GATING — no tokens removed, all weighted")

    def forward(self,
                out_doc: 'np.ndarray',
                out_que: 'np.ndarray') -> Tuple['np.ndarray', 'np.ndarray']:
        """
        Forward Pass — Adaptive soft gating.

        Inputs:
            out_doc : (B, T_doc, d_model) — document repr from attention
            out_que : (B, T_que, d_model) — question repr from attention

        Returns:
            gated_doc : (B, T_doc, d_model) — ALL tokens, weighted by relevance gate
            gates     : (B, T_doc) — gate values for inspection/monitoring
        """
        B, T_doc, d = out_doc.shape
        _, T_que, _ = out_que.shape

        # ── 1. Question context vector (average pool) ──
        c_que = np.mean(out_que, axis=1)  # (B, d)

        # ── 2. Project to relevance matching space ──
        p_doc = out_doc @ self.W_d        # (B, T_doc, d)
        p_que = c_que @ self.W_q          # (B, d)

        # ── 3. Relevance scores per token ──
        # dot product between each doc token and the question context
        scores = np.sum(p_doc * p_que[:, np.newaxis, :], axis=-1)  # (B, T_doc)

        # ── 4. Adaptive threshold (per document, adapts to score distribution) ──
        mean_s = np.mean(scores, axis=1, keepdims=True)   # (B, 1)
        std_s  = np.std(scores,  axis=1, keepdims=True)   # (B, 1)

        # k_factor is learned (via log parameterization to keep positive)
        k = np.exp(self.log_k[0])
        threshold = mean_s + k * std_s                    # (B, 1) — adaptive per doc

        # ── 5. Soft gate — sigmoid with sharpness scaling ──
        # Gate logit: how far above/below threshold each token's score is
        gate_logit = (scores - threshold) * self.gate_sharpness  # (B, T_doc)
        gates = sigmoid(gate_logit)                               # (B, T_doc) in [0, 1]

        # ── 6. Apply gates — ALL tokens kept, just weighted ──
        gated_doc = out_doc * gates[:, :, np.newaxis]  # (B, T_doc, d)

        # Cache for backward
        self._cache = {
            'B': B, 'T_doc': T_doc, 'T_que': T_que,
            'out_doc': out_doc,
            'out_que': out_que,
            'c_que': c_que,
            'p_doc': p_doc,
            'p_que': p_que,
            'scores': scores,
            'mean_s': mean_s,
            'std_s': std_s,
            'threshold': threshold,
            'gate_logit': gate_logit,
            'gates': gates,
            'k': float(to_cpu(k)),
        }

        return gated_doc, gates

    def backward(self, d_gated_doc: 'np.ndarray') -> Tuple['np.ndarray', 'np.ndarray']:
        """
        Backward Pass — chain rule through all adaptive gating operations.

        Input:
            d_gated_doc : (B, T_doc, d_model) — upstream gradient

        Outputs:
            d_out_doc : (B, T_doc, d_model)
            d_out_que : (B, T_que, d_model)
        """
        cache = self._cache
        B, T_doc, T_que = cache['B'], cache['T_doc'], cache['T_que']
        out_doc = cache['out_doc']
        out_que = cache['out_que']
        c_que   = cache['c_que']
        p_doc   = cache['p_doc']
        p_que   = cache['p_que']
        scores  = cache['scores']
        mean_s  = cache['mean_s']
        std_s   = cache['std_s']
        gates   = cache['gates']
        gate_logit = cache['gate_logit']
        k       = cache['k']

        # ── Step 1: Backward through gate application ──
        # gated_doc = out_doc * gates[:, :, None]
        # d_out_doc contribution from gating:
        d_out_doc_from_gate = d_gated_doc * gates[:, :, np.newaxis]  # (B, T_doc, d)

        # d_gates contribution:
        d_gates = np.sum(d_gated_doc * out_doc, axis=-1)  # (B, T_doc)

        # ── Step 2: Backward through sigmoid ──
        # gate = sigmoid(gate_logit)
        # d_gate_logit = d_gates * gate * (1 - gate)
        d_gate_logit = d_gates * gates * (1.0 - gates)  # (B, T_doc)

        # ── Step 3: Backward through sharpness scaling ──
        # gate_logit = (scores - threshold) * sharpness
        d_scores_from_gate = d_gate_logit * self.gate_sharpness  # (B, T_doc)
        d_threshold = -np.sum(d_gate_logit * self.gate_sharpness, axis=1, keepdims=True)  # (B, 1)

        # ── Step 4: Backward through adaptive threshold ──
        # threshold = mean_s + k * std_s
        d_mean_s = d_threshold.copy()
        d_std_s  = d_threshold * k

        # Backward through log_k
        # k = exp(log_k)  →  d_log_k = d_k * k  where d_k = sum(d_threshold * std_s)
        d_k = float(to_cpu(np.sum(d_threshold * std_s)))
        self.dlog_k += np.array([d_k * k])

        # Backward through mean and std
        # mean_s = mean(scores, axis=1)  →  d_scores from mean: d_mean_s / T_doc
        d_scores_from_mean = np.broadcast_to(d_mean_s / T_doc, scores.shape)

        # std_s = std(scores)  →  d_scores from std: d_std_s * (scores - mean_s) / (std_s * T_doc)
        d_scores_from_std  = d_std_s * (scores - mean_s) / (std_s * T_doc + 1e-8)

        # Total gradient to scores
        d_scores = d_scores_from_gate + d_scores_from_mean + d_scores_from_std  # (B, T_doc)

        # ── Step 5: Backward through score computation ──
        # scores = sum(p_doc * p_que[:, None, :], axis=-1)
        d_p_doc = d_scores[:, :, np.newaxis] * p_que[:, np.newaxis, :]  # (B, T_doc, d)
        d_p_que = np.sum(d_scores[:, :, np.newaxis] * p_doc, axis=1)    # (B, d)

        # ── Step 6: Backward through projection matrices W_d and W_q ──
        self.dW_d += np.sum(out_doc.transpose(0, 2, 1) @ d_p_doc, axis=0)
        self.dW_q += c_que.T @ d_p_que

        d_out_doc_from_proj = d_p_doc @ self.W_d.T       # (B, T_doc, d)
        d_c_que             = d_p_que @ self.W_q.T       # (B, d)

        # Backward through question average pooling
        d_out_que = np.tile(d_c_que[:, np.newaxis, :], (1, T_que, 1)) / T_que  # (B, T_que, d)

        # Total doc gradient
        d_out_doc = d_out_doc_from_gate + d_out_doc_from_proj  # (B, T_doc, d)

        return d_out_doc, d_out_que

    def zero_grad(self):
        self.dW_d[:] = 0
        self.dW_q[:] = 0
        self.dlog_k[:] = 0

    def update(self, lr: float, clip_norm: float = 1.0):
        grads  = [self.dW_d, self.dW_q]
        params = [self.W_d, self.W_q]

        grad_norm = float(to_cpu(np.sqrt(sum(np.sum(g**2) for g in grads))))
        clip_scale = min(1.0, clip_norm / (grad_norm + 1e-8))

        for p, g in zip(params, grads):
            p -= lr * g * clip_scale

        # Update log_k (small lr — changes pruning aggressiveness slowly)
        self.log_k -= lr * 0.1 * float(self.dlog_k[0])
        # Clamp: don't let k go below 0.0 or above 3.0
        self.log_k[:] = np.clip(self.log_k, np.log(0.05), np.log(3.0))

    def get_effective_keep_rate(self) -> float:
        """
        Approximate fraction of tokens with gate > 0.5 on average.
        Useful for monitoring how aggressive pruning is.
        """
        k = float(to_cpu(np.exp(self.log_k[0])))
        # For normal distribution: P(Z > k) where Z ~ N(0,1)
        # Approximate: P(gate > 0.5) ≈ P(score > threshold) ≈ P(Z > k)
        # Using simple approximation
        import math
        try:
            keep_rate = 0.5 * (1 - math.erf(k / math.sqrt(2)))
        except:
            keep_rate = max(0.05, 0.5 - k * 0.15)
        return max(0.05, min(0.95, keep_rate))
