"""
PDCL Embedding Layer — Raw Math
=================================
No libraries. Pure NumPy matrix operations.

Three embeddings combined:
1. Token Embedding    — what is this token
2. Position Embedding — where is this token (from real char offsets)
3. Segment Embedding  — is this document or question

Math:
    E_final[t] = E_token[id[t]] + E_pos[pos[t]] + E_seg[seg[t]]

All three matrices are learnable.
Gradients flow back through all three during backprop.

Initialization:
    Xavier uniform: W ~ U(-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out)))
    This keeps variance stable at initialization — critical for deep networks.
"""

from pdcl_backend import xp as np, to_cpu, to_device, GPU_AVAILABLE
import json
import os
from typing import Dict, List, Tuple, Optional


# ─────────────────────────────────────────────
# MATH UTILITIES
# ─────────────────────────────────────────────

def xavier_uniform(shape: Tuple, gain: float = 1.0) -> np.ndarray:
    """
    Xavier uniform initialization.

    Math:
        fan_in  = shape[0]  (inputs)
        fan_out = shape[1]  (outputs)
        bound   = gain * sqrt(6 / (fan_in + fan_out))
        W ~ Uniform(-bound, +bound)

    Why: keeps variance of activations stable across layers.
    Without this — deep networks explode or vanish from layer 1.
    """
    fan_in  = shape[0]
    fan_out = shape[1] if len(shape) > 1 else shape[0]
    bound   = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-bound, bound, shape)


def layer_norm(x: np.ndarray,
               gamma: np.ndarray,
               beta: np.ndarray,
               eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Layer Normalization — applied after embedding combination.

    Math:
        mu    = mean(x)          over last dimension
        sigma = std(x)           over last dimension
        x_hat = (x - mu) / (sigma + eps)
        out   = gamma * x_hat + beta

    gamma, beta are learnable scale and shift parameters.
    eps prevents division by zero.

    Returns: (normalized output, mu, sigma) — mu and sigma needed for backprop.
    """
    mu    = np.mean(x, axis=-1, keepdims=True)       # (batch, seq, 1)
    var   = np.var(x,  axis=-1, keepdims=True)        # (batch, seq, 1)
    x_hat = (x - mu) / np.sqrt(var + eps)             # (batch, seq, d)
    out   = gamma * x_hat + beta                       # (batch, seq, d)
    return out, x_hat, np.sqrt(var + eps)


def layer_norm_backward(d_out: np.ndarray,
                        x_hat: np.ndarray,
                        sigma: np.ndarray,
                        gamma: np.ndarray,
                        eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Backward pass through layer norm.

    Math (chain rule through normalization):
        d_gamma = sum(d_out * x_hat, axis=(0,1))
        d_beta  = sum(d_out, axis=(0,1))
        d_x_hat = d_out * gamma
        d_x     = (1/sigma) * (d_x_hat - mean(d_x_hat)
                  - x_hat * mean(d_x_hat * x_hat))

    Returns gradients for (x, gamma, beta).
    """
    N = x_hat.shape[-1]  # dimension size

    d_gamma = np.sum(d_out * x_hat, axis=(0, 1))     # (d,)
    d_beta  = np.sum(d_out,         axis=(0, 1))      # (d,)
    d_x_hat = d_out * gamma                            # (batch, seq, d)

    # Gradient through normalization
    d_x = (1.0 / sigma) * (
        d_x_hat
        - np.mean(d_x_hat, axis=-1, keepdims=True)
        - x_hat * np.mean(d_x_hat * x_hat, axis=-1, keepdims=True)
    )

    return d_x, d_gamma, d_beta


def encode_position(positions: List[int],
                    max_pos: int,
                    d_model: int) -> np.ndarray:
    """
    Convert raw character positions to position bucket IDs.

    PDCL uses real character offsets from tokenizer.
    We bucket them into max_pos bins so the embedding
    matrix has a fixed size.

    Math:
        bucket[t] = floor(pos[t] / max_char * max_pos)
        bucket[t] = clip(bucket[t], 0, max_pos - 1)

    This preserves relative ordering of positions
    while fitting into a fixed embedding table.
    """
    if max(positions) == 0:
        return np.zeros(len(positions), dtype=np.int32)

    max_char = max(positions) + 1
    buckets  = np.floor(
        np.array(positions, dtype=np.float32) / max_char * max_pos
    ).astype(np.int32)
    buckets  = np.clip(buckets, 0, max_pos - 1)
    return buckets


# ─────────────────────────────────────────────
# EMBEDDING LAYER
# ─────────────────────────────────────────────

class EmbeddingLayer:
    """
    PDCL Embedding Layer.

    Combines three learned matrices:
        E_token   : (vocab_size,  d_model) — one row per token
        E_pos     : (max_pos,     d_model) — one row per position bucket
        E_seg     : (num_segments,d_model) — one row per segment type

    Forward:
        out[t] = E_token[id[t]] + E_pos[pos[t]] + E_seg[seg[t]]
        out    = LayerNorm(out)

    Backward:
        Gradients flow back into all three matrices
        via simple index-based scatter-add operations.
    """

    def __init__(self,
                 vocab_size : int,
                 d_model    : int   = 256,
                 max_pos    : int   = 512,
                 num_segments: int  = 2,
                 dropout_rate: float = 0.1):

        self.vocab_size   = vocab_size
        self.d_model      = d_model
        self.max_pos      = max_pos
        self.num_segments = num_segments
        self.dropout_rate = dropout_rate

        # ── Learnable parameters ──
        # Token embedding matrix
        self.E_token = xavier_uniform((vocab_size, d_model))

        # Position embedding matrix
        self.E_pos   = xavier_uniform((max_pos, d_model))

        # Segment embedding matrix
        self.E_seg   = xavier_uniform((num_segments, d_model))

        # Layer norm parameters
        self.gamma   = np.ones(d_model)   # scale — initialized to 1
        self.beta    = np.zeros(d_model)  # shift — initialized to 0

        # ── Gradient accumulators ──
        self.dE_token = np.zeros_like(self.E_token)
        self.dE_pos   = np.zeros_like(self.E_pos)
        self.dE_seg   = np.zeros_like(self.E_seg)
        self.dgamma   = np.zeros_like(self.gamma)
        self.dbeta    = np.zeros_like(self.beta)

        # ── Cache for backprop ──
        self._cache = {}

        print(f"EmbeddingLayer initialized:")
        print(f"  vocab_size    : {vocab_size}")
        print(f"  d_model       : {d_model}")
        print(f"  max_pos       : {max_pos}")
        print(f"  num_segments  : {num_segments}")
        print(f"  Parameters    : {self.count_params():,}")

    def count_params(self) -> int:
        """Total number of learnable parameters."""
        return (
            self.E_token.size +
            self.E_pos.size   +
            self.E_seg.size   +
            self.gamma.size   +
            self.beta.size
        )

    def forward(self,
                token_ids    : np.ndarray,
                positions    : np.ndarray,
                segment_mask : np.ndarray,
                training     : bool = True) -> np.ndarray:
        """
        Forward pass.

        Inputs:
            token_ids    : (batch, seq)  — integer token IDs
            positions    : (batch, seq)  — character offset positions
            segment_mask : (batch, seq)  — 0=document, 1=question

        Output:
            embeddings   : (batch, seq, d_model)

        Math:
            raw[b,t]  = E_token[id[b,t]]
                      + E_pos[pos_bucket[b,t]]
                      + E_seg[seg[b,t]]
            out[b,t]  = LayerNorm(raw[b,t])
        """
        batch_size, seq_len = token_ids.shape

        # ── Step 1: Convert positions to bucket IDs (Vectorized — all B parallel) ──
        # Process all samples' positions simultaneously using batched array ops
        # instead of looping over each sample
        pos_float = positions.astype(np.float32)  # (B, T)
        # Per-sample max position (avoid div-by-zero)
        max_chars = np.max(pos_float, axis=1, keepdims=True) + 1  # (B, 1)
        # Bucketize: floor(pos / max_char * max_pos), clipped to valid range
        pos_buckets = np.floor(pos_float / max_chars * self.max_pos).astype(np.int32)
        pos_buckets = np.clip(pos_buckets, 0, self.max_pos - 1)  # (B, T)

        # ── Step 2: Lookup all three embeddings ──
        # Token lookup: (batch, seq, d_model)
        tok_embed = self.E_token[token_ids]

        # Position lookup: (batch, seq, d_model)
        pos_embed = self.E_pos[pos_buckets]

        # Segment lookup: (batch, seq, d_model)
        seg_clipped = np.clip(segment_mask, 0, self.num_segments - 1)
        seg_embed   = self.E_seg[seg_clipped]

        # ── Step 3: Add all three ──
        # Math: raw = E_token + E_pos + E_seg
        raw = tok_embed + pos_embed + seg_embed    # (batch, seq, d_model)

        # ── Step 4: Layer normalization ──
        out, x_hat, sigma = layer_norm(raw, self.gamma, self.beta)

        # ── Step 5: Dropout (training only) ──
        if training and self.dropout_rate > 0:
            # Inverted dropout — scale by 1/(1-p) to keep expected value same
            mask  = (np.random.rand(*out.shape) > self.dropout_rate).astype(np.float32)
            scale = 1.0 / (1.0 - self.dropout_rate)
            out   = out * mask * scale
        else:
            mask  = np.ones_like(out)
            scale = 1.0

        # ── Cache everything needed for backward ──
        self._cache = {
            'token_ids'   : token_ids,
            'pos_buckets' : pos_buckets,
            'seg_clipped' : seg_clipped,
            'x_hat'       : x_hat,
            'sigma'       : sigma,
            'mask'        : mask,
            'scale'       : scale,
        }

        return out   # (batch, seq, d_model)

    def backward(self, d_out: np.ndarray) -> None:
        """
        Backward pass — compute gradients for all parameters.

        Math (chain rule):

        1. Undo dropout:
            d_out = d_out * mask * scale

        2. Backprop through LayerNorm:
            d_raw, d_gamma, d_beta = layer_norm_backward(d_out)

        3. Backprop through addition (trivial — gradient passes through):
            d_tok_embed = d_raw
            d_pos_embed = d_raw
            d_seg_embed = d_raw

        4. Backprop through embedding lookup (scatter-add):
            For each position t where token_ids[b,t] == v:
                dE_token[v] += d_tok_embed[b,t]

        Scatter-add is just: np.add.at(matrix, indices, gradients)
        """
        cache = self._cache

        # ── Step 1: Undo dropout ──
        d_out = d_out * cache['mask'] * cache['scale']

        # ── Step 2: Backprop through LayerNorm ──
        d_raw, d_gamma, d_beta = layer_norm_backward(
            d_out, cache['x_hat'], cache['sigma'], self.gamma
        )

        # Accumulate LayerNorm gradients
        self.dgamma += d_gamma
        self.dbeta  += d_beta

        # ── Step 3: Gradient of addition = gradient itself ──
        d_tok = d_raw   # (batch, seq, d_model)
        d_pos = d_raw
        d_seg = d_raw

        # ── Step 4: Scatter-add into embedding matrices ──
        # For each token ID — add gradient to that row
        # np.add.at handles repeated indices correctly (accumulates).
        # We flatten the indices and reshape the gradients to (B*T, d_model)
        # to ensure compatibility with both NumPy and CuPy's advanced indexing.
        np.add.at(self.dE_token, cache['token_ids'].ravel(),   d_tok.reshape(-1, self.d_model))
        np.add.at(self.dE_pos,   cache['pos_buckets'].ravel(), d_pos.reshape(-1, self.d_model))
        np.add.at(self.dE_seg,   cache['seg_clipped'].ravel(), d_seg.reshape(-1, self.d_model))

    def zero_grad(self):
        """Reset all gradient accumulators to zero before each batch."""
        self.dE_token[:] = 0
        self.dE_pos[:]   = 0
        self.dE_seg[:]   = 0
        self.dgamma[:]   = 0
        self.dbeta[:]    = 0

    def update(self, lr: float = 0.001,
               clip_norm: float = 1.0):
        """
        SGD update with gradient clipping.

        Math:
            grad_norm = sqrt(sum of all grad^2)
            if grad_norm > clip_norm:
                grads = grads * (clip_norm / grad_norm)
            W = W - lr * grad

        Gradient clipping prevents exploding gradients.
        """
        # Collect all gradients
        grads  = [self.dE_token, self.dE_pos, self.dE_seg,
                  self.dgamma,   self.dbeta]
        params = [self.E_token,  self.E_pos,  self.E_seg,
                  self.gamma,    self.beta]

        # Compute global gradient norm
        grad_norm = np.sqrt(sum(np.sum(g**2) for g in grads))

        # Clip if necessary
        if grad_norm > clip_norm:
            clip_scale = clip_norm / (grad_norm + 1e-8)
        else:
            clip_scale = 1.0

        # Apply updates
        for param, grad in zip(params, grads):
            param -= lr * grad * clip_scale

    def save(self, path: str):
        """Save embedding weights."""
        os.makedirs(path, exist_ok=True)
        np.savez(
            os.path.join(path, 'embedding.npz'),
            E_token = self.E_token,
            E_pos   = self.E_pos,
            E_seg   = self.E_seg,
            gamma   = self.gamma,
            beta    = self.beta,
        )
        config = {
            'vocab_size'   : self.vocab_size,
            'd_model'      : self.d_model,
            'max_pos'      : self.max_pos,
            'num_segments' : self.num_segments,
            'dropout_rate' : self.dropout_rate,
        }
        with open(os.path.join(path, 'embedding_config.json'), 'w') as f:
            json.dump(config, f)
        print(f"Embedding saved to {path}")

    def load(self, path: str):
        """Load embedding weights."""
        data = np.load(os.path.join(path, 'embedding.npz'))
        self.E_token = data['E_token']
        self.E_pos   = data['E_pos']
        self.E_seg   = data['E_seg']
        self.gamma   = data['gamma']
        self.beta    = data['beta']
        print(f"Embedding loaded from {path}")


# ─────────────────────────────────────────────
# MAIN — verify forward and backward pass
# ─────────────────────────────────────────────

if __name__ == '__main__':
    from pdcl_tokenizer import BPETokenizer

    print("Loading tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load('./tokenizer')

    # Build embedding layer
    embed = EmbeddingLayer(
        vocab_size   = tokenizer.vocab_size,
        d_model      = 128,
        max_pos      = 512,
        num_segments = 2,
        dropout_rate = 0.1
    )

    # Load a sample
    with open('./data/train.json') as f:
        data = json.load(f)
    sample = data[0]

    print(f"\nEncoding sample...")
    enc = tokenizer.encode_qa(
        sample['document'],
        sample['question'],
        max_doc_length = 128,
        max_q_length   = 32
    )

    # Create batch of 2 (same sample twice — just for testing)
    token_ids    = np.array([enc['token_ids'],    enc['token_ids']],    dtype=np.int32)
    positions    = np.array([enc['positions'],     enc['positions']],    dtype=np.int32)
    segment_mask = np.array([enc['segment_mask'],  enc['segment_mask']], dtype=np.int32)

    print(f"Batch shape: {token_ids.shape}")

    # ── Forward pass ──
    print("\nRunning forward pass...")
    out = embed.forward(token_ids, positions, segment_mask, training=True)
    print(f"Output shape      : {out.shape}")
    print(f"Output mean       : {out.mean():.6f}")
    print(f"Output std        : {out.std():.6f}")
    print(f"Output min        : {out.min():.6f}")
    print(f"Output max        : {out.max():.6f}")

    # ── Backward pass ──
    print("\nRunning backward pass...")
    embed.zero_grad()
    d_out = np.random.randn(*out.shape) * 0.01  # fake upstream gradient
    embed.backward(d_out)

    print(f"dE_token norm     : {np.linalg.norm(embed.dE_token):.6f}")
    print(f"dE_pos norm       : {np.linalg.norm(embed.dE_pos):.6f}")
    print(f"dE_seg norm       : {np.linalg.norm(embed.dE_seg):.6f}")
    print(f"dgamma norm       : {np.linalg.norm(embed.dgamma):.6f}")
    print(f"dbeta norm        : {np.linalg.norm(embed.dbeta):.6f}")

    # ── Gradient check — numerical vs analytical ──
    print("\nNumerical gradient check on E_token[0]...")
    eps    = 1e-4
    loss_fn = lambda x: np.sum(x ** 2)  # simple test loss

    # Analytical gradient
    embed.zero_grad()
    out1 = embed.forward(token_ids, positions, segment_mask, training=False)
    d_loss = 2 * out1  # gradient of sum(x^2) = 2x
    embed.backward(d_loss)
    analytic_grad = embed.dE_token[token_ids[0, 5]].copy()

    # Numerical gradient
    orig_val = embed.E_token[token_ids[0, 5], 0]
    embed.E_token[token_ids[0, 5], 0] += eps
    out_plus = embed.forward(token_ids, positions, segment_mask, training=False)
    loss_plus = loss_fn(out_plus)

    embed.E_token[token_ids[0, 5], 0] -= 2 * eps
    out_minus = embed.forward(token_ids, positions, segment_mask, training=False)
    loss_minus = loss_fn(out_minus)

    embed.E_token[token_ids[0, 5], 0] = orig_val
    numerical_grad = (loss_plus - loss_minus) / (2 * eps)

    print(f"Analytical grad[0]: {analytic_grad[0]:.6f}")
    print(f"Numerical  grad[0]: {numerical_grad:.6f}")
    print(f"Relative error    : {abs(analytic_grad[0] - numerical_grad) / (abs(numerical_grad) + 1e-8):.6f}")

    # ── Update test ──
    print("\nRunning parameter update...")
    before_norm = np.linalg.norm(embed.E_token)
    embed.update(lr=0.001, clip_norm=1.0)
    after_norm  = np.linalg.norm(embed.E_token)
    print(f"E_token norm before: {before_norm:.6f}")
    print(f"E_token norm after : {after_norm:.6f}")
    print(f"Parameters updated : {'YES' if before_norm != after_norm else 'NO'}")

    # Save
    embed.save('./embedding_weights')

    print(f"\nEmbedding layer ready for PDCL.")
    print(f"Total parameters: {embed.count_params():,}")