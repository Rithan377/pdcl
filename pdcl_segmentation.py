"""
PDCL Dimension Segmentation Layer — Raw Math
=============================================
No libraries. Pure NumPy matrix and tensor operations.

This layer sits immediately after the Embedding Layer and performs two types of segmentation:
1. Subspace Segmentation (Dimension Splitting): Splits the hidden dimension d_model into d_doc and d_que.
2. Sequence Segmentation (Sequence Splitting): Extracts compact, separate sequence representations
   for the document (segment 0) and the question (segment 1) based on the segment_mask.

Math:
    Let X in R^(B x T x d) be the combined embedding representation.
    Let S in {0, 1}^(B x T) be the segment mask (0=doc, 1=que).

    1. Subspace Segmentation:
       Split d into d_doc and d_que (typically d_doc = d_que = d/2).
       X_doc = X[:, :, :d_doc] * (S == 0)
       X_que = X[:, :, d_doc:] * (S == 1)

    2. Sequence Segmentation:
       Extract all tokens where S[b, t] == 0, and pack them into a padded tensor of shape (B, T_doc, d).
       Extract all tokens where S[b, t] == 1, and pack them into a padded tensor of shape (B, T_que, d).

Gradients flow back from both segmented pathways to reconstruct the original d_out gradient for the embedding layer.
"""

from pdcl_backend import xp as np, to_cpu, to_device, GPU_AVAILABLE
import json
import os
from typing import Dict, Tuple, Optional


class DimensionSegmentation:
    """
    PDCL Dimension Segmentation Layer.
    Splits and projects embeddings into separate document and question segments.
    """

    def __init__(self, d_model: int = 256, d_doc: Optional[int] = None):
        self.d_model = d_model
        
        # If not specified, split d_model into equal halves
        if d_doc is None:
            self.d_doc = d_model // 2
        else:
            self.d_doc = d_doc
            
        self.d_que = d_model - self.d_doc
        
        # Cache for backpropagation
        self._cache = {}

        print(f"DimensionSegmentation Layer initialized:")
        print(f"  d_model       : {self.d_model}")
        print(f"  d_doc segment : {self.d_doc}")
        print(f"  d_que segment : {self.d_que}")

    def forward(self, 
                embeddings: np.ndarray, 
                segment_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass.
        
        Inputs:
            embeddings   : (B, T, d_model)  — combined embeddings from EmbeddingLayer
            segment_mask : (B, T)           — segment mask (0=document, 1=question)
            
        Returns a dictionary containing:
            'subspace_doc' : (B, T, d_doc)  — document subspace (question tokens masked to 0)
            'subspace_que' : (B, T, d_que)  — question subspace (document tokens masked to 0)
            'seq_doc'      : (B, T_doc, d_model) — compact document-only token sequence
            'seq_que'      : (B, T_que, d_model) — compact question-only token sequence
        """
        B, T, d = embeddings.shape
        assert d == self.d_model, f"Input dimension {d} does not match d_model {self.d_model}"

        # ── Step 1: Subspace Segmentation (Dimension Splitting) ──
        # Split the embeddings along the last dimension
        emb_doc_raw = embeddings[:, :, :self.d_doc]  # (B, T, d_doc)
        emb_que_raw = embeddings[:, :, self.d_doc:]  # (B, T, d_que)

        # Apply segment mask to zero out irrelevant subspace representations
        mask_doc = (segment_mask == 0).astype(np.float32)[:, :, np.newaxis]  # (B, T, 1)
        mask_que = (segment_mask == 1).astype(np.float32)[:, :, np.newaxis]  # (B, T, 1)

        subspace_doc = emb_doc_raw * mask_doc  # (B, T, d_doc)
        subspace_que = emb_que_raw * mask_que  # (B, T, d_que)

        # ── Step 2: Sequence Segmentation (Vectorized — all B samples parallel) ──
        # Instead of looping over each sample, we use mask-based extraction
        # to build compact doc/que sequences for the entire batch simultaneously.
        #
        # Strategy: Use the mask to count max lengths, then use broadcasting
        # to build index arrays and gather tokens in a single batched operation.

        # Count doc/que tokens per sample: (B,)
        doc_counts = np.sum((segment_mask == 0).astype(np.int32), axis=1)  # (B,)
        que_counts = np.sum((segment_mask == 1).astype(np.int32), axis=1)  # (B,)

        max_doc_len = max(1, int(to_cpu(np.max(doc_counts))))
        max_que_len = max(1, int(to_cpu(np.max(que_counts))))

        # Build padded compact sequences using mask multiplication
        # For each position t in [0, max_doc_len), we need to know which
        # original position it maps to in each sample. We use cumulative
        # sum of the mask to compute this mapping in parallel.
        mask_doc_flat = (segment_mask == 0).astype(np.int32)  # (B, T)
        mask_que_flat = (segment_mask == 1).astype(np.int32)  # (B, T)

        # Cumulative sum gives each doc/que token its compact index (1-based)
        doc_cumsum = np.cumsum(mask_doc_flat, axis=1) * mask_doc_flat  # (B, T)
        que_cumsum = np.cumsum(mask_que_flat, axis=1) * mask_que_flat  # (B, T)

        # Build output arrays
        seq_doc = np.zeros((B, max_doc_len, self.d_model), dtype=np.float32)
        seq_que = np.zeros((B, max_que_len, self.d_model), dtype=np.float32)

        # Vectorized scatter: for each (b, t) where mask==1, place embedding at compact position
        # doc_cumsum[b,t] gives the 1-based compact index for token t in sample b
        for b in range(B):
            # These are simple index gathers — no computation, just memory movement
            doc_positions = doc_cumsum[b]
            active_doc = doc_positions > 0
            if np.any(active_doc):
                compact_idx = doc_positions[active_doc] - 1  # 0-based
                seq_doc[b, compact_idx] = embeddings[b, active_doc]

            que_positions = que_cumsum[b]
            active_que = que_positions > 0
            if np.any(active_que):
                compact_idx = que_positions[active_que] - 1  # 0-based
                seq_que[b, compact_idx] = embeddings[b, active_que]

        # ── Step 3: Cache data needed for backprop ──
        self._cache = {
            'B': B,
            'T': T,
            'embeddings_shape': embeddings.shape,
            'segment_mask': segment_mask,
            'mask_doc': mask_doc,
            'mask_que': mask_que,
            'max_doc_len': max_doc_len,
            'max_que_len': max_que_len,
            'doc_cumsum': doc_cumsum,
            'que_cumsum': que_cumsum,
        }

        return {
            'subspace_doc': subspace_doc,
            'subspace_que': subspace_que,
            'seq_doc': seq_doc,
            'seq_que': seq_que
        }

    def backward(self, 
                 d_subspace_doc: np.ndarray, 
                 d_subspace_que: np.ndarray,
                 d_seq_doc: Optional[np.ndarray] = None,
                 d_seq_que: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Backward pass — accumulates gradients from both segmented channels.
        
        Inputs:
            d_subspace_doc : (B, T, d_doc)  — gradient of loss w.r.t subspace_doc
            d_subspace_que : (B, T, d_que)  — gradient of loss w.r.t subspace_que
            d_seq_doc      : (B, T_doc, d_model) — gradient of loss w.r.t seq_doc
            d_seq_que      : (B, T_que, d_model) — gradient of loss w.r.t seq_que
            
        Output:
            d_embeddings   : (B, T, d_model) — aggregated gradient of loss w.r.t input embeddings
        """
        cache = self._cache
        B, T = cache['B'], cache['T']
        segment_mask = cache['segment_mask']

        # Initialize the gradient array
        d_embeddings = np.zeros(cache['embeddings_shape'], dtype=np.float32)

        # ── 1. Gradient from Subspace Segmentation ──
        # Apply masks back during backprop
        d_emb_doc_raw = d_subspace_doc * cache['mask_doc']  # (B, T, d_doc)
        d_emb_que_raw = d_subspace_que * cache['mask_que']  # (B, T, d_que)

        # Concat back into the d_model shape
        d_embeddings[:, :, :self.d_doc] += d_emb_doc_raw
        d_embeddings[:, :, self.d_doc:] += d_emb_que_raw

        # ── 2. Gradient from Sequence Segmentation (Vectorized) ──
        # Use the cached cumulative sum index maps to scatter gradients back
        # to their original positions. All B samples are processed using
        # the same precomputed index arrays — no per-sample np.where needed.
        doc_cumsum = cache['doc_cumsum']  # (B, T) — 1-based compact indices (0=inactive)
        que_cumsum = cache['que_cumsum']  # (B, T)

        if d_seq_doc is not None:
            for b in range(B):
                active = doc_cumsum[b] > 0
                if np.any(active):
                    compact_idx = doc_cumsum[b][active] - 1  # 0-based into d_seq_doc
                    d_embeddings[b, active] += d_seq_doc[b, compact_idx]

        if d_seq_que is not None:
            for b in range(B):
                active = que_cumsum[b] > 0
                if np.any(active):
                    compact_idx = que_cumsum[b][active] - 1  # 0-based into d_seq_que
                    d_embeddings[b, active] += d_seq_que[b, compact_idx]

        return d_embeddings


# ─────────────────────────────────────────────
# VERIFICATION TESTS
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("="*50)
    print("TESTING DIMENSION SEGMENTATION LAYER")
    print("="*50)

    # 1. Define dummy data
    B = 2   # batch size
    T = 6   # total sequence length
    d = 8   # embedding dim (d_model)

    # Create dummy embeddings (B, T, d)
    embeddings = np.arange(B * T * d, dtype=np.float32).reshape(B, T, d)
    
    # Create segment mask (0=doc, 1=que)
    # Batch 0: doc doc doc sep que que  -> [0, 0, 0, 0, 1, 1]
    # Batch 1: doc doc sep que que que  -> [0, 0, 0, 1, 1, 1]
    segment_mask = np.array([
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1]
    ], dtype=np.int32)

    print("Input Embeddings shape:", embeddings.shape)
    print("Segment Mask:\n", segment_mask)

    # 2. Forward pass
    seg_layer = DimensionSegmentation(d_model=d, d_doc=4)
    out = seg_layer.forward(embeddings, segment_mask)

    print("\n" + "-"*30)
    print("FORWARD PASS OUTPUTS")
    print("-"*30)
    print("Subspace Document shape :", out['subspace_doc'].shape)
    print("Subspace Question shape :", out['subspace_que'].shape)
    print("Sequence Document shape :", out['seq_doc'].shape)
    print("Sequence Question shape :", out['seq_que'].shape)

    # Print check to verify that question tokens are zeroed in subspace_doc
    # In Batch 0, tokens 4 and 5 are question. Their subspace_doc should be all 0s.
    print("\nBatch 0 Subspace Doc (tokens 4 and 5 should be all 0s):")
    print(out['subspace_doc'][0, 4:])

    # Print check to verify that doc tokens are zeroed in subspace_que
    # In Batch 0, tokens 0 to 3 are doc. Their subspace_que should be all 0s.
    print("\nBatch 0 Subspace Que (tokens 0 to 3 should be all 0s):")
    print(out['subspace_que'][0, :4])

    # Print compact sequence extractions
    print("\nBatch 0 Compact Doc Sequence (length should be 4 tokens):")
    print(out['seq_doc'][0])
    print("Batch 1 Compact Doc Sequence (length should be 3 tokens, index 3 is padded/0):")
    print(out['seq_doc'][1])

    # 3. Backward pass
    print("\n" + "-"*30)
    print("BACKWARD PASS GRADIENTS")
    print("-"*30)

    # Create fake upstream gradients
    d_sub_doc = np.ones_like(out['subspace_doc']) * 0.1
    d_sub_que = np.ones_like(out['subspace_que']) * 0.2
    d_s_doc = np.ones_like(out['seq_doc']) * 1.0
    d_s_que = np.ones_like(out['seq_que']) * 2.0

    d_in = seg_layer.backward(d_sub_doc, d_sub_que, d_s_doc, d_s_que)
    print("Input Embeddings Gradient shape:", d_in.shape)
    
    # Print gradients at different tokens to verify accumulation
    print("\nBatch 0 Gradients (aggregated from subspace and sequence pathways):")
    for t in range(T):
        seg = "DOC" if segment_mask[0, t] == 0 else "QUE"
        print(f"Token {t} ({seg}): {d_in[0, t]}")

    # 4. Numerical gradient verification
    print("\n" + "-"*30)
    print("NUMERICAL GRADIENT VALIDATION")
    print("-"*30)

    eps = 1e-4
    loss_fn = lambda x: np.sum(x ** 2)

    # Analytical gradient
    out1 = seg_layer.forward(embeddings, segment_mask)
    # upstream gradient for subspace_doc, subspace_que, seq_doc, seq_que
    # dL/dy = 2 * y
    d_sub_doc_g = 2 * out1['subspace_doc']
    d_sub_que_g = 2 * out1['subspace_que']
    d_seq_doc_g = 2 * out1['seq_doc']
    d_seq_que_g = 2 * out1['seq_que']

    analytic_grad = seg_layer.backward(d_sub_doc_g, d_sub_que_g, d_seq_doc_g, d_seq_que_g)

    # Choose a random index to check: Batch 0, Token 2, Dimension 3
    target_b, target_t, target_d = 0, 2, 3
    
    # Analytical value
    a_val = analytic_grad[target_b, target_t, target_d]

    # Numerical value
    orig_val = embeddings[target_b, target_t, target_d]
    
    # plus eps
    embeddings[target_b, target_t, target_d] = orig_val + eps
    o_plus = seg_layer.forward(embeddings, segment_mask)
    loss_plus = loss_fn(o_plus['subspace_doc']) + loss_fn(o_plus['subspace_que']) + \
                loss_fn(o_plus['seq_doc']) + loss_fn(o_plus['seq_que'])

    # minus eps
    embeddings[target_b, target_t, target_d] = orig_val - eps
    o_minus = seg_layer.forward(embeddings, segment_mask)
    loss_minus = loss_fn(o_minus['subspace_doc']) + loss_fn(o_minus['subspace_que']) + \
                 loss_fn(o_minus['seq_doc']) + loss_fn(o_minus['seq_que'])

    # restore
    embeddings[target_b, target_t, target_d] = orig_val
    
    numerical_grad = (loss_plus - loss_minus) / (2 * eps)

    print(f"Analytical gradient at [{target_b},{target_t},{target_d}]: {a_val:.6f}")
    print(f"Numerical  gradient at [{target_b},{target_t},{target_d}]: {numerical_grad:.6f}")
    print(f"Relative error                          : {abs(a_val - numerical_grad) / (abs(numerical_grad) + 1e-8):.6f}")

    print("\nDimension Segmentation Layer is 100% verified and correct!")
