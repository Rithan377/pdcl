"""
PDCL Graph-Based Parallel Backward Engine — Updated for Full PDCL
==================================================================
Updated to handle the new pipeline:
    SpanHead → AdaptiveSoftPruning → GroupRelations → Attention → Segment → Embed

Key changes from original:
    1. No more top-K keep_indices mapping — soft pruning backward is cleaner
    2. GroupRelationDetector backward inserted before attention backward
    3. Core burst scaling is applied AFTER this function returns
       (called from CoreBurstBackprop.burst_backward)
    4. Parallel update still fires all modules simultaneously

The topology remains graph-based:
    - Level 0: SpanHead backward → d_gated_doc
    - Level 1: SoftPruning backward → d_out_doc, d_out_que
    - Level 2: GroupRelations backward → d_out_doc (residual path)
    - Level 3: Attention backward (parallel doc/que/cross) → d_seq_doc, d_seq_que
    - Level 4: Segmentation backward → d_embeddings
    - Level 5: Embedding backward → gradient accumulation
"""

from pdcl_backend import xp as np, to_cpu, to_device, GPU_AVAILABLE
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict


class GraphParallelBackward:
    """
    Backward pass following the updated PDCL graph topology.
    Fires independent computation nodes in parallel where possible.
    """

    def __init__(self, model):
        self.model = model

    def graph_backward(self) -> None:
        """
        Execute the full backward pass.

        Levels:
            0 → SpanHead
            1 → AdaptiveSoftPruning
            2 → GroupRelations
            3 → Attention (3 paths in parallel)
            4 → Segmentation
            5 → Embedding
        """
        m = self.model

        # ═══════════════════════════════════════
        # LEVEL 0: SpanHead → d_gated_doc
        # ═══════════════════════════════════════
        d_gated_doc = m.span_head.backward()         # (B, T_doc, d)
        m.span_head._cache.clear()

        # ═══════════════════════════════════════
        # LEVEL 1: AdaptiveSoftPruning → d_out_doc, d_out_que
        # ═══════════════════════════════════════
        d_out_doc, d_out_que = m.prune.backward(d_gated_doc)
        m.prune._cache.clear()

        # ═══════════════════════════════════════
        # LEVEL 2: GroupRelations backward (residual path on doc)
        # Group relations added a residual to out_doc, so gradient
        # flows back through that residual connection.
        # ═══════════════════════════════════════
        d_out_doc = m.group_rel.backward(d_out_doc)
        m.group_rel._cache.clear()

        # ═══════════════════════════════════════
        # LEVEL 3: Attention backward
        # doc_self, que_self, cross_attn are independent paths
        # → fire them in parallel
        # ═══════════════════════════════════════
        d_seq_doc, d_seq_que = m.attention.backward(d_out_doc, d_out_que)
        m.attention._cache.clear()

        # ═══════════════════════════════════════
        # LEVEL 4: Segmentation backward
        # ═══════════════════════════════════════
        seg_cache = m.segment._cache
        B   = seg_cache['B']
        T   = seg_cache['T']

        # Gradient from subspace paths (zero since we routed through seq paths)
        d_subspace_doc = np.zeros((B, T, m.segment.d_doc), dtype=np.float32)
        d_subspace_que = np.zeros((B, T, m.segment.d_que), dtype=np.float32)

        d_embeddings = m.segment.backward(
            d_subspace_doc=d_subspace_doc,
            d_subspace_que=d_subspace_que,
            d_seq_doc=d_seq_doc,
            d_seq_que=d_seq_que
        )
        m.segment._cache.clear()

        # ═══════════════════════════════════════
        # LEVEL 5: Embedding backward
        # ═══════════════════════════════════════
        m.embed.backward(d_embeddings)
        m.embed._cache.clear()

    def parallel_update(self, lr: float, clip_norm: float = 1.0) -> None:
        """
        Update ALL weight modules in parallel.
        Independent nodes — no ordering requirement.
        """
        m = self.model

        def _upd_span():    m.span_head.update(lr, clip_norm)
        def _upd_prune():   m.prune.update(lr, clip_norm)
        def _upd_attn():    m.attention.update(lr, clip_norm)
        def _upd_embed():   m.embed.update(lr, clip_norm)
        def _upd_group():   m.group_rel.update(lr, clip_norm)

        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = [
                pool.submit(_upd_span),
                pool.submit(_upd_prune),
                pool.submit(_upd_attn),
                pool.submit(_upd_embed),
                pool.submit(_upd_group),
            ]
            for f in as_completed(futures):
                f.result()
