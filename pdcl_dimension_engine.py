"""
PDCL Parallel Dimension Engine — Updated with Cross-Dimension Graph
====================================================================
Manages D processing dimensions, each handling a partition of the dataset.

Key PDCL principle: Data is segmented across dimensions. Each dimension
learns from its territory independently, processes it in parallel,
and fires its own backpropagation burst simultaneously.

Updates from original:
    1. CrossDimensionGraph integration — tracks activations per dimension
    2. Graph-weighted gradient aggregation (instead of uniform 1/D averaging)
    3. Activation tracking passed to graph builder after each forward
    4. Feature pruning masks applied to cloned models each step

Compiler-style parallelism:
    - All D dimensions process simultaneously (ThreadPoolExecutor)
    - All samples within each dimension processed simultaneously (batch)
    - Gradient aggregation uses graph-learned weights (not uniform)
    - Master model updated once with aggregated gradients
"""

import json, copy, time
from pdcl_backend import xp as np, to_cpu, to_device, CPU_NP, GPU_AVAILABLE
from pdcl_tokenizer import BPETokenizer
from pdcl_burst_backprop import CoreBurstBackprop
from pdcl_graph_formation import CrossDimensionGraph
from pdcl_utils import find_answer_in_doc_tokens, fuzzy_find_answer
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional
import numpy as _cpu_np

# CPU numpy alias
try:
    from pdcl_backend import cpu_np as CPU_NP
except ImportError:
    import numpy as CPU_NP


# ─────────────────────────────────────────────
# ANSWER ALIGNMENT HELPER — WITH FUZZY MATCHING
# ─────────────────────────────────────────────

class AnswerAlignmentError(Exception):
    """Raised when answer cannot be reliably aligned to document tokens."""
    pass


def align_answer_to_tokens(sample: dict,
                            tokenizer: BPETokenizer,
                            max_doc: int = 256,
                            max_q: int = 32,
                            skip_on_failure: bool = True) -> Tuple[dict, int, int]:
    """
    Encode document + question and find the token-level positions
    of the answer span within the doc sequence using fuzzy matching.
    
    Handles OCR noise and number format variations (e.g., "5,000,000" vs "5,0O0,000").

    Args:
        sample           : Dict with 'document', 'question', 'answer' fields
        tokenizer        : BPE tokenizer
        max_doc          : Max document tokens
        max_q            : Max question tokens
        skip_on_failure  : If True, raise AnswerAlignmentError on bad alignment
                          (upstream code skips this sample). If False, raise anyway.

    Returns:
        (qa_enc, ans_start_doc, ans_end_doc)
        qa_enc        : full encoding dict from tokenizer
        ans_start_doc : start index in doc token sequence (0-based)
        ans_end_doc   : end index in doc token sequence (0-based, inclusive)

    Raises:
        AnswerAlignmentError: If answer cannot be reliably found (alignment confidence too low)
    """
    qa_enc = tokenizer.encode_qa(
        sample['document'], sample['question'],
        max_doc_length=max_doc, max_q_length=max_q
    )

    doc_length = qa_enc['doc_length']
    answer = sample['answer'].lower().strip()

    if len(answer) == 0:
        raise AnswerAlignmentError("Answer is empty")

    # Encode answer to get token count
    ans_ids, _ = tokenizer.encode(answer, max_length=None)
    ans_len = max(1, len([x for x in ans_ids if x != tokenizer.token2id.get('<pad>', 0)]))

    # Extract doc token ids and their character positions
    doc_ids = qa_enc['token_ids'][1: doc_length + 1]  # Skip <doc> token
    doc_positions = qa_enc['positions'][1: doc_length + 1]

    # Strategy 1: Try exact token sequence match + accurate fuzzy fallback
    # find_answer_in_doc_tokens now handles both exact token matching AND
    # proper character-to-token mapping via per-token decoding.
    ans_start_doc, ans_end_doc = find_answer_in_doc_tokens(ans_ids, doc_ids, tokenizer)
    if ans_start_doc >= 0:
        # Verify: decode the span and check it actually contains the answer
        span_ids = doc_ids[ans_start_doc: ans_end_doc + 1]
        span_decoded = tokenizer.decode(span_ids).replace(" ", "").lower()
        answer_clean = answer.replace(" ", "").lower()
        if answer_clean in span_decoded or span_decoded in answer_clean:
            return qa_enc, ans_start_doc + 1, ans_end_doc + 1
        # If verification fails, fall through to skip

    # Strategy 2: No reliable alignment found
    # Raise exception — upstream training loop will skip this sample
    raise AnswerAlignmentError(
        f"Could not align answer '{answer}' to document tokens. "
        f"Exact match failed, fuzzy match failed (edit distance > 3). "
        f"This sample will be skipped."
    )


# ─────────────────────────────────────────────
# GRADIENT SYNCHRONIZATION HELPERS
# ─────────────────────────────────────────────

def _zero_master_grad(master: CoreBurstBackprop) -> None:
    master.embed.zero_grad()
    master.attention.zero_grad()
    master.prune.zero_grad()
    master.group_rel.zero_grad()
    master.span_head.zero_grad()


def _accumulate_gradients(master: CoreBurstBackprop,
                           clone: CoreBurstBackprop,
                           weight: float) -> None:
    """Add clone's gradients * weight into master's gradient buffers."""

    def _add(m_buf, c_buf):
        m_buf += to_device(_cpu_np.array(to_cpu(c_buf)) * weight)

    # Embedding
    _add(master.embed.dE_token, clone.embed.dE_token)
    _add(master.embed.dE_pos,   clone.embed.dE_pos)
    _add(master.embed.dE_seg,   clone.embed.dE_seg)
    _add(master.embed.dgamma,   clone.embed.dgamma)
    _add(master.embed.dbeta,    clone.embed.dbeta)

    # Attention — doc self
    for m_a, c_a in [
        (master.attention.doc_self_attn,      clone.attention.doc_self_attn),
        (master.attention.que_self_attn,      clone.attention.que_self_attn),
        (master.attention.que_doc_cross_attn, clone.attention.que_doc_cross_attn),
    ]:
        _add(m_a.dW_q, c_a.dW_q)
        _add(m_a.dW_k, c_a.dW_k)
        _add(m_a.dW_v, c_a.dW_v)
        _add(m_a.dW_o, c_a.dW_o)

    _add(master.attention.dW_fusion, clone.attention.dW_fusion)

    # Pruning
    _add(master.prune.dW_d, clone.prune.dW_d)
    _add(master.prune.dW_q, clone.prune.dW_q)

    # Span head
    _add(master.span_head.dv_start, clone.span_head.dv_start)
    _add(master.span_head.dv_end,   clone.span_head.dv_end)

    # Group relations
    _add(master.group_rel.dW1, clone.group_rel.dW1)
    _add(master.group_rel.dW2, clone.group_rel.dW2)


def _collect_grad_arrays(model: CoreBurstBackprop) -> list:
    """Collect references to all gradient arrays from a model."""
    grads = []
    # Embedding
    grads.append(model.embed.dE_token)
    grads.append(model.embed.dE_pos)
    grads.append(model.embed.dE_seg)
    grads.append(model.embed.dgamma)
    grads.append(model.embed.dbeta)
    # Attention — doc self, que self, cross
    for attn in [model.attention.doc_self_attn,
                 model.attention.que_self_attn,
                 model.attention.que_doc_cross_attn]:
        grads.append(attn.dW_q)
        grads.append(attn.dW_k)
        grads.append(attn.dW_v)
        grads.append(attn.dW_o)
    grads.append(model.attention.dW_fusion)
    # Pruning
    grads.append(model.prune.dW_d)
    grads.append(model.prune.dW_q)
    # Span head
    grads.append(model.span_head.dv_start)
    grads.append(model.span_head.dv_end)
    # Group relations
    grads.append(model.group_rel.dW1)
    grads.append(model.group_rel.dW2)
    return grads


def _pcgrad_accumulate_all(master: CoreBurstBackprop,
                           clones: list,
                           weights: list) -> None:
    """
    PCGrad: Project Conflicting Gradients before accumulation.

    For each gradient parameter:
      1. Collect the gradient from all clones.
      2. For each clone i, project its gradient against all other clones j:
         If dot(g_i, g_j) < 0 (conflict), remove the conflicting component.
      3. Accumulate the projected gradients (weighted) into the master.

    This prevents the tug-of-war between clones processing different data shards.
    Reference: Yu et al., "Gradient Surgery for Multi-Task Learning" (NeurIPS 2020)
    """
    n_dims = len(clones)
    master_grads = _collect_grad_arrays(master)
    clone_grads_all = [_collect_grad_arrays(clone) for clone in clones]
    n_params = len(master_grads)

    for p in range(n_params):
        # Collect this param's gradient from each clone (CPU numpy copies)
        raw_grads = []
        for d in range(n_dims):
            g = _cpu_np.array(to_cpu(clone_grads_all[d][p]), dtype=_cpu_np.float32).copy()
            raw_grads.append(g)

        # PCGrad projection: for each clone, project against all others
        projected = [g.copy() for g in raw_grads]
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue
                g_j_flat = raw_grads[j].ravel()
                p_i_flat = projected[i].ravel()
                dot = float(_cpu_np.dot(p_i_flat, g_j_flat))
                if dot < 0.0:  # Conflict detected
                    norm_sq = float(_cpu_np.dot(g_j_flat, g_j_flat)) + 1e-8
                    # Remove the conflicting component
                    projected[i] = projected[i] - (dot / norm_sq) * raw_grads[j]

        # Accumulate projected gradients into master
        for d in range(n_dims):
            master_grads[p] += to_device(projected[d] * float(weights[d]))


def _sync_weights_to_clone(master: CoreBurstBackprop,
                             clone: CoreBurstBackprop) -> None:
    """Copy master weights into a dimension clone before each step."""
    def _cp(m_p, c_p):
        c_p[:] = m_p

    # Embedding
    _cp(master.embed.E_token, clone.embed.E_token)
    _cp(master.embed.E_pos,   clone.embed.E_pos)
    _cp(master.embed.E_seg,   clone.embed.E_seg)
    _cp(master.embed.gamma,   clone.embed.gamma)
    _cp(master.embed.beta,    clone.embed.beta)

    # Attention
    for m_a, c_a in [
        (master.attention.doc_self_attn,      clone.attention.doc_self_attn),
        (master.attention.que_self_attn,      clone.attention.que_self_attn),
        (master.attention.que_doc_cross_attn, clone.attention.que_doc_cross_attn),
    ]:
        _cp(m_a.W_q, c_a.W_q); _cp(m_a.W_k, c_a.W_k)
        _cp(m_a.W_v, c_a.W_v); _cp(m_a.W_o, c_a.W_o)

    _cp(master.attention.W_fusion, clone.attention.W_fusion)
    _cp(master.prune.W_d,     clone.prune.W_d)
    _cp(master.prune.W_q,     clone.prune.W_q)
    _cp(master.prune.log_k,   clone.prune.log_k)
    _cp(master.span_head.v_start, clone.span_head.v_start)
    _cp(master.span_head.v_end,   clone.span_head.v_end)
    _cp(master.group_rel.W1,  clone.group_rel.W1)
    _cp(master.group_rel.W2,  clone.group_rel.W2)

    # Sync epoch for group relations
    clone.current_epoch = master.current_epoch


# ─────────────────────────────────────────────
# PARALLEL DIMENSION ENGINE
# ─────────────────────────────────────────────

class ParallelDimensionEngine:
    """
    Manages D dimension clones, distributes data, runs forward+backward
    simultaneously, aggregates gradients with graph-learned weights,
    and updates the master model.
    
    SUPPORTS TWO MODES:
        1. Legacy: dataset as List[dict] → tokenize each step (slower, no GPU memory)
        2. Optimized: pre_tokenized_dataset from PreTokenizedDataset → index GPU arrays (faster)
    """

    def __init__(self,
                 n_dimensions: int,
                 master_model: CoreBurstBackprop,
                 dataset: List[dict] = None,
                 tokenizer: BPETokenizer = None,
                 max_doc: int = 256,
                 max_q: int = 32,
                 pre_tokenized_dataset = None):
        """
        Args:
            n_dimensions: Number of processing dimensions
            master_model: Master model to sync from
            dataset: List of sample dicts (legacy mode, optional if pre_tokenized_dataset provided)
            tokenizer: BPE tokenizer (required for legacy mode)
            max_doc: Max document tokens
            max_q: Max question tokens
            pre_tokenized_dataset: PreTokenizedDataset object (new optimized mode)
        """

        self.n_dims = n_dimensions
        self.master = master_model
        self.tokenizer = tokenizer
        self.max_doc = max_doc
        self.max_q = max_q

        # Detect mode
        self.use_pre_tokenized = pre_tokenized_dataset is not None
        self.pre_tokenized = pre_tokenized_dataset

        if self.use_pre_tokenized:
            # New optimized mode: data already on GPU
            # Partition sample indices across dimensions
            all_indices = self.pre_tokenized.get_sample_indices()
            self.dim_indices = self._partition_indices(all_indices)
            self.dim_data = None  # Not used in pre-tokenized mode
        else:
            # Legacy mode: tokenize on the fly
            if dataset is None or tokenizer is None:
                raise ValueError(
                    "Legacy mode requires dataset and tokenizer. "
                    "Or provide pre_tokenized_dataset for optimized mode."
                )
            # Partition dataset across dimensions
            self.dim_data = self._partition_dataset(dataset)
            self.dim_indices = None

        # Create dimension clones (share vocab_size, d_model etc.)
        self.dim_models = [
            CoreBurstBackprop(
                vocab_size=master_model.embed.vocab_size,
                max_positions=master_model.embed.max_pos,
                d_model=master_model.d_model,
                n_heads=master_model.attention.doc_self_attn.n_heads,
                burst_beta=master_model.burst_beta,
            )
            for _ in range(n_dimensions)
        ]

        # Cross-dimension graph for weighted aggregation
        self.graph = CrossDimensionGraph(
            n_dimensions  = n_dimensions,
            d_model       = master_model.d_model,
            min_epochs_before_graph=2
        )

        # Sample index trackers (for cycling through data each epoch)
        self.dim_batch_idx = [0] * n_dimensions

        print(f"ParallelDimensionEngine initialized:")
        print(f"  Dimensions   : {n_dimensions}")
        if self.use_pre_tokenized:
            print(f"  Mode         : Pre-tokenized (GPU) ✓")
            print(f"  Data splits  : {[len(d) for d in self.dim_indices]}")
        else:
            print(f"  Mode         : Legacy (tokenize per step)")
            print(f"  Data splits  : {[len(d) for d in self.dim_data]}")

    def _partition_indices(self, indices: List[int]) -> List[List[int]]:
        """Partition sample indices across n_dims dimensions."""
        partitions = [[] for _ in range(self.n_dims)]
        for i, idx in enumerate(indices):
            partitions[i % self.n_dims].append(idx)
        return partitions

    def _partition_dataset(self, dataset: List[dict]) -> List[List[dict]]:
        """Split dataset into n_dims partitions."""
        partitions = [[] for _ in range(self.n_dims)]
        for i, sample in enumerate(dataset):
            partitions[i % self.n_dims].append(sample)
        return partitions

    def _prepare_batch(self,
                       samples: List[dict] = None,
                       indices: List[int] = None) -> Optional[Tuple]:
        """
        Prepare a batch of samples for forward pass.
        
        Args:
            samples: List of sample dicts (legacy mode)
            indices: List of sample indices (pre-tokenized mode)
            
        Returns:
            (token_ids, positions, segment_mask, start_t, end_t) — all GPU/CPU arrays
        """
        
        # Mode 1: Pre-tokenized (GPU) — pure indexing
        if indices is not None and self.use_pre_tokenized:
            return self.pre_tokenized.get_batch(indices)

        # Mode 2: Legacy (tokenize on the fly)
        if samples is None:
            return None

        all_ids, all_pos, all_seg, all_s, all_e = [], [], [], [], []

        for s in samples:
            try:
                qa_enc, ans_s, ans_e = align_answer_to_tokens(
                    s, self.tokenizer, self.max_doc, self.max_q
                )
                all_ids.append(qa_enc['token_ids'])
                all_pos.append(qa_enc['positions'])
                all_seg.append(qa_enc['segment_mask'])
                all_s.append(ans_s)
                all_e.append(ans_e)
            except AnswerAlignmentError:
                continue

        if not all_ids:
            return None

        max_len = max(len(x) for x in all_ids)
        pad_id = self.tokenizer.token2id.get('<pad>', 0)

        def pad(lst, pad_val=0):
            return [x + [pad_val] * (max_len - len(x)) for x in lst]

        token_ids    = np.array(pad(all_ids, pad_id),  dtype=_cpu_np.int32)
        positions    = np.array(pad(all_pos, 0),        dtype=_cpu_np.int32)
        segment_mask = np.array(pad(all_seg, 0),        dtype=_cpu_np.int32)
        start_t      = np.array(all_s, dtype=_cpu_np.int32)
        end_t        = np.array(all_e, dtype=_cpu_np.int32)

        return token_ids, positions, segment_mask, start_t, end_t

    def _run_dimension(self,
                       dim_id: int,
                       batch_size: int = 4) -> Tuple[float, CoreBurstBackprop]:
        """
        Run one forward + backward on a dimension clone.
        
        Supports both:
            - Legacy: Get samples from self.dim_data, tokenize them
            - Pre-tokenized: Get indices from self.dim_indices, index GPU arrays
        
        Returns (loss, clone_with_gradients).
        """
        clone = self.dim_models[dim_id]

        # Sync weights from master
        _sync_weights_to_clone(self.master, clone)
        clone.zero_grad()

        if self.use_pre_tokenized:
            # Pre-tokenized mode: index GPU arrays
            indices = self.dim_indices[dim_id]
            if not indices:
                return 0.0, clone

            start = self.dim_batch_idx[dim_id] % len(indices)
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]
            self.dim_batch_idx[dim_id] = end % len(indices)

            # Convert to numpy array for indexing
            import numpy as _np_cpu
            batch_indices_np = _np_cpu.array(batch_indices, dtype=_np_cpu.int32)
            prepared = self._prepare_batch(indices=batch_indices_np)
        else:
            # Legacy mode: tokenize on the fly
            data = self.dim_data[dim_id]
            if not data:
                return 0.0, clone

            start = self.dim_batch_idx[dim_id] % len(data)
            end = min(start + batch_size, len(data))
            batch = data[start:end]
            self.dim_batch_idx[dim_id] = end % len(data)

            prepared = self._prepare_batch(samples=batch)

        if prepared is None:
            return 0.0, clone

        token_ids, positions, segment_mask, start_t, end_t = prepared

        # Forward
        _, _, loss = clone.forward(
            token_ids=token_ids,
            positions=positions,
            segment_mask=segment_mask,
            start_targets=start_t,
            end_targets=end_t,
            training=True
        )

        # Track activations for cross-dimension graph
        # Use the attention output before pruning as the activation signal
        if hasattr(clone.attention.doc_self_attn, '_cache') and clone.attention.doc_self_attn._cache:
            doc_repr = clone.segment._cache.get('embeddings_shape')
            # Use embedding output as activation proxy
            # (attn output already cleared — use segment output shape)
            # Simple proxy: use prune input scores as activation signal
            try:
                scores = clone.prune._cache.get('scores')
                if scores is not None:
                    proxy = _cpu_np.abs(to_cpu(scores))  # (B, T_doc)
                    # Expand to d_model shape by tiling
                    proxy_d = _cpu_np.tile(proxy.mean(axis=0)[:, None],
                                           (1, clone.d_model)).mean(axis=0)
                    self.graph.update_activations(dim_id, to_device(proxy_d[None, None, :]))
            except Exception:
                pass

        # Backward
        clone.burst_backward()

        return float(loss), clone

    def execute_parallel_step(self,
                               lr: float,
                               clip_norm: float = 1.0,
                               batch_size: int = 4) -> Tuple[float, List[float]]:
        """
        Run all D dimensions simultaneously.
        Aggregate gradients with graph-weighted averaging.
        Update master model.

        Returns:
            global_loss : weighted average loss across dimensions
            dim_losses  : per-dimension loss list
        """
        results     = {}
        dim_losses  = [0.0] * self.n_dims

        # ── Fire all dimensions in parallel ──
        with ThreadPoolExecutor(max_workers=self.n_dims) as pool:
            futures = {
                pool.submit(self._run_dimension, d, batch_size): d
                for d in range(self.n_dims)
            }
            for future in as_completed(futures):
                d = futures[future]
                try:
                    loss, clone = future.result()
                    results[d] = (loss, clone)
                    dim_losses[d] = loss
                except Exception as ex:
                    print(f"  Dimension {d} error: {ex}")
                    results[d] = (0.0, self.dim_models[d])

        # ── Graph-weighted gradient aggregation ──
        agg_weights = self.graph.get_aggregation_weights()  # (n_dims,)

        _zero_master_grad(self.master)
        for d in range(self.n_dims):
            _, clone = results[d]
            _accumulate_gradients(self.master, clone, weight=float(agg_weights[d]))

        # ── Single synchronized master update ──
        self.master.update(lr=lr, clip_norm=clip_norm)

        global_loss = float(_cpu_np.dot(
            _cpu_np.array(dim_losses), agg_weights
        ))

        return global_loss, dim_losses

    def update_graph(self, epoch: int) -> Dict:
        """
        Called at epoch boundaries.
        Updates cross-dimension correlation graph.
        """
        corr = self.graph.compute_graph(epoch)
        summary = self.graph.get_graph_summary()
        return summary
