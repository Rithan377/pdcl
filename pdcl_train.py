"""
PDCL Full Training Loop — All PDCL Components Active
======================================================
Implements the complete PDCL training procedure:

Per-step operations (every batch):
    1. All D dimensions process simultaneously (compiler-style)
    2. Forward → Adaptive soft pruning → Span prediction
    3. Group relation detection (active from epoch 3)
    4. Burst backward with core burst gradient scaling
    5. Graph-weighted gradient aggregation and master update

Per-epoch operations (at epoch boundaries):
    1. Feature dimension pruning — prune low-importance features
    2. Connection pruning — zero persistent weak attention connections
    3. Cross-dimension graph update — recompute correlation weights
    4. Sparsity reporting

Validation:
    - No teacher forcing (soft pruning handles this naturally)
    - Exact match and F1 over full doc sequence predictions
"""

import json, time, os
from pdcl_backend import xp as np, to_cpu, to_device, GPU_AVAILABLE
from pdcl_tokenizer import BPETokenizer
from pdcl_burst_backprop import CoreBurstBackprop
from pdcl_dimension_engine import (
    ParallelDimensionEngine, align_answer_to_tokens, AnswerAlignmentError
)
from pdcl_feature_pruning import FeatureDimensionPruner
from pdcl_utils import PDCLConfig, validate_config, PreTokenizedDataset
from typing import List, Dict, Tuple
import numpy as _cpu_np


# ─────────────────────────────────────────────
# METRIC HELPERS
# ─────────────────────────────────────────────

def compute_f1(pred_tokens: List[str], target_tokens: List[str]) -> float:
    if not pred_tokens or not target_tokens:
        return 1.0 if pred_tokens == target_tokens else 0.0
    target_counts = {}
    for t in target_tokens:
        target_counts[t] = target_counts.get(t, 0) + 1
    num_same = 0
    pred_counts = {}
    for t in pred_tokens:
        if t in target_counts and pred_counts.get(t, 0) < target_counts[t]:
            num_same += 1
            pred_counts[t] = pred_counts.get(t, 0) + 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


# ─────────────────────────────────────────────
# VALIDATION — BATCHED INFERENCE
# ─────────────────────────────────────────────

def validate(master_model: CoreBurstBackprop,
             val_samples: List[dict],
             tokenizer: BPETokenizer,
             max_doc: int,
             max_q: int,
             batch_size: int = 8,
             pre_tokenized_dataset = None) -> Dict:
    """
    Run batched inference on validation set and compute metrics.
    
    Processes multiple samples at once for 5-10x speedup over single-sample inference.
    Gracefully skips samples with alignment errors.
    
    Supports:
        - Legacy mode: tokenize on the fly
        - Pre-tokenized mode: index GPU arrays (even faster)
    """
    total_loss = 0.0
    total_em   = 0.0
    total_f1   = 0.0
    num_valid = 0
    
    # Use pre-tokenized data if available
    if pre_tokenized_dataset is not None:
        # Pre-tokenized mode: direct indexing
        all_indices = pre_tokenized_dataset.get_sample_indices()
        n_batches = (len(all_indices) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_indices))
            batch_indices = _cpu_np.array(all_indices[start_idx:end_idx], dtype=_cpu_np.int32)
            
            # Get pre-tokenized batch
            token_ids, positions, segment_mask, start_t, end_t = \
                pre_tokenized_dataset.get_batch(batch_indices)
            
            # Batch forward pass
            p_start, p_end, batch_loss = master_model.forward(
                token_ids=token_ids, positions=positions,
                segment_mask=segment_mask,
                start_targets=start_t, end_targets=end_t,
                training=False
            )
            
            total_loss += float(to_cpu(batch_loss)) * len(batch_indices)
            num_valid += len(batch_indices)
            
            # Compute metrics per sample in batch
            sep_id = tokenizer.token2id.get('<sep>', 2)
            for local_idx in range(len(batch_indices)):
                # Get T_doc by finding the index of sep_id in token_ids
                sep_indices = np.where(token_ids[local_idx] == sep_id)[0]
                if len(sep_indices) > 0:
                    T_doc = int(to_cpu(sep_indices[0])) - 1
                else:
                    T_doc = len(token_ids[local_idx]) - 1

                # Joint Span Search (max length 20)
                max_span_len = 20
                best_score = -1.0
                pred_s, pred_e = 1, 1
                
                ps_cpu = to_cpu(p_start[local_idx])
                pe_cpu = to_cpu(p_end[local_idx])
                
                for s in range(1, T_doc + 1):
                    for e in range(s, min(s + max_span_len, T_doc + 1)):
                        score = float(ps_cpu[s] * pe_cpu[e])
                        if score > best_score:
                            best_score = score
                            pred_s, pred_e = s, e
                
                # For pre-tokenized, we need to get the original sample to decode prediction
                # This is a limitation but acceptable for validation-only path
                global_sample_idx = all_indices[start_idx + local_idx]
                original_sample = val_samples[global_sample_idx]
                
                # Decode prediction from token IDs
                doc_ids = to_cpu(token_ids[local_idx, 1:])  # Skip CLS token
                pred_ids = doc_ids[pred_s - 1: pred_e]
                try:
                    pred_ans = tokenizer.decode(pred_ids.tolist()).strip().lower()
                except:
                    pred_ans = ""
                
                true_ans = original_sample['answer'].strip().lower()
                
                em = 1.0 if pred_ans == true_ans else 0.0
                f1 = compute_f1(pred_ans.split(), true_ans.split())
                
                total_em += em
                total_f1 += f1
    else:
        # Legacy mode: tokenize on the fly
        # Pre-process valid samples (skip those with alignment errors)
        valid_samples = []
        sample_alignments = {}  # sample_idx -> (qa_enc, ans_s, ans_e)
        
        for idx, sample in enumerate(val_samples):
            try:
                qa_enc, ans_s, ans_e = align_answer_to_tokens(
                    sample, tokenizer, max_doc, max_q
                )
                valid_samples.append(sample)
                sample_alignments[len(valid_samples) - 1] = (qa_enc, ans_s, ans_e)
            except AnswerAlignmentError:
                continue  # Skip samples with bad alignment
        
        if len(valid_samples) == 0:
            return {'val_loss': 0.0, 'val_em': 0.0, 'val_f1': 0.0, 'n_skipped': len(val_samples)}

        # Process in batches
        for batch_start in range(0, len(valid_samples), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_samples))
            batch_samples = valid_samples[batch_start:batch_end]
            
            # Prepare batch arrays
            batch_token_ids = []
            batch_positions = []
            batch_segment_mask = []
            batch_start_targets = []
            batch_end_targets = []
            batch_doc_lengths = []
            
            for local_idx, sample in enumerate(batch_samples):
                global_idx = batch_start + local_idx
                qa_enc, ans_s, ans_e = sample_alignments[global_idx]
                
                batch_token_ids.append(qa_enc['token_ids'])
                batch_positions.append(qa_enc['positions'])
                batch_segment_mask.append(qa_enc['segment_mask'])
                batch_start_targets.append(ans_s)
                batch_end_targets.append(ans_e)
                batch_doc_lengths.append(qa_enc['doc_length'])
            
            # Convert to arrays
            token_ids = np.array(batch_token_ids, dtype=_cpu_np.int32)
            positions = np.array(batch_positions, dtype=_cpu_np.int32)
            segment_mask = np.array(batch_segment_mask, dtype=_cpu_np.int32)
            start_t = np.array(batch_start_targets, dtype=_cpu_np.int32)
            end_t = np.array(batch_end_targets, dtype=_cpu_np.int32)
            
            # Batch forward pass
            p_start, p_end, batch_loss = master_model.forward(
                token_ids=token_ids, positions=positions,
                segment_mask=segment_mask,
                start_targets=start_t, end_targets=end_t,
                training=False
            )
            
            total_loss += float(to_cpu(batch_loss)) * len(batch_samples)
            num_valid += len(batch_samples)
            
            # Compute metrics per sample in batch
            for local_idx, sample in enumerate(batch_samples):
                T_doc = batch_doc_lengths[local_idx]
                
                # Joint Span Search (max length 20)
                max_span_len = 20
                best_score = -1.0
                pred_s, pred_e = 1, 1
                
                ps_cpu = to_cpu(p_start[local_idx])
                pe_cpu = to_cpu(p_end[local_idx])
                
                for s in range(1, T_doc + 1):
                    for e in range(s, min(s + max_span_len, T_doc + 1)):
                        score = float(ps_cpu[s] * pe_cpu[e])
                        if score > best_score:
                            best_score = score
                            pred_s, pred_e = s, e
                
                # Decode prediction
                global_idx = batch_start + local_idx
                qa_enc, _, _ = sample_alignments[global_idx]
                doc_ids = qa_enc['token_ids'][1: T_doc + 1]
                pred_ids = doc_ids[pred_s - 1: pred_e]
                pred_ans = tokenizer.decode(pred_ids).strip().lower()
                true_ans = sample['answer'].strip().lower()
                
                em = 1.0 if pred_ans == true_ans else 0.0
                f1 = compute_f1(pred_ans.split(), true_ans.split())
                
                total_em += em
                total_f1 += f1

    if num_valid == 0:
        return {'val_loss': 0.0, 'val_em': 0.0, 'val_f1': 0.0, 'n_skipped': len(val_samples)}

    return {
        'val_loss': total_loss / num_valid,
        'val_em':   (total_em / num_valid) * 100.0,
        'val_f1':   (total_f1 / num_valid) * 100.0,
        'n_skipped': len(val_samples) - num_valid,
    }



# ─────────────────────────────────────────────
# MAIN TRAINING FUNCTION — IMPROVED
# ─────────────────────────────────────────────

def train_pdcl(config: PDCLConfig = None, **kwargs):
    """
    Full PDCL training with all components and safety checks.
    
    Args:
        config : PDCLConfig object (preferred) or None to use kwargs
        **kwargs : Legacy parameter-based config (for backward compatibility)
        
    Epoch-level PDCL operations:
        - Feature dimension pruning (from epoch fp_warmup_epochs)
        - Connection pruning (from epoch conn_prune_from_epoch)
        - Cross-dimension graph recomputation
    
    ──────────────────────────────────────────────────
    IMPROVEMENTS:
        1. Config validation before training (prevents 30-min crashes)
        2. Fuzzy answer alignment with fallback to skip (not middle-of-doc)
        3. Pre-tokenized GPU dataset (eliminates CPU→GPU transfers)
        4. Batched validation (5-10x faster inference)
        5. Centralized hyperparameters
    """

    # ────────────────────────────────────
    # Step 0: Build or validate config
    # ────────────────────────────────────
    if config is None:
        # Legacy: build config from kwargs
        config = PDCLConfig(**kwargs)

    print(f"\n{'='*70}")
    print(f"  PDCL TRAINING — VALIDATING CONFIGURATION")
    print(f"{'='*70}\n")

    is_valid, error_msg = validate_config(config)
    if not is_valid:
        print(f"❌ CONFIGURATION ERROR:\n   {error_msg}")
        raise ValueError(f"Invalid PDCL config: {error_msg}")

    print(f"✓ Configuration validated successfully")
    print(f"  Dimensions: {config.n_dimensions} | d_model: {config.d_model}")
    print(f"  Data: {config.train_subset_size} train, {config.val_subset_size} val")
    print(f"  GPU: {GPU_AVAILABLE}\n")

    # ────────────────────────────────────
    # Step 1: Load tokenizer
    # ────────────────────────────────────
    print("Step 1: Loading tokenizer...")
    try:
        tokenizer = BPETokenizer()
        tokenizer.load(config.tokenizer_path)
        print(f"  ✓ Tokenizer loaded | Vocab: {tokenizer.vocab_size}\n")
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Tokenizer not found at {config.tokenizer_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    # ────────────────────────────────────
    # Step 2: Load data
    # ────────────────────────────────────
    print("Step 2: Loading data...")
    try:
        with open(config.train_data_path) as f:
            train_data = json.load(f)
        with open(config.val_data_path) as f:
            val_data = json.load(f)
        
        train_samples = train_data[:config.train_subset_size]
        val_samples = val_data[:config.val_subset_size]
        print(f"  ✓ Data loaded | Train: {len(train_samples)} | Val: {len(val_samples)}\n")
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

    # ────────────────────────────────────
    # Step 3: Build master model
    # ────────────────────────────────────
    print("Step 3: Building master model...")
    try:
        master = CoreBurstBackprop(
            vocab_size=tokenizer.vocab_size,
            max_positions=config.max_doc + config.max_q + 4,
            d_model=config.d_model,
            n_heads=config.n_heads,
            burst_beta=config.burst_beta,
            k_factor=config.k_factor,
            gate_sharpness=config.gate_sharpness,
            burst_freeze_threshold=getattr(config, 'burst_freeze_threshold', 0.15)
        )
        print(f"  ✓ Master model created\n")
    except Exception as e:
        raise RuntimeError(f"Failed to build master model: {e}")

    # ────────────────────────────────────
    # Step 3.5: Warm up CuPy JIT (if GPU active)
    # ────────────────────────────────────
    if GPU_AVAILABLE:
        print("Step 3.5: Warming up CuPy kernels (compiling JIT sequentially)...")
        try:
            import cupy as cp
            # Create a small dummy batch
            B_dummy = 2
            T_dummy = config.max_doc + config.max_q + 4
            dummy_token_ids = cp.zeros((B_dummy, T_dummy), dtype=cp.int32)
            dummy_positions = cp.zeros((B_dummy, T_dummy), dtype=cp.int32)
            dummy_segment_mask = cp.zeros((B_dummy, T_dummy), dtype=cp.int32)
            # Route some tokens to document (segment 0) and some to question (segment 1)
            dummy_segment_mask[:, T_dummy // 2:] = 1
            dummy_start_targets = cp.zeros((B_dummy,), dtype=cp.int32)
            dummy_end_targets = cp.zeros((B_dummy,), dtype=cp.int32)

            # Set a dummy current_epoch to activate group relation detector if needed
            master.current_epoch = 3

            # Forward pass
            _, _, _ = master.forward(
                token_ids=dummy_token_ids,
                positions=dummy_positions,
                segment_mask=dummy_segment_mask,
                start_targets=dummy_start_targets,
                end_targets=dummy_end_targets,
                training=True
            )

            # Backward pass
            master.burst_backward()
            master.zero_grad()
            master.current_epoch = 0
            print("  ✓ CuPy kernels compiled and cached successfully.\n")
        except Exception as e:
            print(f"  ⚠ CuPy warmup failed/skipped: {e}\n")
            master.zero_grad()
            master.current_epoch = 0

    # ────────────────────────────────────
    # Step 4: Feature Dimension Pruner
    # ────────────────────────────────────
    feature_pruner = FeatureDimensionPruner(
        d_model=config.d_model,
        base_prune_pct=config.fp_base_pct,
        max_prune_pct=config.fp_max_pct,
        total_epochs=config.epochs,
        min_epochs_before_prune=config.fp_warmup_epochs,
        ema_decay=config.fp_ema_decay,
    )

    # ────────────────────────────────────
    # Step 4.5: Pre-tokenize data (GPU acceleration)
    # ────────────────────────────────────
    print("Step 4.5: Pre-tokenizing training data...")
    try:
        pre_tokenized_dataset = PreTokenizedDataset(
            samples=train_samples,
            tokenizer=tokenizer,
            max_doc=config.max_doc,
            max_q=config.max_q,
            gpu_available=GPU_AVAILABLE
        )
        print(f"  ✓ Pre-tokenization complete | GPU pinned: {GPU_AVAILABLE}")
        print(f"    Total samples pre-cached: {pre_tokenized_dataset.n_samples}\n")
    except Exception as e:
        print(f"  ⚠ Pre-tokenization failed, falling back to legacy mode: {e}")
        pre_tokenized_dataset = None

    # ────────────────────────────────────
    # Step 5: Parallel Dimension Engine
    # ────────────────────────────────────
    print("Step 5: Initializing parallel engine...")
    try:
        engine = ParallelDimensionEngine(
            n_dimensions=config.n_dimensions,
            master_model=master,
            dataset=train_samples if pre_tokenized_dataset is None else None,
            tokenizer=tokenizer if pre_tokenized_dataset is None else None,
            max_doc=config.max_doc,
            max_q=config.max_q,
            pre_tokenized_dataset=pre_tokenized_dataset,
        )
        print(f"  ✓ Parallel engine ready ({config.n_dimensions} dimensions)\n")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize parallel engine: {e}")

    # ────────────────────────────────────
    # TRAINING HEADER
    # ────────────────────────────────────
    print(f"{'='*70}")
    print(f"  PDCL TRAINING — TWO-PHASE MATURITY PIPELINE")
    print(f"{'='*70}")
    print(f"  Model Config:")
    print(f"    • Dimensions       : {config.n_dimensions}")
    print(f"    • Epochs           : {config.epochs} × {config.steps_per_epoch} steps")
    print(f"    • d_model          : {config.d_model}")
    print(f"    • n_heads          : {config.n_heads}")
    print(f"    • Batch size       : {config.batch_size} (Phase 1) → 1 (Phase 2)")
    print(f"    • Learning rate    : {config.lr}")
    print(f"\n  Phase 1 — General Maturation:")
    print(f"    • Burst Backprop   : OFF (free gradient flow)")
    print(f"    • Feature Pruning  : OFF (tracking importance only)")
    print(f"    • Connection Prune : OFF")
    print(f"    • Maturity trigger : EM≥70% or F1≥70% or loss↓70% (dynamic, no epoch fallback)")
    print(f"\n  Phase 2 — Specialization & Pruning:")
    print(f"    • Burst Backprop   : ON (blast range freezing, β={config.burst_beta})")
    print(f"    • Feature Pruning  : ON ({config.fp_base_pct}%→{config.fp_max_pct}%)")
    print(f"    • Connection Prune : ON (δ={config.conn_prune_threshold})")
    print(f"    • Batch per dim    : 1 (sharp core burst focus)")
    print(f"\n  Soft Pruning (Adaptive, always active):")
    print(f"    • k_factor         : {config.k_factor}")
    print(f"    • gate_sharpness   : {config.gate_sharpness}")
    print(f"\n  Hardware:")
    print(f"    • GPU              : {GPU_AVAILABLE}")
    print(f"    • Answer matching  : fuzzy (edit distance ≤ 3)")
    print(f"    • Bad samples      : skip (no poisoned targets)")
    print(f"{'='*70}\n")

    # Pre-tokenize validation set if pre-tokenization is enabled
    pre_tokenized_val = None
    if pre_tokenized_dataset is not None:
        print("Pre-tokenizing validation data...")
        try:
            pre_tokenized_val = PreTokenizedDataset(
                samples=val_samples,
                tokenizer=tokenizer,
                max_doc=config.max_doc,
                max_q=config.max_q,
                gpu_available=GPU_AVAILABLE
            )
            print(f"  ✓ Validation pre-tokenization complete\n")
        except Exception as e:
            print(f"  ⚠ Validation pre-tokenization failed: {e}\n")
            pre_tokenized_val = None

    history = []

    # ─────────────────────────────────────
    # EARLY STOPPING & MATURITY SETUP
    # ─────────────────────────────────────
    best_val_loss = float('inf')
    best_val_em = 0.0
    patience_counter = 0
    patience = 15  # Stop if no improvement for 15 epochs
    best_checkpoint_path = config.checkpoint_path.replace('.pkl', '_best.pkl')

    # Two-phase maturity tracking
    maturity_reached = False
    initial_loss = None
    phase2_patience = 0          # counts epochs without val_loss improvement
    phase2_patience_limit = 3    # activate Phase 2 after 3 stagnant epochs
    phase2_best_val_loss = float('inf')

    # Dynamic learning rate tracking
    base_lr = config.lr
    initial_val_loss = None

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()
        master.current_epoch = epoch

        # ─────────────────────────────────────
        # TRAINING STEPS (all dimensions parallel)
        # ─────────────────────────────────────
        epoch_losses = []
        epoch_dim_losses = []

        # Phase 2 uses batch_size=1 per dimension for sharp core burst focus
        effective_batch_size = config.batch_size if getattr(config, 'keep_batch_size_phase2', False) else (1 if maturity_reached else config.batch_size)

        for step in range(config.steps_per_epoch):
            try:
                global_loss, dim_losses = engine.execute_parallel_step(
                    lr=config.lr,
                    clip_norm=config.clip_norm,
                    batch_size=effective_batch_size
                )
                epoch_losses.append(global_loss)
                epoch_dim_losses.append(dim_losses)

                # Update feature pruner importance tracking after each backward
                feature_pruner.update_importance(master)
            except AnswerAlignmentError as e:
                # Log skipped batch but continue training
                print(f"  [Step {step+1}] Skipped batch (alignment error): {e}")
                continue
            except Exception as e:
                print(f"❌ Training step failed: {e}")
                raise

        if len(epoch_losses) == 0:
            print(f"⚠ Epoch {epoch}: All batches skipped (no valid alignments)")
            continue

        avg_train_loss = float(_cpu_np.mean(epoch_losses))

        # Track initial loss for convergence measurement
        if initial_loss is None:
            initial_loss = avg_train_loss

        # ─────────────────────────────────────
        # EPOCH-LEVEL PDCL OPERATIONS
        # ─────────────────────────────────────

        epoch_report = {}

        # 1. Feature Dimension Pruning (only after maturity)
        if maturity_reached:
            sparsity_stats = feature_pruner.apply_masks(epoch, master)
            if sparsity_stats:
                avg_sparsity = float(_cpu_np.mean(list(sparsity_stats.values())))
                epoch_report['feature_sparsity'] = round(avg_sparsity, 4)

            # Apply gradient masks to prevent pruned features from reviving
            feature_pruner.apply_gradient_masks(master)

        # 2. Connection Pruning (conservative — only truly dead connections)
        if maturity_reached:
            if not hasattr(engine, '_conn_prune_start_epoch'):
                engine._conn_prune_start_epoch = epoch
            conn_epochs_active = epoch - engine._conn_prune_start_epoch

            # Only prune every 5 epochs to give the model time to stabilize
            if conn_epochs_active > 0 and conn_epochs_active % 5 == 0:
                # Use a tiny fixed threshold — only prune connections that are
                # essentially dead (near-zero attention weight across all training)
                effective_conn_threshold = 1e-6

                conn_stats = master.attention.update_connection_masks(
                    threshold=effective_conn_threshold
                )
                total_pruned = sum(conn_stats.values())
                epoch_report['conn_pruned'] = total_pruned
                epoch_report['conn_threshold'] = effective_conn_threshold
            else:
                epoch_report['conn_pruned'] = 0
                epoch_report['conn_threshold'] = 0.0

        # 3. Cross-Dimension Graph Update
        graph_summary = engine.update_graph(epoch)
        epoch_report['graph_edges'] = graph_summary.get('n_edges', 0)
        epoch_report['graph_max_corr'] = round(graph_summary.get('max_correlation', 0.0), 3)

        # 4. Effective keep rate of adaptive soft pruning
        keep_rate = master.prune.get_effective_keep_rate()
        epoch_report['effective_keep_rate'] = round(keep_rate, 3)

        # ─────────────────────────────────────
        # VALIDATION (batched, with GPU pre-tokenization)
        # ─────────────────────────────────────
        try:
            val_metrics = validate(
                master, val_samples, tokenizer,
                config.max_doc, config.max_q,
                batch_size=config.batch_size,
                pre_tokenized_dataset=pre_tokenized_val
            )
        except Exception as e:
            print(f"⚠ Validation failed: {e}")
            val_metrics = {
                'val_loss': 0.0, 'val_em': 0.0,
                'val_f1': 0.0, 'n_skipped': len(val_samples)
            }

        elapsed = time.time() - t0

        # Per-dim loss breakdown
        avg_dim = _cpu_np.mean(epoch_dim_losses, axis=0).tolist()
        dim_str = " | ".join([f"D{i}:{v:.3f}" for i, v in enumerate(avg_dim)])

        print(f"Epoch {epoch:2d}/{config.epochs} | "
              f"Train: {avg_train_loss:.4f} | "
              f"Val Loss: {val_metrics['val_loss']:.4f} | "
              f"EM: {val_metrics['val_em']:.1f}% | "
              f"F1: {val_metrics['val_f1']:.1f}% | "
              f"Time: {elapsed:.1f}s")
        print(f"  Dims: {dim_str}")
        if epoch_report:
            rep_str = " | ".join([f"{k}: {v}" for k, v in epoch_report.items()])
            print(f"  PDCL: {rep_str}")
        if val_metrics.get('n_skipped', 0) > 0:
            print(f"  ⚠ Val: skipped {val_metrics['n_skipped']} samples (alignment)")

        # ─────────────────────────────────────
        # DYNAMIC LEARNING RATE ADJUSTMENT (Concept A)
        # ─────────────────────────────────────
        if getattr(config, 'dynamic_lr_loss_ratio', False):
            if initial_val_loss is None and val_metrics['val_loss'] > 1e-5:
                initial_val_loss = val_metrics['val_loss']
                
            if initial_val_loss and initial_val_loss > 1e-5:
                ratio = val_metrics['val_loss'] / initial_val_loss
                ratio = max(0.05, min(ratio, 1.5))
                config.lr = base_lr * (ratio ** 2)
                config.lr = max(1e-5, config.lr)
                if maturity_reached:
                    max_phase2_lr = getattr(config, 'max_phase2_lr', 0.003)
                    config.lr = min(config.lr, max_phase2_lr)
                print(f"  ⚡ Dynamic LR Adjusted: {config.lr:.6f} (Val Loss Ratio: {ratio:.3f})")

        # ─────────────────────────────────────
        # DYNAMIC MATURITY DETECTION
        # ─────────────────────────────────────
        if not maturity_reached:
            loss_reduction_pct = 0.0
            if initial_loss and initial_loss > 1e-5:
                loss_reduction_pct = (initial_loss - avg_train_loss) / initial_loss

            is_mature = (
                val_metrics['val_em'] >= 70.0 or
                val_metrics['val_f1'] >= 70.0 or
                loss_reduction_pct >= 0.70
            )
            if is_mature:
                maturity_reached = True
                master.maturity_reached = True
                # Propagate to all dimension clones
                for clone in engine.dim_models:
                    clone.maturity_reached = True
                print(f"  🎯 PHASE 2 ACTIVATED — Model matured at epoch {epoch}")
                print(f"     Loss reduction: {loss_reduction_pct*100:.1f}% | "
                      f"EM: {val_metrics['val_em']:.1f}% | F1: {val_metrics['val_f1']:.1f}%")
                print(f"     → Core Burst Freezing: ON")
                print(f"     → Feature/Connection Pruning: ON")
                print(f"     → Batch size per dimension: 1 (sharp focus)")

        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': val_metrics['val_loss'],
            'val_em': val_metrics['val_em'],
            'val_f1': val_metrics['val_f1'],
            'time': elapsed,
            **epoch_report
        })

        # ─────────────────────────────────────
        # EARLY STOPPING CHECK
        # ─────────────────────────────────────
        improved = False
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_val_em = val_metrics['val_em']
            patience_counter = 0
            improved = True
            
            # Save best checkpoint
            try:
                master.save_checkpoint(best_checkpoint_path)
                print(f"  🏆 New best val_loss: {best_val_loss:.4f} | EM: {best_val_em:.1f}%")
            except Exception as e:
                print(f"  ⚠ Failed to save best checkpoint: {e}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹ Early stopping triggered!")
                print(f"   Best val_loss: {best_val_loss:.4f} (Epoch {epoch - patience_counter})")
                print(f"   No improvement for {patience} epochs. Stopping training.")
                break

    # ── Final summary ──
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    
    # Load best checkpoint if it exists
    if os.path.exists(best_checkpoint_path):
        try:
            master.load_checkpoint(best_checkpoint_path)
            print(f"✓ Loaded best checkpoint (val_loss: {best_val_loss:.4f})")
        except Exception as e:
            print(f"⚠ Failed to load best checkpoint: {e}")
    
    if history:
        print(f"Final Val EM : {history[-1]['val_em']:.1f}%")
        print(f"Final Val F1 : {history[-1]['val_f1']:.1f}%")

    sparsity_report = feature_pruner.get_sparsity_report()
    if sparsity_report:
        avg_sp = float(_cpu_np.mean(list(sparsity_report.values())))
        print(f"Avg Feature Sparsity: {avg_sp*100:.1f}%")

    # Save final checkpoint
    try:
        master.save_checkpoint(config.checkpoint_path)
        print(f"✓ Final checkpoint saved to {config.checkpoint_path}")
    except Exception as e:
        print(f"⚠ Failed to save final checkpoint: {e}")

    # Show example predictions
    print(f"\nExample Predictions:")
    print("-" * 70)
    example_count = 0
    for sample in val_samples:
        if example_count >= 5:
            break
        try:
            qa_enc, ans_s, ans_e = align_answer_to_tokens(
                sample, tokenizer, config.max_doc, config.max_q
            )
            token_ids = np.array([qa_enc['token_ids']], dtype=_cpu_np.int32)
            positions = np.array([qa_enc['positions']], dtype=_cpu_np.int32)
            segment_mask = np.array([qa_enc['segment_mask']], dtype=_cpu_np.int32)

            p_start, p_end, _ = master.forward(
                token_ids=token_ids, positions=positions,
                segment_mask=segment_mask,
                start_targets=np.array([ans_s], dtype=_cpu_np.int32),
                end_targets=np.array([ans_e], dtype=_cpu_np.int32),
                training=False
            )

            T_doc = qa_enc['doc_length']
            
            # Joint Span Search (max length 20)
            max_span_len = 20
            best_score = -1.0
            pred_s, pred_e = 1, 1
            
            ps_cpu = to_cpu(p_start[0])
            pe_cpu = to_cpu(p_end[0])
            
            for s in range(1, T_doc + 1):
                for e in range(s, min(s + max_span_len, T_doc + 1)):
                    score = float(ps_cpu[s] * pe_cpu[e])
                    if score > best_score:
                        best_score = score
                        pred_s, pred_e = s, e

            doc_ids = qa_enc['token_ids'][1: T_doc + 1]
            pred_ids = doc_ids[pred_s - 1: pred_e]
            pred_ans = tokenizer.decode(pred_ids).strip()

            gate_rate = master.prune.get_effective_keep_rate()
            print(f"  Q: {sample['question']}")
            print(f"  Target : {sample['answer']}")
            print(f"  Predict: {pred_ans}")
            print(f"  Noise  : {sample.get('noise_level', 'N/A')} | "
                  f"Gate keep≈{gate_rate:.0%}")
            print("-" * 70)
            example_count += 1
        except AnswerAlignmentError as ex:
            print(f"  [skipped: {ex}]")
        except Exception as ex:
            print(f"  [error: {ex}]")

    return history


if __name__ == '__main__':
    # Example 1: Using PDCLConfig (recommended)
    config = PDCLConfig(
        n_dimensions=16,  # Increased from 4 — graph more informative
        d_model=128,
        n_heads=4,
        max_doc=256,
        max_q=32,
        epochs=10,
        steps_per_epoch=10,
        batch_size=4,
        lr=0.005,
        burst_beta=2.0,
        k_factor=0.5,
        gate_sharpness=3.0,
        fp_base_pct=10.0,
        fp_max_pct=35.0,
        fp_warmup_epochs=3,
        conn_prune_threshold=0.02,
        conn_prune_from_epoch=3,
        train_subset_size=200,
        val_subset_size=40,
    )

    # Run training
    history = train_pdcl(config)

    print(f"\n✓ Training finished. History saved with {len(history)} epochs.")
    train_pdcl(
        n_dimensions      = 4,
        train_subset_size = 200,
        val_subset_size   = 40,
        epochs            = 10,
        steps_per_epoch   = 10,
        batch_size        = 4,
        lr                = 0.005,
        d_model           = 128,
        n_heads           = 2,
        burst_beta        = 2.0,
        k_factor          = 0.5,
        gate_sharpness    = 3.0,
        max_doc           = 512,
        max_q             = 64,
        fp_base_pct       = 10.0,
        fp_max_pct        = 35.0,
        fp_warmup_epochs  = 3,
        conn_prune_threshold=0.02,
        conn_prune_from_epoch=3,
    )
