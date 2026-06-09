"""
PDCL Utilities — Fuzzy Matching, Config, and Validation
========================================================
Shared utilities for alignment, configuration, and data validation.
"""

import os
import json
from typing import Tuple, Dict, List, Optional


# ─────────────────────────────────────────────
# FUZZY STRING MATCHING
# ─────────────────────────────────────────────

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.
    
    This measures how many single-character edits (insert, delete, replace)
    are needed to transform s1 into s2. Tolerates OCR artifacts and number
    format variations (e.g., "5,000,000" vs "5,0O0,000").
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def fuzzy_find_answer(answer_text: str, 
                      doc_text: str, 
                      max_distance: int = 3) -> int:
    """
    Find approximate position of answer text within document, tolerating noise.
    
    Args:
        answer_text: The answer string we're looking for (e.g., "5,000,000")
        doc_text: The document text to search in
        max_distance: Maximum edit distance to tolerate (Levenshtein)
        
    Returns:
        Character position in doc_text of best fuzzy match, or -1 if no good match found.
        
    Algorithm:
        Slide a window of answer length through document.
        For each window position, compute edit distance.
        Return position with minimum distance (if distance <= max_distance).
    """
    answer_len = len(answer_text)
    if answer_len == 0 or len(doc_text) < answer_len:
        return -1

    best_distance = max_distance + 1
    best_position = -1

    # Slide window through document
    for i in range(len(doc_text) - answer_len + 1):
        window = doc_text[i:i + answer_len]
        distance = levenshtein_distance(answer_text, window)

        if distance < best_distance:
            best_distance = distance
            best_position = i

        # Early exit if perfect match found
        if distance == 0:
            break

    # Return position only if distance is within tolerance
    return best_position if best_distance <= max_distance else -1


def find_answer_in_doc_tokens(answer_ids: List[int],
                               doc_ids: List[int],
                               tokenizer) -> Tuple[int, int]:
    """
    Find token-level position of answer within document tokens.
    Uses token-level matching first, then falls back to accurate
    character-to-token mapping via per-token decoding.
    
    Args:
        answer_ids: Token IDs of the answer
        doc_ids: Token IDs of the document
        tokenizer: BPE tokenizer (has decode() method)
        
    Returns:
        (ans_start_doc, ans_end_doc) token indices, or (-1, -1) if not found
    """
    if len(answer_ids) == 0 or len(doc_ids) < len(answer_ids):
        return (-1, -1)

    ans_len = len(answer_ids)
    
    # Strategy 1: Try to find exact token sequence match
    for i in range(len(doc_ids) - ans_len + 1):
        if doc_ids[i:i + ans_len] == answer_ids:
            return (i, i + ans_len - 1)

    # Strategy 2: Build accurate char→token map by decoding each token
    # individually, then find which tokens cover the answer text.
    token_texts = []
    token_char_starts = []
    running_len = 0
    for tid in doc_ids:
        text = tokenizer.decode([tid])
        token_char_starts.append(running_len)
        token_texts.append(text)
        running_len += len(text)

    # Reconstruct full document text from tokens
    reconstructed = "".join(token_texts).lower()
    
    # Try multiple answer text variants
    answer_text_decoded = tokenizer.decode(answer_ids).strip().lower()
    # Also try the raw answer (handles cases where tokenizer decode adds/removes chars)
    raw_answer = ""
    for tid in answer_ids:
        if tid != tokenizer.token2id.get('<pad>', 0):
            raw_answer += tokenizer.decode([tid])
    raw_answer = raw_answer.strip().lower()
    
    answer_variants = [answer_text_decoded]
    if raw_answer != answer_text_decoded:
        answer_variants.append(raw_answer)

    char_pos = -1
    used_answer = answer_variants[0]
    
    for ans_text in answer_variants:
        if not ans_text:
            continue
        # Find ALL occurrences and prefer word-boundary matches
        positions = []
        search_start = 0
        while True:
            pos = reconstructed.find(ans_text, search_start)
            if pos < 0:
                break
            positions.append(pos)
            search_start = pos + 1
        
        if positions:
            # Prefer positions at word boundaries (preceded by space, punct, or start)
            boundary_chars = set(' .,;:!?()[]{}"\'-/$')
            best_pos = positions[0]
            for p in positions:
                if p == 0 or reconstructed[p-1] in boundary_chars:
                    best_pos = p
                    break
            char_pos = best_pos
            used_answer = ans_text
            break
    
    # Fuzzy fallback if exact substring not found
    if char_pos < 0:
        char_pos = fuzzy_find_answer(answer_variants[0], reconstructed, max_distance=3)
        used_answer = answer_variants[0]

    if char_pos >= 0:
        ans_end_char = char_pos + len(used_answer)
        # Find which tokens cover [char_pos, ans_end_char)
        start_tok = None
        end_tok = None
        for i, cs in enumerate(token_char_starts):
            next_cs = token_char_starts[i + 1] if i + 1 < len(token_char_starts) else running_len
            # Token i covers characters [cs, next_cs)
            if start_tok is None and next_cs > char_pos:
                start_tok = i
            if cs < ans_end_char:
                end_tok = i
        if start_tok is not None and end_tok is not None:
            return (start_tok, end_tok)

    return (-1, -1)


# ─────────────────────────────────────────────
# PDCL CONFIGURATION
# ─────────────────────────────────────────────

class PDCLConfig:
    """
    Configuration object for PDCL training.
    Centralizes all hyperparameters and settings.
    """

    def __init__(self,
                 # Model architecture
                 n_dimensions: int = 16,
                 d_model: int = 128,
                 n_heads: int = 4,
                 max_doc: int = 256,
                 max_q: int = 32,

                 # Training
                 epochs: int = 10,
                 steps_per_epoch: int = 10,
                 batch_size: int = 4,
                 lr: float = 0.005,
                 clip_norm: float = 1.0,

                 # Burst backprop
                 burst_beta: float = 2.0,

                 # Adaptive soft pruning
                 k_factor: float = 0.5,
                 gate_sharpness: float = 3.0,

                 # Feature pruning
                 fp_base_pct: float = 10.0,
                 fp_max_pct: float = 35.0,
                 fp_warmup_epochs: int = 3,
                 fp_ema_decay: float = 0.95,

                 # Connection pruning
                 conn_prune_threshold: float = 0.02,
                 conn_prune_from_epoch: int = 3,
                 conn_ema_decay: float = 0.95,

                 # Graph formation
                 graph_correlation_lambda: float = 0.5,
                 graph_ema_decay: float = 0.9,
                 graph_min_epochs: int = 2,

                 # Data
                 train_subset_size: int = 200,
                 val_subset_size: int = 40,
                 train_data_path: str = './data/train.json',
                 val_data_path: str = './data/val.json',
                 tokenizer_path: str = './tokenizer',

                 # Checkpointing
                 checkpoint_path: str = './pdcl_checkpoint.npz',
                 save_interval: int = 1,

                 # Fuzzy matching
                 answer_fuzzy_max_distance: int = 3,
                 skip_bad_alignments: bool = True,
                 dynamic_lr_loss_ratio: bool = False,
                 burst_freeze_threshold: float = 0.15,
                 keep_batch_size_phase2: bool = False,
                 max_phase2_lr: float = 0.003):
        """Initialize PDCL configuration."""
        
        self.n_dimensions = n_dimensions
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_doc = max_doc
        self.max_q = max_q
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.clip_norm = clip_norm
        self.burst_beta = burst_beta
        self.k_factor = k_factor
        self.gate_sharpness = gate_sharpness
        self.fp_base_pct = fp_base_pct
        self.fp_max_pct = fp_max_pct
        self.fp_warmup_epochs = fp_warmup_epochs
        self.fp_ema_decay = fp_ema_decay
        self.conn_prune_threshold = conn_prune_threshold
        self.conn_prune_from_epoch = conn_prune_from_epoch
        self.conn_ema_decay = conn_ema_decay
        self.graph_correlation_lambda = graph_correlation_lambda
        self.graph_ema_decay = graph_ema_decay
        self.graph_min_epochs = graph_min_epochs
        self.train_subset_size = train_subset_size
        self.val_subset_size = val_subset_size
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.tokenizer_path = tokenizer_path
        self.checkpoint_path = checkpoint_path
        self.save_interval = save_interval
        self.answer_fuzzy_max_distance = answer_fuzzy_max_distance
        self.skip_bad_alignments = skip_bad_alignments
        self.dynamic_lr_loss_ratio = dynamic_lr_loss_ratio
        self.burst_freeze_threshold = burst_freeze_threshold
        self.keep_batch_size_phase2 = keep_batch_size_phase2
        self.max_phase2_lr = max_phase2_lr

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return self.__dict__.copy()

    def save(self, path: str) -> None:
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str) -> 'PDCLConfig':
        """Load config from JSON file."""
        with open(path) as f:
            cfg_dict = json.load(f)
        return PDCLConfig(**cfg_dict)


# ─────────────────────────────────────────────
# VALIDATION & SAFETY CHECKS
# ─────────────────────────────────────────────

def validate_config(config: PDCLConfig) -> Tuple[bool, str]:
    """
    Validate PDCL configuration before training starts.
    Catches preventable crashes early.
    
    Returns:
        (is_valid, error_message)
        If is_valid=True, message is empty.
        If is_valid=False, message explains the problem.
    """

    # 1. d_model must be divisible by n_heads
    if config.d_model % config.n_heads != 0:
        return False, (
            f"d_model ({config.d_model}) must be divisible by n_heads ({config.n_heads}). "
            f"Got remainder {config.d_model % config.n_heads}."
        )

    # 2. n_heads must be >= 1
    if config.n_heads < 1:
        return False, f"n_heads must be >= 1, got {config.n_heads}"

    # 3. n_dimensions must be >= 1
    if config.n_dimensions < 1:
        return False, f"n_dimensions must be >= 1, got {config.n_dimensions}"

    # 4. max_doc must be reasonable (at least 16 tokens)
    if config.max_doc < 16:
        return False, f"max_doc must be >= 16, got {config.max_doc}"

    # 5. max_q must be reasonable (at least 2 tokens)
    if config.max_q < 2:
        return False, f"max_q must be >= 2, got {config.max_q}"

    # 6. Batch size must be >= 1
    if config.batch_size < 1:
        return False, f"batch_size must be >= 1, got {config.batch_size}"

    # 7. Learning rate must be positive
    if config.lr <= 0:
        return False, f"lr must be > 0, got {config.lr}"

    # 8. Epochs must be >= 1
    if config.epochs < 1:
        return False, f"epochs must be >= 1, got {config.epochs}"

    # 9. Warmup epochs must be < total epochs
    if config.fp_warmup_epochs >= config.epochs:
        return False, (
            f"fp_warmup_epochs ({config.fp_warmup_epochs}) must be < "
            f"total epochs ({config.epochs})"
        )

    # 10. Connection pruning epoch must be < total epochs
    if config.conn_prune_from_epoch >= config.epochs:
        return False, (
            f"conn_prune_from_epoch ({config.conn_prune_from_epoch}) must be < "
            f"total epochs ({config.epochs})"
        )

    # 11. Files must exist
    if not os.path.exists(config.tokenizer_path):
        return False, f"Tokenizer not found at {config.tokenizer_path}"

    if not os.path.exists(config.train_data_path):
        return False, f"Train data not found at {config.train_data_path}"

    if not os.path.exists(config.val_data_path):
        return False, f"Val data not found at {config.val_data_path}"

    # 12. Data paths must be readable JSON
    try:
        with open(config.train_data_path) as f:
            train_data = json.load(f)
        if not isinstance(train_data, list) or len(train_data) == 0:
            return False, f"Train data must be a non-empty JSON list"
    except Exception as e:
        return False, f"Failed to read train data: {e}"

    try:
        with open(config.val_data_path) as f:
            val_data = json.load(f)
        if not isinstance(val_data, list) or len(val_data) == 0:
            return False, f"Val data must be a non-empty JSON list"
    except Exception as e:
        return False, f"Failed to read val data: {e}"

    # 13. Subset sizes must be valid
    if config.train_subset_size > len(train_data):
        return False, (
            f"train_subset_size ({config.train_subset_size}) exceeds available "
            f"train samples ({len(train_data)})"
        )

    if config.val_subset_size > len(val_data):
        return False, (
            f"val_subset_size ({config.val_subset_size}) exceeds available "
            f"val samples ({len(val_data)})"
        )

    # 14. Data format sanity check
    required_fields = ['document', 'question', 'answer']
    for field in required_fields:
        if not all(field in sample for sample in train_data[:10]):
            return False, f"Train samples missing required field: {field}"
        if not all(field in sample for sample in val_data[:10]):
            return False, f"Val samples missing required field: {field}"

    # 15. Pruning percentages must be valid
    if not (0 <= config.fp_base_pct <= 100):
        return False, f"fp_base_pct must be in [0, 100], got {config.fp_base_pct}"

    if not (0 <= config.fp_max_pct <= 100):
        return False, f"fp_max_pct must be in [0, 100], got {config.fp_max_pct}"

    if config.fp_base_pct > config.fp_max_pct:
        return False, (
            f"fp_base_pct ({config.fp_base_pct}) must be <= "
            f"fp_max_pct ({config.fp_max_pct})"
        )

    return True, ""


# ─────────────────────────────────────────────
# PRE-TOKENIZED DATASET (GPU OPTIMIZATION)
# ─────────────────────────────────────────────

class PreTokenizedDataset:
    """
    Pre-computes and caches all tokenization on GPU before training.
    
    MOTIVATION:
        Current flow (SLOW):
            Every step: tokenize CPU → transfer to GPU → compute → transfer to CPU
            
        New flow (FAST):
            Once: tokenize all samples → pin to GPU
            Every step: pure GPU indexing, NO transfers
    
    SAVINGS:
        - With 1000 samples × 600 tokens: ~7MB VRAM (negligible)
        - Eliminates millions of small CPU→GPU transfers
        - Typical speedup: 10-20% on short epochs, 30-50% on long epochs
    
    USAGE:
        # Before training
        dataset = PreTokenizedDataset(
            samples=train_samples,
            tokenizer=tokenizer,
            max_doc=256,
            max_q=32,
            gpu_available=True
        )
        
        # During training: index into GPU arrays (no tokenization)
        batch_ids, batch_pos, batch_masks, batch_starts, batch_ends = \
            dataset.get_batch(indices=[0, 1, 2, 3])
    """

    def __init__(self,
                 samples: List[dict],
                 tokenizer,
                 max_doc: int = 256,
                 max_q: int = 32,
                 gpu_available: bool = False):
        """
        Pre-tokenize all samples and optionally pin to GPU.
        
        Args:
            samples: List of sample dicts with 'document', 'question', 'answer'
            tokenizer: BPE tokenizer with encode_qa() method
            max_doc: Max document tokens
            max_q: Max question tokens
            gpu_available: Whether to use GPU (CuPy) or CPU (NumPy)
        """
        from pdcl_backend import xp as np, to_device
        from pdcl_dimension_engine import align_answer_to_tokens, AnswerAlignmentError

        self.tokenizer = tokenizer
        self.max_doc = max_doc
        self.max_q = max_q
        self.gpu_available = gpu_available
        self.n_samples = len(samples)

        print(f"\n{'='*70}")
        print(f"  PRE-TOKENIZATION TO GPU")
        print(f"{'='*70}")
        print(f"  Samples  : {self.n_samples}")
        print(f"  Device   : {'GPU (CuPy)' if gpu_available else 'CPU (NumPy)'}")

        # Pre-compute all tokenizations
        all_token_ids = []
        all_positions = []
        all_segment_masks = []
        all_starts = []
        all_ends = []
        n_valid = 0

        for idx, sample in enumerate(samples):
            try:
                qa_enc, ans_s, ans_e = align_answer_to_tokens(
                    sample, tokenizer, max_doc, max_q
                )
                all_token_ids.append(qa_enc['token_ids'])
                all_positions.append(qa_enc['positions'])
                all_segment_masks.append(qa_enc['segment_mask'])
                all_starts.append(ans_s)
                all_ends.append(ans_e)
                n_valid += 1
            except AnswerAlignmentError:
                # Skip samples with bad alignment
                continue

        print(f"  Valid    : {n_valid}/{self.n_samples} samples aligned")
        print(f"  Skipped  : {self.n_samples - n_valid} (bad alignment)")

        # Find max length for padding
        max_len = max(len(x) for x in all_token_ids) if all_token_ids else 0
        pad_id = tokenizer.token2id.get('<pad>', 0)

        def pad_array(arrays, pad_val=0):
            """Pad variable-length arrays to max_len."""
            padded = []
            for arr in arrays:
                if len(arr) < max_len:
                    padded.append(arr + [pad_val] * (max_len - len(arr)))
                else:
                    padded.append(arr[:max_len])
            return padded

        # Pad all sequences to same length
        all_token_ids = pad_array(all_token_ids, pad_id)
        all_positions = pad_array(all_positions, 0)
        all_segment_masks = pad_array(all_segment_masks, 0)

        # Convert to numpy arrays
        import numpy as _np
        token_ids_np = _np.array(all_token_ids, dtype=_np.int32)
        positions_np = _np.array(all_positions, dtype=_np.int32)
        masks_np = _np.array(all_segment_masks, dtype=_np.int32)
        starts_np = _np.array(all_starts, dtype=_np.int32)
        ends_np = _np.array(all_ends, dtype=_np.int32)

        # Move to GPU if available (stays on GPU for entire training)
        if gpu_available:
            self.token_ids = to_device(token_ids_np)
            self.positions = to_device(positions_np)
            self.segment_masks = to_device(masks_np)
            self.starts = to_device(starts_np)
            self.ends = to_device(ends_np)
            print(f"  VRAM     : ~{n_valid * max_len * 4 * 3 / (1024**2):.1f} MB")
        else:
            self.token_ids = token_ids_np
            self.positions = positions_np
            self.segment_masks = masks_np
            self.starts = starts_np
            self.ends = ends_np
            print(f"  RAM      : ~{n_valid * max_len * 4 * 3 / (1024**2):.1f} MB")

        self.valid_samples = n_valid
        print(f"{'='*70}\n")

    def get_batch(self, indices: List[int]):
        """
        Get batch of pre-tokenized samples (no tokenization happens here).
        
        Args:
            indices: Sample indices to retrieve
            
        Returns:
            (token_ids, positions, segment_masks, starts, ends) — all GPU arrays
        """
        # Pure indexing — no CPU→GPU transfer, already on GPU
        batch_token_ids = self.token_ids[indices]
        batch_positions = self.positions[indices]
        batch_masks = self.segment_masks[indices]
        batch_starts = self.starts[indices]
        batch_ends = self.ends[indices]

        return batch_token_ids, batch_positions, batch_masks, batch_starts, batch_ends

    def get_sample_indices(self):
        """Return all valid sample indices (for cycling through data)."""
        return list(range(self.valid_samples))
