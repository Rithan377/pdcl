# PDCL Quick Start — Using New Improvements

## Run Training with Validation

### Recommended Approach (Config-based)

```python
from pdcl_train import train_pdcl
from pdcl_utils import PDCLConfig

# Create config
config = PDCLConfig(
    # Model architecture
    n_dimensions=16,          # Increased from 4 for better graph learning
    d_model=128,
    n_heads=4,
    max_doc=256,              # Allow longer documents
    max_q=32,
    
    # Training schedule
    epochs=15,
    steps_per_epoch=20,
    batch_size=4,
    lr=0.005,
    
    # Burst backprop
    burst_beta=2.0,
    
    # Soft pruning (adaptive)
    k_factor=0.5,
    gate_sharpness=3.0,
    
    # Feature pruning
    fp_base_pct=10.0,
    fp_max_pct=35.0,
    fp_warmup_epochs=3,
    
    # Connection pruning
    conn_prune_threshold=0.02,
    conn_prune_from_epoch=3,
    
    # Data
    train_subset_size=500,    # More training data
    val_subset_size=100,
)

# Validate before running (catches errors early)
from pdcl_utils import validate_config
is_valid, error = validate_config(config)
if not is_valid:
    print(f"Config error: {error}")
    exit(1)

# Run training
history = train_pdcl(config)

# Save config for reproducibility
config.save('./experiments/config_v1.json')
```

### Legacy Approach (Still Works)

```python
from pdcl_train import train_pdcl

history = train_pdcl(
    n_dimensions=16,
    epochs=15,
    d_model=128,
    batch_size=4,
    # ... other parameters
)
```

---

## Key Improvements You're Using

### 1. Fuzzy Answer Alignment
Automatically handles noisy data:
- "5,000,000" in metadata matches "5,0O0,000" in OCR document
- Edit distance tolerance = 3 (covers most OCR errors)
- Skips samples that can't be reliably aligned

### 2. Config Validation
All these checked at startup (prevents 30-min crashes):
- d_model divisible by n_heads
- Files exist and are valid JSON
- All dimensions >= 1
- Learning rate > 0
- Subset sizes don't exceed available data
- ... 10+ more checks

### 3. Batched Validation
Processes 4-8 validation samples at once instead of 1.
**Result**: 5-10x faster validation without losing precision.

### 4. Better Error Handling
- Clear error messages if tokenizer/data not found
- Handles bad alignments gracefully (skips samples)
- Logs which batches had issues
- Continues training instead of crashing

### 5. Centralized Config
All hyperparameters in one object:
```python
config.to_dict()  # Serialize to dict/JSON
config.save(path)  # Save for reproducibility
config = PDCLConfig.load(path)  # Load later
```

---

## Monitoring Training

The training loop prints detailed information:

```
======================================================================
  PDCL TRAINING — VALIDATING CONFIGURATION
======================================================================

✓ Configuration validated successfully
  Dimensions: 16 | d_model: 128
  Data: 500 train, 100 val
  GPU: True

Step 1: Loading tokenizer...
  ✓ Tokenizer loaded | Vocab: 10000

Step 2: Loading data...
  ✓ Data loaded | Train: 500 | Val: 100

Step 3: Building master model...
  ✓ Master model created

Step 5: Initializing parallel engine...
  ✓ Parallel engine ready (16 dimensions)

======================================================================
  PDCL TRAINING — ALL COMPONENTS ACTIVE
======================================================================
  Model Config:
    • Dimensions       : 16
    • Epochs           : 15 × 20 steps
    • d_model          : 128
    • n_heads          : 4
    • Batch size       : 4
    • Learning rate    : 0.005

  ...other config...

  Hardware:
    • GPU              : True
    • Answer matching  : fuzzy (edit distance ≤ 3)
    • Bad samples      : skip (no poisoned targets)
======================================================================

Epoch  1/15 | Train: 2.3421 | Val Loss: 2.1245 | EM: 15.0% | F1: 28.3% | Time: 45.2s
  Dims: D0:2.34 | D1:2.28 | D2:2.31 | D3:2.25 | D4:2.32 | D5:2.29 | ...
  PDCL: feature_sparsity: 0.1234 | graph_edges: 45 | graph_max_corr: 0.782 | effective_keep_rate: 0.654
  ⚠ Val: skipped 2 samples (alignment)

Epoch  2/15 | Train: 2.1432 | Val Loss: 2.0123 | EM: 18.0% | F1: 31.2% | Time: 44.8s
  ...
```

---

## Interpreting Output

### Metrics:
- **Train Loss**: Cross-entropy on training batch
- **Val Loss**: Cross-entropy on validation set
- **EM**: Exact match percentage on full answer span
- **F1**: Word-level F1 score (handles partial matches)
- **Time**: Epoch runtime in seconds

### PDCL-specific:
- **feature_sparsity**: % of features permanently pruned
- **graph_edges**: Non-zero entries in correlation graph
- **graph_max_corr**: Strongest correlation between any two dimensions
- **effective_keep_rate**: % of tokens kept by adaptive pruning
- **n_skipped**: # of validation samples that couldn't be aligned

### Dimension breakdown:
- `D0:2.34 | D1:2.28 | ...`: Loss per processing dimension
- Shows if one dimension is struggling while others converge

---

## Troubleshooting

### Error: "d_model (128) must be divisible by n_heads (5)"
Fix: Use n_heads that divides d_model evenly (e.g., 4, 8)
```python
config = PDCLConfig(d_model=128, n_heads=4)  # ✓ 128 / 4 = 32
```

### Error: "Train data not found at ./data/train.json"
Fix: Ensure files exist and config paths are correct
```python
config = PDCLConfig(
    train_data_path='./data/train.json',  # Verify this path exists
    val_data_path='./data/val.json',
    tokenizer_path='./tokenizer',
)
```

### Error: "⚠ Val: skipped X samples (alignment)"
**This is normal**: Some validation samples may not align perfectly.
- If very high (>30%), check data quality
- Model still trains on aligned samples
- Non-critical warning, not an error

### Training is slow
- Enable GPU: Verify `GPU: True` in header
- Increase batch_size (if VRAM allows)
- Next optimization: Pre-tokenization to GPU (see IMPROVEMENTS.md)

---

## Experiment Tracking

```python
# Run with modified config
config = PDCLConfig(n_dimensions=16, epochs=20, ...)
history = train_pdcl(config)

# Save for reproducibility
config.save('./experiments/config_nd16_ep20.json')

# Later: reload exact config
config_loaded = PDCLConfig.load('./experiments/config_nd16_ep20.json')
```

Each experiment now has:
- Reproducible hyperparameters (config)
- Training history (returned as list of dicts)
- Clear data alignment (fuzzy matching traces)
- Validation metrics (including skipped samples)

---

## Next Steps

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for:
1. Pre-tokenization to GPU (highest priority)
2. Increasing n_dimensions to 32
3. Early stopping
4. Learning rate scheduling

