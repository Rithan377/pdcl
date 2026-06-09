"""
GPU-Optimized Training Script for PDCL
========================================
Memory-efficient training on RTX 4080 (11.5 GB VRAM available)
Uses pre-tokenization to GPU for ~40-50% epoch speedup
"""

import time
from pdcl_backend import GPU_AVAILABLE, xp as np
from pdcl_train import train_pdcl
from pdcl_utils import PDCLConfig

# ─────────────────────────────────────────────
# GPU MEMORY MONITOR
# ─────────────────────────────────────────────

def log_gpu_memory(label=""):
    """Print current GPU memory usage."""
    if GPU_AVAILABLE:
        try:
            import cupy as cp_internal
            device = cp_internal.cuda.Device()
            mem_info = device.mem_info
            free_gb = mem_info[0] / (1024 ** 3)
            used_gb = (mem_info[1] - mem_info[0]) / (1024 ** 3)
            total_gb = mem_info[1] / (1024 ** 3)
            util_pct = (used_gb / total_gb) * 100
            print(f"[GPU {label}] Used: {used_gb:.2f} GB / {total_gb:.2f} GB ({util_pct:.1f}%) | Free: {free_gb:.2f} GB")
        except Exception as e:
            print(f"[GPU {label}] (could not read VRAM: {e})")


# ─────────────────────────────────────────────
# MEMORY-OPTIMIZED CONFIG FOR 11.5 GB VRAM
# ─────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"  PDCL GPU Training — RTX 4080 (11.5 GB VRAM)")
print(f"{'='*70}\n")

log_gpu_memory("Initial")

# Create memory-optimized config
config = PDCLConfig(
    # ─────────────────────────────────────────
    # Model Architecture (balanced for 11.5 GB)
    # ─────────────────────────────────────────
    n_dimensions=16,        # Keep high for better graph
    d_model=512,            # Max model capacity per clone (~7.5M parameters)
    n_heads=16,             # d_model must be divisible by n_heads
    max_doc=384,            # Document max length (increased from 256 for larger context capacity)
    max_q=32,               # Question max length
    
    # ─────────────────────────────────────────
    # Training (memory-conscious)
    # ─────────────────────────────────────────
    epochs=80,              # Enough for convergence with early stopping
    steps_per_epoch=50,     # Per-epoch training steps (scaled for 700 samples)
    batch_size=4,           # Adjusted to 4 to fit the 120M parameters safely in 11.5GB VRAM
    lr=0.005,               # Learning rate
    clip_norm=1.0,          # Gradient clipping
    
    # ─────────────────────────────────────────
    # Burst Backprop
    # ─────────────────────────────────────────
    burst_beta=2.0,         # Sharpness of burst (unchanged)
    
    # ─────────────────────────────────────────
    # Adaptive Soft Pruning
    # ─────────────────────────────────────────
    k_factor=0.5,           # Soft pruning strength
    gate_sharpness=3.0,     # Gate sharpness
    
    # ─────────────────────────────────────────
    # Feature Pruning
    # ─────────────────────────────────────────
    fp_base_pct=10.0,       # Start pruning 10%
    fp_max_pct=35.0,        # Max prune to 35%
    fp_warmup_epochs=3,     # Warmup before feature pruning starts
    fp_ema_decay=0.95,      # EMA decay for importance tracking
    
    # ─────────────────────────────────────────
    # Connection Pruning
    # ─────────────────────────────────────────
    conn_prune_threshold=0.02,
    conn_prune_from_epoch=3,
    conn_ema_decay=0.95,
    
    # ─────────────────────────────────────────
    # Cross-Dimension Graph
    # ─────────────────────────────────────────
    graph_correlation_lambda=0.5,
    graph_ema_decay=0.9,
    graph_min_epochs=2,
    
    # ─────────────────────────────────────────
    # Data Paths
    # ─────────────────────────────────────────
    train_subset_size=10500, # Use ALL training samples (pre-tokenized to GPU)
    val_subset_size=2250,    # Use ALL validation samples
    train_data_path='./data/train.json',
    val_data_path='./data/val.json',
    tokenizer_path='./tokenizer',
    
    # ─────────────────────────────────────────
    # Checkpointing
    # ─────────────────────────────────────────
    checkpoint_path='./checkpoints/pdcl_gpu_trained.pkl',
    save_interval=2,        # Save every 2 epochs
)

print("Configuration:")
print(f"  • Dimensions       : {config.n_dimensions}")
print(f"  • d_model          : {config.d_model}")
print(f"  • Batch size       : {config.batch_size}")
print(f"  • Train samples    : {config.train_subset_size} (pre-tokenized to GPU)")
print(f"  • Val samples      : {config.val_subset_size}")
print(f"  • Epochs           : {config.epochs}")
print(f"  • Steps/epoch      : {config.steps_per_epoch}")
print()

log_gpu_memory("Pre-training")

# ─────────────────────────────────────────────
# START TRAINING
# ─────────────────────────────────────────────

print(f"{'='*70}")
print(f"  STARTING GPU TRAINING")
print(f"{'='*70}\n")

t0 = time.time()

try:
    train_pdcl(config=config)
except KeyboardInterrupt:
    print("\n⏹ Training interrupted by user")
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    raise

elapsed = time.time() - t0

# ─────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────

print(f"\n{'='*70}")
print(f"  GPU TRAINING COMPLETE")
print(f"{'='*70}")
print(f"Total time: {elapsed/60:.1f} minutes")
log_gpu_memory("Final")
print()
