import os
import sys
import json
import traceback

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pdcl_backend import GPU_AVAILABLE
from pdcl_utils import PDCLConfig
from pdcl_tokenizer import BPETokenizer
from pdcl_burst_backprop import CoreBurstBackprop
from pdcl_dimension_engine import ParallelDimensionEngine
from pdcl_utils import PreTokenizedDataset

def main():
    print("GPU_AVAILABLE:", GPU_AVAILABLE)
    if not GPU_AVAILABLE:
        print("GPU is not available. Please run with GPU active.")
        sys.exit(1)

    config = PDCLConfig(
        n_dimensions=2, # Keep it small for debugging
        d_model=128,
        n_heads=4,
        max_doc=256,
        max_q=32,
        epochs=1,
        steps_per_epoch=1,
        batch_size=2,
        lr=0.005,
        train_subset_size=10,
        val_subset_size=5,
        train_data_path='./data/train.json',
        val_data_path='./data/val.json',
        tokenizer_path='./tokenizer',
        checkpoint_path='./checkpoints/debug_pdcl.pkl',
    )

    print("Loading tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(config.tokenizer_path)

    print("Loading data...")
    with open(config.train_data_path) as f:
        train_data = json.load(f)
    train_samples = train_data[:config.train_subset_size]

    print("Building master model...")
    master = CoreBurstBackprop(
        vocab_size=tokenizer.vocab_size,
        max_positions=config.max_doc + config.max_q + 4,
        d_model=config.d_model,
        n_heads=config.n_heads,
        burst_beta=config.burst_beta,
        k_factor=config.k_factor,
        gate_sharpness=config.gate_sharpness,
    )

    print("Pre-tokenizing data...")
    pre_tokenized = PreTokenizedDataset(
        samples=train_samples,
        tokenizer=tokenizer,
        max_doc=config.max_doc,
        max_q=config.max_q,
        gpu_available=True
    )

    print("Initializing parallel engine...")
    engine = ParallelDimensionEngine(
        n_dimensions=config.n_dimensions,
        master_model=master,
        pre_tokenized_dataset=pre_tokenized,
        max_doc=config.max_doc,
        max_q=config.max_q,
    )

    print("Running a single parallel step to trigger and capture the error...")
    try:
        # Run _run_dimension directly to see the traceback
        engine._run_dimension(dim_id=0, batch_size=config.batch_size)
    except Exception as e:
        print("\n=== CAPTURED EXCEPTION ===")
        print(f"Error type: {type(e)}")
        print(f"Error message: {e}")
        print("Traceback:")
        traceback.print_exc()
        print("==========================\n")

if __name__ == '__main__':
    main()
