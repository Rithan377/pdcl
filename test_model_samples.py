import json
import os
import sys

# Add current directory to path
sys.path.append("/home/rithan/pdcl_v2_full")

from pdcl_tokenizer import BPETokenizer
from pdcl_burst_backprop import CoreBurstBackprop
from pdcl_backend import xp, to_cpu
from pdcl_dimension_engine import align_answer_to_tokens, AnswerAlignmentError
import numpy as np

class Config:
    tokenizer_path = "./tokenizer"
    val_data_path = "./data/val.json"
    d_model = 512
    n_heads = 16
    max_doc = 384
    max_q = 32
    burst_beta = 2.0
    k_factor = 0.5
    gate_sharpness = 3.0

def test_samples():
    config = Config()
    
    # 1. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.load(config.tokenizer_path)
    
    # 2. Build model and load checkpoint
    print("Building model...")
    model = CoreBurstBackprop(
        vocab_size=tokenizer.vocab_size,
        max_positions=config.max_doc + config.max_q + 4,
        d_model=config.d_model,
        n_heads=config.n_heads,
        burst_beta=config.burst_beta,
        k_factor=config.k_factor,
        gate_sharpness=config.gate_sharpness,
    )
    
    best_checkpoint = "./checkpoints/pdcl_gpu_trained_loss_ratio_best.pkl.npz"
    final_checkpoint = "./checkpoints/pdcl_gpu_trained_loss_ratio.pkl.npz"
    
    checkpoint_to_load = final_checkpoint
    
    if os.path.exists(checkpoint_to_load):
        print(f"Loading checkpoint from: {checkpoint_to_load}")
        model.load_checkpoint(checkpoint_to_load)
    else:
        print("Error: No checkpoint found!")
        return

    # 3. Load Validation Data
    print("Loading validation dataset...")
    with open(config.val_data_path) as f:
        val_data = json.load(f)
    
    # Let's extract some samples
    print("Preparing validation samples...")
    
    valid_samples = []
    for sample in val_data:
        try:
            # Check if this sample aligns correctly (not skipped)
            align_answer_to_tokens(sample, tokenizer, max_doc=config.max_doc, max_q=config.max_q)
            valid_samples.append(sample)
            if len(valid_samples) >= 5:
                break
        except AnswerAlignmentError:
            continue
            
    print(f"Found {len(valid_samples)} valid samples to test on.")
    print("=" * 80)
    
    # 4. Run prediction
    sep_id = tokenizer.token2id.get('<sep>', 2)
    
    for i, sample in enumerate(valid_samples):
        print(f"\nSAMPLE #{i+1}")
        print(f"Question : {sample['question']}")
        print(f"Document : {sample['document'][:180]}...")
        print(f"Target   : {sample['answer']}")
        print("-" * 50)
        
        # Tokenize and format inputs
        qa_enc, ans_s, ans_e = align_answer_to_tokens(
            sample,
            tokenizer,
            max_doc=config.max_doc,
            max_q=config.max_q
        )
        
        # Move to GPU/XP arrays
        token_ids = xp.array([qa_enc['token_ids']], dtype=xp.int32)
        positions = xp.array([qa_enc['positions']], dtype=xp.int32)
        segment_mask = xp.array([qa_enc['segment_mask']], dtype=xp.int32)
        start_t = xp.array([ans_s], dtype=xp.int32)
        end_t = xp.array([ans_e], dtype=xp.int32)
        
        # Forward pass
        model.current_epoch = 80
        p_start, p_end, loss = model.forward(
            token_ids=token_ids,
            positions=positions,
            segment_mask=segment_mask,
            start_targets=start_t,
            end_targets=end_t,
            training=False
        )
        
        # Determine document length limit (up to sep_id)
        sep_indices = xp.where(token_ids[0] == sep_id)[0]
        if len(sep_indices) > 0:
            T_doc = int(to_cpu(sep_indices[0])) - 1
        else:
            T_doc = len(token_ids[0]) - 1
            
        doc_ids = to_cpu(token_ids[0, 1:])  # skip CLS token
        
        # --- METHOD A: Independent argmax ---
        pred_s_ind = int(to_cpu(xp.argmax(p_start[0])))
        pred_e_ind = int(to_cpu(xp.argmax(p_end[0])))
        pred_s_ind = max(1, min(pred_s_ind, T_doc))
        pred_e_ind = max(1, min(pred_e_ind, T_doc))
        if pred_e_ind < pred_s_ind:
            pred_e_ind = pred_s_ind
            
        pred_ids_ind = doc_ids[pred_s_ind - 1: pred_e_ind]
        pred_ans_ind = tokenizer.decode(pred_ids_ind.tolist())
        
        # --- METHOD B: Joint span search (max length 8) ---
        max_span_len = 8
        best_score = -1.0
        best_s, best_e = 1, 1
        
        for s in range(1, T_doc + 1):
            for e in range(s, min(s + max_span_len, T_doc + 1)):
                score = float(to_cpu(p_start[0, s] * p_end[0, e]))
                if score > best_score:
                    best_score = score
                    best_s, best_e = s, e
                    
        pred_ids_joint = doc_ids[best_s - 1: best_e]
        pred_ans_joint = tokenizer.decode(pred_ids_joint.tolist())
        
        print(f"Method A (Independent Argmax) Predict: '{pred_ans_ind}'")
        print(f"Method B (Joint Span Search)  Predict: '{pred_ans_joint}'")
        print("=" * 80)

if __name__ == "__main__":
    test_samples()
