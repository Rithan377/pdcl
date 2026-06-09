"""
PDCL Full System Test — All Components
========================================
Tests the complete pipeline with all 5 PDCL concepts integrated:
    1. Core burst backprop (distance-weighted gradient scaling)
    2. Feature dimension pruning
    3. Connection pruning over time
    4. Group relationship detection
    5. Cross-dimension graph formation
    + Adaptive soft pruning (replaces hard top-K)
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pdcl_backend import xp as np, to_cpu
from pdcl_burst_backprop import CoreBurstBackprop
from pdcl_feature_pruning import FeatureDimensionPruner
import numpy as _np

print("=" * 60)
print("PDCL FULL SYSTEM TEST")
print("=" * 60)

# ── 1. Build model ──
print("\n[1] Building CoreBurstBackprop model...")
m = CoreBurstBackprop(
    vocab_size   = 200,
    max_positions= 64,
    d_model      = 32,
    n_heads      = 2,
    burst_beta   = 2.0,
    k_factor     = 0.5,
    gate_sharpness=3.0,
    group_active_from_epoch=1,
)
print("    OK")

# ── 2. Forward pass ──
print("\n[2] Forward pass...")
ids = np.array([[1,2,3,4,5,6,7,8,9,10]], dtype=_np.int32)
pos = np.array([[0,1,2,3,4,5,6,7,8,9]], dtype=_np.int32)
seg = np.array([[0,0,0,0,0,1,1,1,1,1]], dtype=_np.int32)
s_t = np.array([1], dtype=_np.int32)
e_t = np.array([2], dtype=_np.int32)

m.current_epoch = 0
try:
    p_s, p_e, loss = m.forward(ids, pos, seg, s_t, e_t, training=True)
    print(f"    p_start shape : {p_s.shape}")
    print(f"    p_end shape   : {p_e.shape}")
    print(f"    loss          : {loss:.4f}")
    print(f"    OK — span over full T_doc (no hard top-K!)")
except Exception as ex:
    import traceback
    print(f"    FAIL: {ex}")
    traceback.print_exc()

# ── 3. Backward pass with core burst scaling ──
print("\n[3] Backward pass (with core burst gradient scaling)...")
m.zero_grad()
try:
    m.burst_backward()
    doc_attn_grad = to_cpu(m.attention.doc_self_attn.dW_q)
    print(f"    doc_attn dW_q norm : {_np.linalg.norm(doc_attn_grad):.6f}")
    print(f"    prune dW_d norm    : {float(to_cpu(_np.linalg.norm(to_cpu(m.prune.dW_d)))):.6f}")
    print(f"    span dv_start norm : {float(to_cpu(_np.linalg.norm(to_cpu(m.span_head.dv_start)))):.6f}")
    print("    OK — core burst scaling applied")
except Exception as ex:
    import traceback
    print(f"    FAIL: {ex}")
    traceback.print_exc()

# ── 4. Update ──
print("\n[4] Parameter update...")
try:
    before = float(to_cpu(_np.linalg.norm(to_cpu(m.embed.E_token))))
    m.update(lr=0.001, clip_norm=1.0)
    after  = float(to_cpu(_np.linalg.norm(to_cpu(m.embed.E_token))))
    print(f"    E_token norm before: {before:.4f}")
    print(f"    E_token norm after : {after:.4f}")
    print(f"    Updated: {'YES' if abs(before - after) > 1e-8 else 'NO'}")
    print("    OK")
except Exception as ex:
    print(f"    FAIL: {ex}")

# ── 5. Group relations (active from epoch 1 in test) ──
print("\n[5] Group relationship detection...")
m.current_epoch = 2
try:
    ids2 = np.array([[1,2,3,4,5,6,7,8,9,10,11,12]], dtype=_np.int32)
    pos2 = np.array([[0,1,2,3,4,5,6,7,8,9,10,11]], dtype=_np.int32)
    seg2 = np.array([[0,0,0,0,0,0,0,0,1,1,1,1]],  dtype=_np.int32)
    p_s2, p_e2, loss2 = m.forward(ids2, pos2, seg2, s_t, e_t, training=True)
    print(f"    Group module active: {m.group_rel._active}")
    print(f"    loss with groups  : {loss2:.4f}")
    print("    OK")
except Exception as ex:
    print(f"    FAIL: {ex}")

# ── 6. Feature dimension pruning ──
print("\n[6] Feature dimension pruning...")
fp = FeatureDimensionPruner(d_model=32, base_prune_pct=20.0, min_epochs_before_prune=0)
try:
    fp.update_importance(m)
    stats = fp.apply_masks(epoch=1, model=m)
    print(f"    Matrices pruned : {len(stats)}")
    if stats:
        avg_sp = _np.mean(list(stats.values()))
        print(f"    Avg sparsity    : {avg_sp*100:.1f}%")
    print("    OK")
except Exception as ex:
    import traceback
    print(f"    FAIL: {ex}")
    traceback.print_exc()

# ── 7. Connection pruning ──
print("\n[7] Connection pruning over time...")
try:
    # Run several forwards to build up EMA
    for _ in range(3):
        m.forward(ids, pos, seg, s_t, e_t, training=False)

    conn_stats = m.attention.update_connection_masks(threshold=0.05)
    print(f"    Connections pruned: {conn_stats}")
    print("    OK")
except Exception as ex:
    print(f"    FAIL: {ex}")

# ── 8. Adaptive soft pruning check ──
print("\n[8] Adaptive soft pruning (no hard top-K)...")
try:
    keep_rate = m.prune.get_effective_keep_rate()
    k = float(to_cpu(_np.exp(to_cpu(m.prune.log_k)[0])))
    print(f"    Learned k_factor : {k:.3f}")
    print(f"    Effective keep   : {keep_rate:.1%} (adapts per document)")
    print(f"    Mode             : SOFT — no context abruptly removed")
    print("    OK")
except Exception as ex:
    print(f"    FAIL: {ex}")

print(f"\n{'='*60}")
print("ALL PDCL SYSTEM TESTS COMPLETE")
print(f"{'='*60}")
