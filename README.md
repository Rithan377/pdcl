# Parallel Dimensional Core Learning (PDCL)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CuPy](https://img.shields.io/badge/CuPy-CUDA%20Acceleration-76B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)](https://cupy.dev/)
[![NumPy](https://img.shields.io/badge/NumPy-Zero%20Framework-013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Hardware](https://img.shields.io/badge/Validated%20On-RTX%204080%20(11.5GB)-76B900.svg?style=for-the-badge&logo=nvidia&logoColor=white)](https://www.nvidia.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> **A Library-Free, Graph-Based Multidimensional Deep Learning Architecture with Compiler-Style Parallel Training, Adaptive Pruning, and Graph-Weighted Gradient Aggregation**

**Author**: **Rithan Nandakumar** ([@Rithan377](https://github.com/Rithan377))  
Department of Artificial Intelligence and Data Science, Anna University, Chennai 600025, India  
*Corresponding Contact*: `nsrithan15@gmail.com` | `he/him`

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. The PDCL Philosophy (Built from First Principles)](#2-the-pdcl-philosophy-built-from-first-principles)
- [3. Core Architectural Mechanisms](#3-core-architectural-mechanisms)
  - [3.1 Parallel Dimension Engine (PDE)](#31-parallel-dimension-engine-pde)
  - [3.2 Cross-Dimension Correlation Graph (CDCG)](#32-cross-dimension-correlation-graph-cdcg)
  - [3.3 Core Burst Backpropagation (CBB)](#33-core-burst-backpropagation-cbb)
  - [3.4 Adaptive Soft Pruning (ASP)](#34-adaptive-soft-pruning-asp)
  - [3.5 Three-Way Group Relationship Detector (GRD)](#35-three-way-group-relationship-detector-grd)
- [4. Three-Phase Evolutionary Training Pipeline](#4-three-phase-evolutionary-training-pipeline)
- [5. GPU Performance Optimization Engine](#5-gpu-performance-optimization-engine)
- [6. Robust Multi-Strategy Fuzzy Alignment](#6-robust-multi-strategy-fuzzy-alignment)
- [7. Repository File Map & Modular Architecture](#7-repository-file-map--modular-architecture)
- [8. Installation & Environment Setup](#8-installation--environment-setup)
- [9. Quick Start & Training Workflows](#9-quick-start--training-workflows)
- [10. Hyperparameter Configuration Reference (`PDCLConfig`)](#10-hyperparameter-configuration-reference-pdclconfig)
- [11. Experimental Verification & Benchmarks](#11-experimental-verification--benchmarks)
- [12. Roadmap & Extending PDCL](#12-roadmap--extending-pdcl)
- [13. Citation](#13-citation)

---

## 1. Executive Summary

**Parallel Dimensional Core Learning (PDCL)** is a deep learning architecture designed from first principles without relying on high-level deep learning frameworks (no PyTorch, TensorFlow, JAX, or Keras). Every forward pass, backward pass, gradient accumulation, optimizer update, attention operator, and loss function is derived from raw matrix calculus and implemented using only **NumPy** (CPU) and **CuPy** (NVIDIA CUDA GPU).

PDCL addresses key architectural limitations in monolithic transformer models:
1. **Rigid Uniform Gradient Averaging**: In standard data-parallel training, gradients from all workers/mini-batches are uniformly averaged ($\frac{1}{D} \sum g_i$), treating noisy and informative updates identically. PDCL constructs an evolving **Cross-Dimension Correlation Graph** based on Pearson correlation of hidden state representations, performing data-driven, degree-normalized gradient aggregation.
2. **Uniform Weight Updating**: Conventional backpropagation updates all weights uniformly according to their instantaneous gradients. PDCL's **Core Burst Backpropagation** identifies the singular highest-variance "burst center" feature in each parameter matrix and applies an information-guided exponential decay outwards, concentrating learning energy where signal representation is most acute.
3. **Non-Differentiable Hard Pruning**: Standard top-$K$ sparsification creates non-differentiable step boundaries and ignores token score distributions. PDCL implements **Adaptive Soft Pruning (ASP)** using a continuous sigmoid gate parameterized dynamically by the per-document mean ($\mu_s$) and standard deviation ($\sigma_s$) of token relevance scores.
4. **Shallow Pairwise Interactions**: Transformers compute purely pairwise bilinear attention ($Q K^T$). PDCL integrates a **Three-Way Group Relationship Detector** that computes trilinear third-order tensor interactions across token triples.

```
                     ┌─────────────────────────────────────────────────────────┐
                     │                   INPUT DOCUMENT & QUERY                │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │          CUSTOM BPE TOKENIZER & 3-WAY EMBEDDING         │
                     │           (Token + Sinusoidal Pos + Segment ID)         │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │             DUAL DOCUMENT-QUERY SEGMENTATION            │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │               PARALLEL ATTENTION CONTROLLER             │
                     │    ├── Document Self-Attention (with EMA conn tracking) │
                     │    ├── Query Self-Attention    (with EMA conn tracking) │
                     │    └── Query-to-Doc Cross Attention                     │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │             ADAPTIVE SOFT PRUNING (ASP) GATE            │
                     │          τ = μ_s + k · σ_s  (Continuous Sigmoid)        │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │             GROUP RELATIONSHIP DETECTOR (GRD)           │
                     │             (Trilinear Third-Order Interaction)         │
                     └────────────────────────────┬────────────────────────────┘
                                                  │
                                                  ▼
                     ┌─────────────────────────────────────────────────────────┐
                     │           SPAN PREDICTION HEAD & BURST BACKPROP         │
                     │           (Start/End Logits + Burst Center Focus)       │
                     └─────────────────────────────────────────────────────────┘
```

---

## 2. The PDCL Philosophy (Built from First Principles)

Modern deep learning research often abstracts the underlying mathematics behind black-box framework autograd engines. PDCL was developed to prove that a radically different learning paradigm can be built and scaled entirely from raw tensor operations:

- **Zero Framework Overhead**: No `torch.autograd`, `tf.GradientTape`, or framework C++ runtime overhead.
- **Hand-Derived Analytical Gradients**: All backward passes for Multi-Head Attention, Cross Attention, LayerNorm, Trilinear Group Detectors, Soft Gating, and Span Cross-Entropy are analytically computed and verified.
- **Unified Hardware Dispatch**: Through `pdcl_backend.py`, operations transparently execute on CUDA hardware via CuPy when an NVIDIA GPU is detected, automatically falling back to NumPy for CPU execution without code duplication.

---

## 3. Core Architectural Mechanisms

### 3.1 Parallel Dimension Engine (PDE)

Instead of a single monolithic network processing a batch sequentially, the **Parallel Dimension Engine** instantiates $D$ independent processing clones of the core architecture:

$$M_1, M_2, \dots, M_D$$

- The training data batch is partitioned across $D$ dimensional streams.
- Each dimension clone independently executes forward inference, computes task loss, and calculates backward gradients for its partition.
- Clones communicate and synchronize through a central **Master Model** via graph-weighted aggregation.

### 3.2 Cross-Dimension Correlation Graph (CDCG)

Standard distributed training aggregates gradients via simple uniform arithmetic mean:

$$\bar{G} = \frac{1}{D} \sum_{d=1}^D G_d$$

PDCL rejects this assumption. After each forward pass, the activation signatures $h_d \in \mathbb{R}^{B \times d_{\text{model}}}$ across all dimensions are tracked. The engine computes the pairwise Pearson correlation matrix $R \in \mathbb{R}^{D \times D}$:

$$R_{i,j} = \frac{\sum_{k} (h_{i,k} - \bar{h}_i)(h_{j,k} - \bar{h}_j)}{\sqrt{\sum_{k} (h_{i,k} - \bar{h}_i)^2} \sqrt{\sum_{k} (h_{j,k} - \bar{h}_j)^2}}$$

From $R$, an adjacency matrix $A$ is formed by thresholding at $\rho > 0$:

$$A_{i,j} = \begin{cases} R_{i,j} & \text{if } R_{i,j} \ge \rho \\ 0 & \text{otherwise} \end{cases}$$

The dynamic aggregation weight $w_d$ for dimension $d$ is derived from its graph degree centrality:

$$w_d = \frac{\sum_{j=1}^D A_{d,j} + \epsilon}{\sum_{i=1}^D \sum_{j=1}^D A_{i,j} + D \cdot \epsilon}, \quad \sum_{d=1}^D w_d = 1$$

Dimensions that learn representations well-aligned with the dimensional consensus receive higher gradient voting weight, while noisy, divergent dimensions have their gradients attenuated automatically.

```
                      [Dim 1] <──── (R_1,2) ────> [Dim 2]
                         ▲                          ▲
                         │ \                      / │
                      (R_1,3) \                / (R_2,4)
                         │     \              /     │
                         ▼      ▼            ▼      ▼
                      [Dim 3] <──── (R_3,4) ────> [Dim 4]
                                       │
                         [Graph Adjacency Centrality]
                                       ▼
                       w_d = degree(Dim_d) / total_degree
                                       ▼
                   G_master = Σ (w_d · G_clone_d)
```

### 3.3 Core Burst Backpropagation (CBB)

In biological neural circuits, localized bursts of electrical activity drive synaptic plasticity. PDCL mimics this with **Core Burst Backpropagation**:

1. For each parameter matrix $W \in \mathbb{R}^{d_{in} \times d_{out}}$ and incoming gradient $\nabla_W L$, the engine determines the **burst center** $f^*$ — the column with maximum gradient magnitude:

   $$f^* = \arg\max_j \sum_i |\nabla_{W_{i,j}} L|$$

2. An information-distance metric is evaluated between each feature column $j$ and the burst center $f^*$:

   $$\text{dist}(j, f^*) = \frac{|j - f^*|}{d_{out}}$$

3. Gradients are scaled by an exponential decay envelope parameterized by $\beta \ge 0$:

   $$\tilde{\nabla}_{W_{i,j}} L = \nabla_{W_{i,j}} L \cdot \exp\left(-\beta \cdot \text{dist}(j, f^*)\right)$$

This forces gradient updates to intensely optimize the most salient feature channel while softly stabilizing peripheral channels against noisy fluctuations.

### 3.4 Adaptive Soft Pruning (ASP)

Instead of hard top-$K$ selection (which kills gradient flow to non-selected tokens), PDCL introduces a continuous, differentiable soft gating mechanism:

1. A projection layer calculates token relevance scores $s_t \in \mathbb{R}$ for each document token:

   $$s = \left( X_{\text{doc}} W_d \right) \odot \left( c_{\text{que}} W_q \right)$$

2. The pruning threshold $\tau$ adapts per document dynamically based on the document's own score distribution:

   $$\tau = \mu_s + k \cdot \sigma_s$$

   where $\mu_s = \text{mean}(s)$, $\sigma_s = \text{std}(s)$, and $k \in \mathbb{R}$ controls selectivity.

3. The soft retention gate $g_t \in (0, 1)$ is evaluated via a temperature-scaled sigmoid:

   $$g_t = \sigma\left(\alpha \cdot (s_t - \tau)\right) = \frac{1}{1 + e^{-\alpha (s_t - \tau)}}$$

Document representations are gated via $X_{\text{gated}} = X_{\text{doc}} \odot g$. Because $\sigma$ is everywhere differentiable, gradients flow backward not only to the retained tokens, but also through $\tau$, $\mu_s$, and $\sigma_s$ to optimize the threshold boundary itself.

### 3.5 Three-Way Group Relationship Detector (GRD)

While multi-head attention excels at pairwise token correlation ($t_i \leftrightarrow t_j$), natural language contains complex tripartite dependencies (e.g., `Subject - Verb - Object` or `Entity - Attribute - Value`).

PDCL computes higher-order interactions by projecting token states into three distinct functional representations:

$$U = X W_u, \quad V = X W_v, \quad Z = X W_z$$

The three-way group interaction tensor $T$ is computed and compressed via trilinear fusion:

$$\text{GRD}(X) = \text{LayerNorm}\left( (U \odot V) W_{\text{uv}} \odot Z \right) W_{\text{out}}$$

This captures tri-token contextual dependencies directly with $O(T \cdot d)$ computational complexity.

---

## 4. Three-Phase Evolutionary Training Pipeline

PDCL does not use static training dynamics. Instead, it employs a biologically inspired **Three-Phase Maturity Schedule**:

```
 [Epoch 0] ─────────────────── [Maturity Epoch] ─────────────────── [Unfreeze Epoch] ─────────────> [Final]
     │                               │                                     │
     ▼                               ▼                                     ▼
 ┌──────────────────────┐      ┌───────────────────────────────┐      ┌──────────────────────────────┐
 │       PHASE 1        │      │            PHASE 2            │      │           PHASE 3            │
 │     Exploration      │      │      Structured Pruning       │      │      Global Fine-Tuning      │
 ├──────────────────────┤      ├───────────────────────────────┤      ├──────────────────────────────┤
 │ • All weights learn  │ ───> │ • Core Burst Freezing: ON     │ ───> │ • Unfreeze all remaining     │
 │ • Build graph corr   │      │ • Feature Pruning: ON         │      │   non-pruned weights         │
 │ • LR warmup          │      │ • Connection Pruning: ON      │      │ • Pruned masks locked        │
 │ • Track feature EMAs │      │ • Conservative ramp-up        │      │ • Recover peak EM / F1       │
 └──────────────────────┘      └───────────────────────────────┘      └──────────────────────────────┘
```

### Phase 1: Unconstrained Exploration (Epoch 0 $\to$ Maturity)
- All network parameters are completely plastic and receive full gradient updates.
- The Cross-Dimension Correlation Graph continuously tracks activation similarities and learns optimal inter-dimension routing.
- Exponential Moving Averages (EMA) of attention head connections and feature column activations accumulate historical stability metrics.

### Phase 2: Structured Pruning & Core Freezing (Maturity $\to$ Fine-Tuning)
Triggered when validation exact match / loss hits the maturity criteria:
1. **Core Burst Freezing**: Foundational weights with high historical confidence are frozen to prevent catastrophic forgetting.
2. **Feature Dimension Pruning**: Underperforming feature channels are masked to zero. Strict **gradient masks** are applied directly to gradient buffers, preventing pruned channels from receiving updates and reviving during optimization.
3. **Connection Pruning**: Infrequently traversed attention paths are pruned based on cross-dimension EMA averages. Pruning utilizes a safety ramp:
   - Capped at maximum 10% connection pruning per event.
   - Enforces a safety floor (at least 30% of connections remain unconditionally active).

### Phase 3: Global Fine-Tuning & Recovery
After $N$ epochs of pruning stabilization, Phase 3 unfreezes all remaining active parameters:
- Core burst freezing is deactivated.
- Pruned feature columns and sparse connection topologies remain permanently locked.
- The remaining sparse sub-network trains end-to-end at a gentle learning rate, recovering representation capacity and reaching peak extraction accuracy.

---

## 5. GPU Performance Optimization Engine

The latest PDCL engine features an optimized GPU backend (`CuPy`) engineered to saturate modern NVIDIA architectures (such as the RTX 4080):

1. **Zero Host-Device Roundtrips**: All intermediate forward states, cache dictionaries, attention weights, and gradient buffers stay resident in GPU VRAM during training.
2. **Batched 2D GEMM Flattening**:
   Instead of allocating high-dimensional 3D/4D intermediate tensors for outer products and reducing across axes:
   $$\sum_{b=1}^B O_b^T \nabla_{O_b} \quad \Longrightarrow \quad O_{\text{flat}}^T \nabla_{O,\text{flat}} \quad \text{where } O_{\text{flat}} \in \mathbb{R}^{(B \cdot T) \times d}$$
   This dispatches directly to high-throughput NVIDIA cuBLAS GEMM kernels.
3. **Pure GPU In-Place AXPY Gradient Accumulation**:
   Dimension clone gradients are accumulated into master buffers using pure device-side fused multiply-adds:
   ```python
   m_buf += c_buf * np.float32(weight)  # Zero host CPU intervention
   ```
4. **Vectorized GPU Cross-Entropy Loss**:
   Replaces per-sample CPU extraction loops with parallel batched indexing and log-sum reductions on device.

---

## 6. Robust Multi-Strategy Fuzzy Alignment

In extractive QA over noisy documents (OCR artifacts, corrupted whitespace, irregular unicode, scan errors), strict substring matching frequently fails. Traditional systems either crash or inject incorrect token indices into the loss, poisoning gradients.

PDCL solves this with a multi-tier alignment algorithm:

```
                  ┌──────────────────────────────────────────────┐
                  │          Raw Answer String & Tokens          │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                             [1. Exact Token Subsequence]
                                     Match found?
                                   ├── YES ──> Return (start_idx, end_idx)
                                   └── NO  ──> Proceed to Step 2
                                         │
                                         ▼
                             [2. Exact Character Search]
                                     Match found?
                                   ├── YES ──> Map chars to token bounds
                                   └── NO  ──> Proceed to Step 3
                                         │
                                         ▼
                            [3. Levenshtein Fuzzy Search]
                                 Tolerates Edit Distance ≤ 3
                           (e.g., "$1,261,000,000" ↔ "1,261,000,000")
                                     Match found?
                                   ├── YES ──> Map fuzzy span to tokens
                                   └── NO  ──> Proceed to Step 4
                                         │
                                         ▼
                             [4. Outlier Sample Skipped]
                             Gracefully skip noisy sample.
                         Zero gradient poisoning. Training continues.
```

---

## 7. Repository File Map & Modular Architecture

```
pdcl/
├── pdcl_backend.py              # Hardware abstraction: transparent CuPy (GPU) / NumPy (CPU) switching
├── pdcl_tokenizer.py            # Custom Byte-Pair Encoding (BPE) tokenizer and vocabulary builder
├── pdcl_embedding.py            # Three-component embeddings (Token + Position + Segment) & LayerNorm
├── pdcl_segmentation.py         # Dual document-query boundary segmentation module
├── pdcl_attention.py            # Multi-head document/query self-attention, cross-attention & sparse EMA
├── pdcl_pruning.py              # Distribution-adaptive sigmoid soft pruning (ASP) layer
├── pdcl_group_relations.py      # Three-way trilinear group relationship detector (GRD)
├── pdcl_burst_backprop.py       # Core Burst Backprop mechanism, span prediction heads & master model
├── pdcl_dimension_engine.py     # Parallel Dimension Engine, clone execution & graph-weighted sync
├── pdcl_graph_formation.py      # Pearson correlation matrix calculation & aggregation weight derivation
├── pdcl_feature_pruning.py      # Feature column importance tracking, mask generator & gradient zeroing
├── pdcl_train.py                # Unified 3-phase training coordinator, validation loop & checkpointing
├── pdcl_utils.py                # Centralized PDCLConfig, 15-point validation engine & fuzzy matchers
├── pdcl_dataset_generator.py    # Synthetic / realistic noisy document QA corpus generator
├── train_gpu.py                 # Out-of-the-box launcher for standard GPU training
├── train_gpu_loos_ratio_scaling.py # Launcher featuring loss-ratio dynamic learning rate scaling
├── test_model_samples.py        # Interactive evaluation script on test samples with span decoding
├── debug_alignment.py           # Diagnostic script for verifying token alignment across datasets
├── init_tokenizer.py            # Script to pre-train and initialize tokenizer vocabulary
├── requirements.txt             # Python package dependencies
├── .gitignore                   # Clean repository hygiene excluding cache, venvs, and checkpoints
├── IMPROVEMENTS.md              # Historical log of structural and architectural revisions
└── USAGE.md                     # Quick usage guide and script walkthroughs
```

---

## 8. Installation & Environment Setup

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA 11.x, 12.x, or 13.x (Optional, but recommended for full parallel speedup)
- 16GB+ System RAM (and 8GB+ VRAM for GPU training)

### 1. Clone the Repository
```bash
git clone https://github.com/Rithan377/pdcl.git
cd pdcl
```

### 2. Create Virtual Environment
```bash
python3 -m venv myenv
source myenv/bin/activate
```

### 3. Install Dependencies
```bash
# Core numerical dependencies
pip install -r requirements.txt

# If using an NVIDIA GPU, install the matching CuPy package:
# For CUDA 12.x:
pip install cupy-cuda12x

# For CUDA 11.x:
# pip install cupy-cuda11x
```

---

## 9. Quick Start & Training Workflows

### Option A: GPU Training with Loss-Ratio Scaling (Recommended)
This launcher uses memory-optimized parameters tuned for modern GPUs (e.g., RTX 4080 with 11.5 GB VRAM) with dynamic loss-ratio learning rate scheduling:

```bash
python3 train_gpu_loos_ratio_scaling.py
```

### Option B: Standard Training Loop (Python API)
Create a custom configuration with built-in startup validation:

```python
from pdcl_utils import PDCLConfig, validate_config
from pdcl_train import train_pdcl

# 1. Instantiate hyperparameter configuration
config = PDCLConfig(
    # Model Dimensionality
    n_dimensions=16,        # Number of parallel dimension clones (D)
    d_model=512,            # Hidden feature dimension (divisible by n_heads)
    n_heads=16,             # Multi-head attention heads
    max_doc=384,            # Document maximum sequence length
    max_q=32,               # Question maximum sequence length

    # Training Dynamics
    epochs=50,
    steps_per_epoch=50,
    batch_size=4,
    lr=0.005,
    clip_norm=1.0,

    # Core Burst Backprop & Pruning
    burst_beta=2.0,         # Exponential gradient decay factor
    burst_freeze_threshold=0.02, # Burst freeze sensitivity
    k_factor=0.5,           # Soft pruning distribution offset (tau = mu + k*std)
    gate_sharpness=3.0,     # Sigmoid temperature sharpness (alpha)
    conn_prune_threshold=0.02,

    # Dynamic Unfreeze (Phase 3)
    unfreeze_after_pruning_epochs=10,
    keep_batch_size_phase2=True
)

# 2. Strict Pre-flight Validation (catches configuration issues immediately)
is_valid, error = validate_config(config)
if not is_valid:
    raise ValueError(f"Configuration validation failed: {error}")

# 3. Launch Training
history = train_pdcl(config)
```

### Option C: Evaluating Trained Models
Run extractive inference and inspect exact span predictions:

```bash
python3 test_model_samples.py
```

---

## 10. Hyperparameter Configuration Reference (`PDCLConfig`)

The `PDCLConfig` class in `pdcl_utils.py` centralizes all hyperparameters.

| Hyperparameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `n_dimensions` | `int` | `16` | Number of parallel processing dimension clones ($D$). |
| `d_model` | `int` | `512` | Model feature hidden size. Must be divisible by `n_heads`. |
| `n_heads` | `int` | `16` | Number of attention heads in parallel attention modules. |
| `max_doc` | `int` | `384` | Maximum document sequence length (tokens). |
| `max_q` | `int` | `32` | Maximum question sequence length (tokens). |
| `batch_size` | `int` | `4` | Batch size per dimension clone per step. |
| `epochs` | `int` | `50` | Maximum number of training epochs. |
| `steps_per_epoch` | `int` | `50` | Gradient accumulation steps executed per epoch. |
| `lr` | `float` | `0.005` | Base optimizer learning rate. |
| `clip_norm` | `float` | `1.0` | Global gradient clipping Euclidean norm threshold. |
| `burst_beta` | `float` | `2.0` | Exponential gradient decay parameter ($\beta$) for CBB. |
| `burst_freeze_threshold`| `float` | `0.02` | Gradient variance floor below which core weights are frozen. |
| `k_factor` | `float` | `0.5` | Threshold offset factor for Adaptive Soft Pruning ($\tau = \mu + k\sigma$). |
| `gate_sharpness` | `float` | `3.0` | Sigmoid steepness coefficient ($\alpha$) for the soft pruning gate. |
| `conn_prune_threshold` | `float` | `0.02` | Base attention connection EMA threshold for Phase 2 pruning. |
| `unfreeze_after_pruning_epochs` | `int` | `10` | Number of Phase 2 epochs before Phase 3 global unfreezing activates. |
| `keep_batch_size_phase2` | `bool` | `True` | Maintain stable batch size during Phase 2 rather than dropping to 1. |
| `max_phase2_lr` | `float` | `0.003` | Learning rate ceiling during Phase 2 to prevent divergence. |

---

## 11. Experimental Verification & Benchmarks

The architecture has been empirically validated on noisy document extractive question answering using an **NVIDIA GeForce RTX 4080 (11.5 GB VRAM)**.

### Training Profile Summary (from `training_output.log`)
- **Total Parameters Across Clones**: ~120 Million Parameters (16 dimensions $\times$ ~7.5M per clone)
- **VRAM Utilization**: Stable at 9.46 GB / 11.60 GB (81.6% capacity)
- **Zero OOM Incidents**: Ensured through flattened 2D GEMMs and dynamic memory management
- **Total Training Duration**: 197.2 minutes (~3.28 hours for 80 full epochs)
- **Adaptive Pruning Sparsity**: The soft pruning gate settled dynamically at retaining **~31%** of document tokens, eliminating nearly 70% of background document noise while retaining critical answer context.

### Sample Span Extractions Under Heavy Noise
```
----------------------------------------------------------------------
Question : How much net income did Cirrus Cloud Systems earn in Q3 2022?
Target   : 1,261,000,000
Predict  : $1,261,000,000
Noise    : HIGH | Gate Keep: ~31% | Status: MATCH (Exact Span Recovered)
----------------------------------------------------------------------
Question : What was Tundra Refrigeration's R&D expenditure in Q3 2023?
Target   : 112,000,000
Predict  : $112,000,000
Noise    : HIGH | Gate Keep: ~31% | Status: MATCH (Exact Span Recovered)
----------------------------------------------------------------------
```

---

## 12. Roadmap & Extending PDCL

PDCL provides a modular framework for machine learning researchers interested in exploring alternatives to standard backpropagation:

- [ ] **Flash Attention in Pure CuPy**: Developing custom fused CUDA kernels via `cupy.RawKernel` to implement memory-efficient attention without materializing the $T \times T$ attention matrix.
- [ ] **Distributed Multi-GPU PDE**: Expanding `ParallelDimensionEngine` across multiple physical GPUs using NCCL / CuPy Distributed.
- [ ] **Autoregressive Generative Decoding**: Extending the Core Burst Backprop mechanism to causal next-token generation and code synthesis.
- [ ] **Dynamic Graph Topology Evolution**: Implementing graph neural network (GNN) message passing over the Cross-Dimension Correlation Graph.

---

## 13. Citation

If you use PDCL in your research, experiments, or publications, please cite:

```bibtex
@article{nandakumar2026pdcl,
  title   = {Parallel Dimensional Core Learning: A Library-Free Graph-Based Multidimensional Deep Learning Architecture with Compiler-Style Parallel Training, Adaptive Pruning, and Graph-Weighted Gradient Aggregation},
  author  = {Nandakumar, Rithan},
  journal = {Department of Artificial Intelligence and Data Science, Anna University},
  year    = {2026},
  url     = {https://github.com/Rithan377/pdcl}
}
```

---

<p align="center">
  <b>Developed with precision by Rithan Nandakumar</b><br>
  Anna University, Chennai, India • <a href="mailto:nsrithan15@gmail.com">nsrithan15@gmail.com</a>
</p>
