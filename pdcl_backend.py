"""
PDCL Compute Backend — GPU/CPU Auto-Selector
==============================================
Automatically uses CuPy (NVIDIA GPU via CUDA) if available,
falls back to NumPy (CPU) otherwise.

CuPy is a drop-in NumPy replacement that executes on CUDA GPUs.
All matrix operations (matmul, attention, softmax, etc.) run on GPU
while maintaining the exact same API.

Usage in all PDCL modules:
    from pdcl_backend import xp as np, to_cpu, to_device, GPU_AVAILABLE
"""

import numpy as _numpy

try:
    import cupy as xp
    GPU_AVAILABLE = True

    # Print GPU info
    _dev = xp.cuda.Device()
    _mem = _dev.mem_info
    _total_gb = _mem[1] / (1024 ** 3)
    _free_gb = _mem[0] / (1024 ** 3)
    print(f"\n{'='*60}")
    print(f"  GPU BACKEND ACTIVE")
    print(f"  Device    : {_dev.id} — {xp.cuda.runtime.getDeviceProperties(_dev.id)['name'].decode()}")
    print(f"  VRAM      : {_free_gb:.1f} GB free / {_total_gb:.1f} GB total")
    print(f"  CuPy ver  : {xp.__version__}")
    print(f"{'='*60}\n")

except Exception as e:
    raise RuntimeError(
        f"GPU training is strictly required, but CUDA/CuPy initialization failed: {e}\n"
        "Please check your NVIDIA drivers and CUDA environment (e.g. export LD_LIBRARY_PATH=\"\")."
    ) from e


def to_cpu(arr):
    """Convert array to CPU (numpy) for file I/O, metrics, and printing."""
    if GPU_AVAILABLE and hasattr(arr, 'get'):
        return arr.get()
    return _numpy.asarray(arr)


def to_device(arr):
    """Convert numpy/list array to device (GPU if available)."""
    if GPU_AVAILABLE:
        return xp.asarray(arr)
    return _numpy.asarray(arr)


# Re-export numpy for explicit CPU-only operations (file I/O, tokenizer)
cpu_np = _numpy
CPU_NP = _numpy  # Alias for backward compatibility
