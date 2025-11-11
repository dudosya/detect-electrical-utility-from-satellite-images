# detect-electrical-utility-from-satellite-images

## Installation Guide

### Prerequisites

1.  Python 3.13 (or compatible version).
2.  NVIDIA Drivers supporting CUDA 12.6 or newer.

### 1. Install Standard Dependencies

First, install all base libraries (e.g., NumPy, Matplotlib) from the locked `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 2. Install PyTorch and Torchvision (CUDA 12.6)

Since PyTorch and Torchvision require a specific CUDA build, install them separately using the custom index:

```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu126
```

---

### For Developers: Updating Standard Dependencies

If you change dependencies in `requirements.in` (excluding `torch` and `torchvision`), regenerate the lock file:

```bash
pip-compile requirements.in
```