# detect-electrical-utility-from-satellite-images

RA project to detect electrical utility from satellite imagery.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed.
- NVIDIA Drivers supporting **CUDA 12.6**.

## Setup

```bash
uv sync
```

_This installs Python 3.13, CUDA-enabled PyTorch, and all dependencies into a local `.venv`._

## Usage

**Run the main application:**

```bash
uv run main
```

**Verify GPU acceleration:**

```bash
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0)})')"
```

## Development

- **Source Code**: All code is in `src/detect_electrical_utility_from_satellite_images/`
- **Add Dependencies**: `uv add <package_name>` (e.g., `uv add pandas`)
- **Sync Changes**: `uv sync` updates the lockfile and environment.

---

**Author:** dudosya ([kenaykay@gmail.com](mailto:kenaykay@gmail.com))
