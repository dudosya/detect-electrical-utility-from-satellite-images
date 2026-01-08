# GridTracer

A three-stage machine learning pipeline for detecting electrical utility infrastructure (towers, power lines, substations) from satellite imagery. Implementation based on the GridTracer paper architecture.

## Overview

GridTracer converts raw satellite imagery into a geospatial graph representing the power grid:

1. **Stage 1 - Tower Detection**: Faster R-CNN with ResNet50-FPN backbone detects transmission towers
2. **Stage 2 - Line Segmentation**: U-Net semantic segmentation produces power line probability maps
3. **Stage 3 - Graph Inference**: Connects detected towers using segmentation scores and distance constraints

## Requirements

- Python 3.13+
- NVIDIA GPU with CUDA 12.6 support
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

```bash
git clone <repository-url>
cd detect-electrical-utility-from-satellite-images
uv sync
```

This creates a virtual environment with all dependencies including CUDA-enabled PyTorch.

Verify GPU availability:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Project Structure

```
.
├── config.yaml                 # All configurable parameters
├── raw_data/                   # Raw satellite imagery and annotations
│   └── {Region}/               # e.g., NZ_Dunedin/
│       ├── *.jpg               # Satellite image tiles
│       ├── *.csv               # Vertex annotations
│       ├── *.geojson           # Geo-coordinates
│       └── *_multiclass.png    # Segmentation masks
├── preprocessed_data/          # Generated training patches
├── checkpoints/                # Saved model weights
├── inference_results/          # Inference outputs
└── src/                        # Source code
```

## Configuration

All parameters are centralized in `config.yaml`:

```yaml
preprocessing:
  patch_size: 500 # Patch dimensions in pixels

tower_detection:
  confidence_threshold: 0.5 # Minimum detection confidence
  nms_threshold: 0.5 # Non-maximum suppression threshold

line_segmentation:
  line_width_train: 30 # Line width for training masks
  line_width_inference: 9 # Line width for graph inference

graph_inference:
  max_distance_m: 600.0 # Maximum connection distance (meters)
  connectivity_threshold: 0.2 # Minimum score for edge creation

training:
  max_epochs: 50
  batch_size: 5
  initial_lr: 0.003
```

## Usage

### 1. Preprocessing

Convert raw imagery into 500x500 pixel training patches:

```bash
# Preprocess a specific region
uv run main preprocess NZ_Dunedin

# Preprocess all available regions
uv run main preprocess --all
```

### 2. Training

Train the tower detection and line segmentation models:

```bash
# Train tower detector (Stage 1)
uv run main train tower

# Train line segmentor (Stage 2)
uv run main train line

# Train both stages sequentially
uv run main train all

# Training options
uv run main train tower --epochs 100        # Override epochs
uv run main train tower --region NZ_Dunedin # Specific region
uv run main train tower --no-wandb          # Disable W&B logging
uv run main train tower --fast              # Quick test run
```

Training logs metrics to Weights & Biases by default. Disable with `--no-wandb` or set `wandb.enabled: false` in config.

### 3. Evaluation

Evaluate trained models on validation data:

```bash
# Evaluate tower detection
uv run main evaluate tower

# Evaluate line segmentation
uv run main evaluate line

# List available checkpoints
uv run main evaluate tower --list

# Use specific checkpoint
uv run main evaluate tower --prev 1         # Second most recent
uv run main evaluate tower -c path/to/ckpt  # Explicit path

# Save without display
uv run main evaluate tower --no-show
```

Evaluation outputs:

- Metrics: mAP, IoU, Precision, Recall, F1
- Visualization saved next to checkpoint

### 4. Inference

Run the complete 3-stage pipeline on a satellite image:

```bash
# Basic inference (auto-selects latest checkpoints)
uv run main infer path/to/image.jpg

# Specify checkpoints explicitly
uv run main infer image.jpg -t tower.ckpt -l line.ckpt

# Options
uv run main infer image.jpg --no-show       # No interactive display
uv run main infer image.jpg -o results/     # Custom output directory
```

Inference outputs (saved to `inference_results/`):

- `pipeline_summary.png`: 4-panel visualization of each stage
- `graph_overlay.png`: Final graph overlaid on image
- `graph.json`: Machine-readable graph data

### 5. System Information

```bash
uv run main info
```

Displays current configuration and GPU status.

## Output Format

The graph inference produces a JSON file with the following structure:

```json
{
  "nodes": [[x1, y1], [x2, y2], ...],
  "adjacency": [[0, 1, 0], [1, 0, 1], ...],
  "node_scores": [0.95, 0.87, ...],
  "edge_scores": [[0.0, 0.45, 0.0], ...],
  "num_nodes": 3,
  "num_edges": 2,
  "resolution_m_per_px": 0.3
}
```

- `nodes`: Tower centroid coordinates in pixels
- `adjacency`: Binary adjacency matrix
- `node_scores`: Detection confidence per tower
- `edge_scores`: Connectivity score per tower pair

## CLI Reference

```
uv run main [COMMAND] [OPTIONS]

Commands:
  preprocess   Preprocess raw imagery into training patches
  train        Train tower detection or line segmentation models
  evaluate     Evaluate trained models on validation data
  infer        Run full 3-stage inference on an image
  info         Display configuration and system information

Global Options:
  --config, -c PATH    Path to configuration file (default: config.yaml)
  --help               Show help message
```

## Tutorial: Complete Training Pipeline

This section walks through training GridTracer from scratch on a new dataset.

### Step 1: Prepare Raw Data

Place satellite imagery and annotations in the `raw_data/` directory:

```
raw_data/
└── NZ_Dunedin/
    ├── NZ_Dunedin_1.jpg              # Satellite tile (~5000x5000 px)
    ├── NZ_Dunedin_1.csv              # Tower/line annotations
    ├── NZ_Dunedin_1_multiclass.png   # Segmentation mask
    ├── NZ_Dunedin_2.jpg
    ├── NZ_Dunedin_2.csv
    └── ...
```

Required files per tile:

- `.jpg`: RGB satellite image
- `.csv`: Annotations with columns `Object ID`, `Type`, `X`, `Y`
- `_multiclass.png`: Mask with pixel values 1=Tower, 2=OtherTower, 3=Line

### Step 2: Preprocess Data

Generate 500x500 pixel training patches:

```bash
uv run main preprocess NZ_Dunedin
```

Output structure:

```
preprocessed_data/patches/NZ_Dunedin/
├── images/           # 500x500 RGB patches
├── masks/            # Corresponding segmentation masks
└── annotations/      # Tower bounding boxes (JSON)
```

Verify preprocessing:

```bash
ls preprocessed_data/patches/NZ_Dunedin/images | wc -l
```

### Step 3: Configure Training

Edit `config.yaml` for your hardware:

```yaml
training:
  max_epochs: 50 # Increase for better results
  batch_size: 2 # Reduce if GPU OOM
  num_workers: 0 # 0 for Windows, 4+ for Linux

wandb:
  enabled: true # Set false to disable tracking
  project: "gridtracer"
```

### Step 4: Train Tower Detector (Stage 1)

```bash
uv run main train tower
```

Training output:

- Progress bar with loss/mAP metrics
- Checkpoints saved to `checkpoints/tower/{region}/{timestamp}/`
- Metrics logged to W&B (if enabled)

Expected duration: ~1-2 hours for 50 epochs on RTX 3050.

Monitor training:

```bash
# List checkpoints
uv run main evaluate tower --list

# Quick evaluation during training
uv run main evaluate tower --no-show
```

### Step 5: Train Line Segmentor (Stage 2)

```bash
uv run main train line
```

Training output:

- Progress bar with loss/IoU metrics
- Checkpoints saved to `checkpoints/line/{region}/{timestamp}/`

Expected duration: ~1-2 hours for 50 epochs on RTX 3050.

### Step 6: Evaluate Models

```bash
# Tower detection metrics
uv run main evaluate tower
# Output: mAP, mAP@50, Precision, Recall, F1

# Line segmentation metrics
uv run main evaluate line
# Output: IoU, Dice, Precision, Recall, F1
```

Visualization files are saved next to checkpoints.

### Step 7: Run Inference

Test on a preprocessed patch:

```bash
uv run main infer preprocessed_data/patches/NZ_Dunedin/images/NZ_Dunedin_1_000003_x500_y500.png
```

Or on a raw tile (requires trained models):

```bash
uv run main infer raw_data/NZ_Dunedin/NZ_Dunedin_1.jpg --no-show
```

Results saved to `inference_results/{timestamp}/`:

- `pipeline_summary.png`: All 3 stages visualized
- `graph_overlay.png`: Final graph on image
- `graph.json`: Machine-readable output

### Step 8: Iterate

To improve results:

1. **More epochs**: Increase `training.max_epochs` in config
2. **More data**: Add additional regions to `raw_data/`
3. **Lower thresholds**: Reduce `graph_inference.connectivity_threshold` for more connections
4. **Tune confidence**: Adjust `tower_detection.confidence_threshold`

Re-train specific stage:

```bash
uv run main train tower --epochs 100
uv run main train line --epochs 100
```

### Quick Reference

| Task                   | Command                             |
| ---------------------- | ----------------------------------- |
| Preprocess all regions | `uv run main preprocess --all`      |
| Train everything       | `uv run main train all`             |
| Quick test             | `uv run main train tower --fast`    |
| Check GPU              | `uv run main info`                  |
| List checkpoints       | `uv run main evaluate tower --list` |
| Full inference         | `uv run main infer image.jpg`       |

## Development

```bash
# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Lint code
uv run ruff check src/

# Format code
uv run ruff format src/

# Type checking
uv run mypy src/
```

## Dataset

This implementation uses the Electric Transmission Infrastructure Satellite Imagery Dataset:

- Resolution: 0.3 meters per pixel
- Coverage: 264 km² across 7 cities (USA and New Zealand)
- Labels: Towers, Lines, Edge Nodes, Substations

See [here](https://figshare.com/articles/dataset/Electric_Transmission_Infrastructure_Satellite_Imagery_Dataset_for_Computer_Vision/14935434) for details.

## Architecture Details

See [https://arxiv.org/abs/2101.06390](https://arxiv.org/abs/2101.06390) for the complete architecture specification from the GridTracer paper.

## License

See [LICENSE](LICENSE) file.
