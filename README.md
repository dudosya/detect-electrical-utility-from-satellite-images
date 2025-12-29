# Detect Electrical Utility from Satellite Images

A machine learning project for semantic segmentation of electrical utilities in satellite imagery using PyTorch.

## Key Features

- **Modular Configuration**: Clean, type-safe configuration system using Pydantic
- **Multiple Architectures**: Support for UNet, DeepLabV3, and FPN with various backbones
- **Comprehensive Training**: Mixed precision, gradient accumulation, W&B integration
- **Production Ready**: Type hints, comprehensive testing, and code quality tools
- **Flexible Data Pipeline**: Support for both mock data (testing) and real satellite data

## Prerequisites

- [uv](https://docs.astral.sh/uv/) - Fast Python package manager
- NVIDIA Drivers supporting **CUDA 12.6** (for GPU training)
- Python 3.13+

## Installation

```bash
# Install dependencies
uv sync

# Verify installation
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Project Structure

```
detect-electrical-utility-from-satellite-images/
├── src/detect_electrical_utility_from_satellite_images/
│   ├── config/                    # Modular configuration
│   │   ├── base.py               # Project root discovery
│   │   ├── preprocessing.py      # Preprocessing settings
│   │   ├── model.py              # Model architecture
│   │   ├── training.py           # Training hyperparameters
│   │   ├── augmentation.py       # Data augmentation
│   │   ├── metrics.py            # Metrics configuration
│   │   ├── wandb.py              # Weights & Biases
│   │   ├── logging.py            # Logging configuration
│   │   └── app.py               # Main app config
│   ├── dataset.py               # Satellite dataset loader
│   ├── model.py                 # Model creation utilities
│   ├── train.py                 # Training loop
│   ├── preprocess.py            # Image preprocessing
│   ├── losses.py                # Loss functions
│   ├── metrics.py               # Evaluation metrics
│   ├── main.py                  # CLI entry point
│   └── utils/                   # Utility functions
│       ├── file_utils.py        # File operations
│       ├── logging_config.py    # Logging setup
│       └── split_utils.py       # Image-level data splitting
├── tests/                       # Test suite
├── config.yaml                  # Main configuration
├── config_experiment_template.yaml  # Experiment template
└── experiment_configs/          # Experiment configurations
```

## Quick Start

### 1. Preprocess Data

```bash
# Create patches from satellite images
uv run main preprocess
```

### 2. Train a Model

```bash
# Train with mock data (for testing)
uv run main train

# Train with real data
uv run main train --real-data

# Train with debug logging
uv run main --config config train --real-data --log-level debug
```

### 3. Alternative Entry Points

```bash
# Use the standalone train command
uv run train --config config_experiment --real-data

# Test dataset loading
uv run main test-dataset --config config

# Run experiments
uv run run-experiments --config-dir experiment_configs
```

## Configuration

The project uses a modular YAML-based configuration system:

### Basic Configuration (`config.yaml`)

```yaml
preprocessing:
  patch_size: 256
  background_fraction: 0.1

paths:
  output_dir: "output"
  data_dir: "data"
  logging_dir_name: "logs"

model:
  architecture: "unet"
  encoder_name: "resnet18"
  encoder_weights: "imagenet"
  in_channels: 3
  classes: 5
  learning_rate: 0.001
  batch_size: 8
  epochs: 10
  device: "cuda"

training:
  experiment_name: "baseline"
  seed: 42
  mixed_precision: true
  loss_function: "cross_entropy"
```

## Development

### Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_config.py -v
```

### Code Quality

```bash
# Lint with ruff
uv run ruff check src/

# Format code
uv run ruff format src/

# Type checking (requires mypy)
uv add mypy --dev
uv run mypy src/
```

### Adding Dependencies

```bash
# Add production dependency
uv add package_name

# Add development dependency
uv add package_name --dev
```

## Model Architectures

Supported architectures:

- **UNet**: Classic encoder-decoder architecture
- **DeepLabV3**: Atrous spatial pyramid pooling
- **FPN**: Feature Pyramid Network

Supported backbones:

- ResNet (18, 34, 50)
- EfficientNet-B0
- MobileNetV2

## Training Features

- **Mixed Precision Training**: Faster training with FP16
- **Gradient Accumulation**: Train with larger effective batch sizes
- **Learning Rate Schedulers**: Cosine, ReduceLROnPlateau, Step
- **Early Stopping**: Prevent overfitting
- **Class Weighting**: Handle imbalanced datasets (calculated from training data only)
- **Data Leakage Prevention**: Image-level train/val split ensures patches from same image don't appear in both sets
- **W&B Integration**: Experiment tracking and visualization

## Evaluation Metrics

- **IoU** (Intersection over Union)
- **Dice Coefficient**
- **Accuracy, Precision, Recall, F1-Score**
- **Per-class metrics**

## Troubleshooting

### Common Issues

1. **CUDA not available**:

   ```bash
   # Check CUDA installation
   uv run python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Missing dependencies**:

   ```bash
   # Re-sync dependencies
   uv sync
   ```

3. **Configuration errors**:
   ```bash
   # Validate configuration
   uv run python -c "from detect_electrical_utility_from_satellite_images.config import AppConfig; import yaml; cfg = AppConfig(**yaml.safe_load(open('config.yaml'))); print('Config valid')"
   ```

## Development Guidelines

- Use type hints for all function signatures
- Write docstrings following Google style
- Add tests for new features
- Run `ruff check` and `ruff format` before committing
- Update documentation when adding features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- **dudosya** - [kenaykay@gmail.com](mailto:kenaykay@gmail.com)
