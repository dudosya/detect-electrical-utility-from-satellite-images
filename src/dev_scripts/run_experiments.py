"""
Experiment runner for systematic comparison of different configurations.
Uses W&B groups and tags for organization.
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
import yaml


def load_template(template_path: Path) -> dict[str, Any]:
    """Load experiment template configuration."""
    with open(template_path) as f:
        return yaml.safe_load(f)


def generate_experiment_variations(
    template: dict[str, Any],
    variations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate multiple experiment configs from template and variations."""
    experiments = []

    for i, variation in enumerate(variations):
        # Create deep copy of template
        import copy

        experiment = copy.deepcopy(template)

        # Apply variation
        for key_path, value in variation.items():
            # Handle nested keys (e.g., "training.loss_function")
            keys = key_path.split(".")
            current = experiment
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value

        # Set experiment name if not specified
        if "training" in experiment and "experiment_name" in experiment["training"]:
            exp_name = experiment["training"]["experiment_name"]
            if exp_name == "baseline":
                experiment["training"]["experiment_name"] = f"exp_{i:03d}"

        experiments.append(experiment)

    return experiments


def save_experiment_config(config: dict[str, Any], output_path: Path):
    """Save experiment configuration to YAML file."""
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def run_experiment(config_path: Path, use_real_data: bool = False):
    """Run a single experiment using the train command."""
    cmd = [
        sys.executable,
        "-m",
        "src.detect_electrical_utility_from_satellite_images.train",
        "--config",
        str(config_path),
    ]

    if use_real_data:
        cmd.append("--real-data")

    print(f"Running experiment: {config_path.name}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Experiment completed successfully: {config_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed: {config_path.name}")
        print(f"Error: {e.stderr}")
        return False


def create_loss_function_experiments():
    """Create experiment variations for different loss functions."""
    variations = [
        {
            "training.loss_function": "cross_entropy",
            "training.experiment_name": "loss_cross_entropy",
            "wandb.tags": ["loss_comparison", "cross_entropy"],
            "wandb.group": "loss_comparison",
            "wandb.notes": "Cross entropy loss baseline",
        },
        {
            "training.loss_function": "dice",
            "training.experiment_name": "loss_dice",
            "wandb.tags": ["loss_comparison", "dice"],
            "wandb.group": "loss_comparison",
            "wandb.notes": "Dice loss for segmentation",
        },
        {
            "training.loss_function": "focal",
            "training.experiment_name": "loss_focal",
            "wandb.tags": ["loss_comparison", "focal"],
            "wandb.group": "loss_comparison",
            "wandb.notes": "Focal loss for class imbalance",
        },
        {
            "training.loss_function": "combined",
            "training.loss_alpha": 0.5,
            "training.experiment_name": "loss_combined",
            "wandb.tags": ["loss_comparison", "combined"],
            "wandb.group": "loss_comparison",
            "wandb.notes": "Combined CrossEntropy + Dice loss",
        },
    ]
    return variations


def create_encoder_experiments():
    """Create experiment variations for different encoders."""
    variations = [
        {
            "model.encoder_name": "mobilenet_v2",
            "training.experiment_name": "encoder_mobilenet_v2",
            "wandb.tags": ["encoder_comparison", "mobilenet_v2"],
            "wandb.group": "encoder_comparison",
            "wandb.notes": "MobileNetV2 encoder (lightweight)",
        },
        {
            "model.encoder_name": "resnet18",
            "training.experiment_name": "encoder_resnet18",
            "wandb.tags": ["encoder_comparison", "resnet18"],
            "wandb.group": "encoder_comparison",
            "wandb.notes": "ResNet18 encoder",
        },
        {
            "model.encoder_name": "resnet34",
            "training.experiment_name": "encoder_resnet34",
            "wandb.tags": ["encoder_comparison", "resnet34"],
            "wandb.group": "encoder_comparison",
            "wandb.notes": "ResNet34 encoder",
        },
        {
            "model.encoder_name": "efficientnet-b0",
            "training.experiment_name": "encoder_efficientnet_b0",
            "wandb.tags": ["encoder_comparison", "efficientnet-b0"],
            "wandb.group": "encoder_comparison",
            "wandb.notes": "EfficientNet-B0 encoder",
        },
    ]
    return variations


def create_patch_size_experiments():
    """Create experiment variations for different patch sizes."""
    variations = [
        {
            "preprocessing.patch_size": 256,
            "training.experiment_name": "patch_256",
            "wandb.tags": ["patch_size", "256"],
            "wandb.group": "patch_size",
            "wandb.notes": "256x256 patches",
        },
        {
            "preprocessing.patch_size": 512,
            "training.experiment_name": "patch_512",
            "wandb.tags": ["patch_size", "512"],
            "wandb.group": "patch_size",
            "wandb.notes": "512x512 patches",
        },
        {
            "preprocessing.patch_size": 1024,
            "training.experiment_name": "patch_1024",
            "wandb.tags": ["patch_size", "1024"],
            "wandb.group": "patch_size",
            "wandb.notes": "1024x1024 patches",
        },
    ]
    return variations


app = typer.Typer(help="Run systematic experiments for electrical utility detection")


@app.command()
def run_experiments(
    template: Path = typer.Option(
        Path("config_experiment_template.yaml"), help="Path to experiment template"
    ),
    experiment_type: str = typer.Option(
        "loss", help="Type of experiments: loss, encoder, patch_size, all"
    ),
    real_data: bool = typer.Option(
        False, "--real-data", help="Use real data instead of mock"
    ),
    output_dir: Path = typer.Option(
        Path("experiment_configs"), help="Directory to save generated configs"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Generate configs but don't run experiments"
    ),
):
    """Run systematic experiments for comparing different configurations.

    Examples:
        # Generate loss function experiments (dry run)
        uv run run-experiments --dry-run --experiment-type loss

        # Run encoder comparison experiments
        uv run run-experiments --experiment-type encoder

        # Run all experiment types
        uv run run-experiments --experiment-type all

        # Use real data instead of mock
        uv run run-experiments --real-data

        # Custom template and output directory
        uv run run-experiments --template my_template.yaml --output-dir my_experiments
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load template
    print(f"Loading template from {template}")
    template_config = load_template(template)

    # Generate variations based on experiment type
    if experiment_type == "loss":
        variations = create_loss_function_experiments()
    elif experiment_type == "encoder":
        variations = create_encoder_experiments()
    elif experiment_type == "patch_size":
        variations = create_patch_size_experiments()
    elif experiment_type == "all":
        # Combine all experiment types
        variations = (
            create_loss_function_experiments()
            + create_encoder_experiments()
            + create_patch_size_experiments()
        )
    else:
        raise typer.BadParameter(f"Unknown experiment type: {experiment_type}")

    # Generate experiment configs
    experiments = generate_experiment_variations(template_config, variations)

    print(f"Generated {len(experiments)} experiment configurations")

    # Save and optionally run experiments
    success_count = 0
    failure_count = 0

    for i, experiment in enumerate(experiments):
        config_path = output_dir / f"experiment_{i:03d}.yaml"
        save_experiment_config(experiment, config_path)
        print(f"Saved config: {config_path}")

        if not dry_run:
            success = run_experiment(config_path, real_data)
            if success:
                success_count += 1
            else:
                failure_count += 1

            # Small delay between experiments
            if i < len(experiments) - 1:
                print("Waiting 5 seconds before next experiment...")
                time.sleep(5)

    # Summary
    print("\n" + "=" * 50)
    print("EXPERIMENT SUMMARY")
    print("=" * 50)
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {failure_count}")

    if not dry_run:
        print(f"\nConfigs saved to: {output_dir}")
        print("View results in W&B:")
        print(
            "- Use groups to filter experiments (e.g., 'loss_comparison', 'encoder_comparison')"
        )
        print("- Use tags to categorize experiments")
        print("- Compare metrics across runs in W&B dashboard")

    if failure_count > 0:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
