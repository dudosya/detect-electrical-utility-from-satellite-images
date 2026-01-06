"""GridTracer CLI - Main entry point for the application."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional

import typer

from detect_electrical_utility_from_satellite_images.config import load_config
from detect_electrical_utility_from_satellite_images.logging_config import (
    get_logger,
    setup_logging,
)

if TYPE_CHECKING:
    from detect_electrical_utility_from_satellite_images.config import Config

app = typer.Typer(
    name="gridtracer",
    help="GridTracer: Detect electrical utility infrastructure from satellite imagery.",
    add_completion=False,
)


def _init_app(config_path: Path | None = None) -> None:
    """Initialize the application with config and logging.

    Args:
        config_path: Optional path to config file.
    """
    config = load_config(config_path)
    setup_logging(
        logger_level=config.logging.logger_lvl,
        console_level=config.logging.console_handler_lvl,
        file_level=config.logging.file_handler_lvl,
        log_dir=Path(config.paths.logging_dir_name),
    )


@app.callback()
def main(
    config: Annotated[
        Optional[Path],
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration file",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
) -> None:
    """GridTracer: ML pipeline for power grid detection from satellite imagery."""
    _init_app(config)


@app.command()
def preprocess(
    region: Annotated[
        Optional[str],
        typer.Argument(help="Region name to preprocess (e.g., 'NZ_Dunedin')"),
    ] = None,
    all_regions: Annotated[
        bool,
        typer.Option("--all", "-a", help="Process all available regions"),
    ] = False,
) -> None:
    """Preprocess raw satellite imagery into training patches."""
    from detect_electrical_utility_from_satellite_images.preprocessing import (
        run_preprocessing,
    )

    log = get_logger()
    config = load_config()

    if all_regions:
        log.info("preprocessing_all_regions")
        run_preprocessing(config, region=None)
    elif region:
        log.info("preprocessing_region", region=region)
        run_preprocessing(config, region=region)
    else:
        typer.echo("Error: Specify a region name or use --all flag")
        raise typer.Exit(code=1)


@app.command()
def train(
    stage: Annotated[
        str,
        typer.Argument(help="Training stage: 'tower', 'line', or 'all'"),
    ] = "tower",
    region: Annotated[
        str,
        typer.Option("--region", "-r", help="Region to train on"),
    ] = "NZ_Dunedin",
    epochs: Annotated[
        Optional[int],
        typer.Option("--epochs", "-e", help="Override max_epochs from config"),
    ] = None,
    no_wandb: Annotated[
        bool,
        typer.Option("--no-wandb", help="Disable W&B logging (overrides config)"),
    ] = False,
    fast_dev_run: Annotated[
        bool,
        typer.Option("--fast", help="Run a quick test with 1 batch"),
    ] = False,
) -> None:
    """Train the GridTracer models."""
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
    from pytorch_lightning.loggers import WandbLogger

    from detect_electrical_utility_from_satellite_images.training import (
        TowerDetectionDataModule,
        TowerDetectorModule,
        set_seed,
    )

    log = get_logger()
    config = load_config()

    # Set tensor core precision for faster training on compatible GPUs
    torch.set_float32_matmul_precision(config.training.tensor_core_precision)

    # Set seed for reproducibility
    set_seed(config.training.seed)

    # Use epochs from CLI if provided, otherwise from config
    max_epochs = epochs if epochs is not None else config.training.max_epochs

    log.info(
        "training_start",
        stage=stage,
        region=region,
        seed=config.training.seed,
        max_epochs=max_epochs,
    )

    if stage not in ("tower", "line", "all"):
        typer.echo(f"Error: Unknown stage '{stage}'. Use 'tower', 'line', or 'all'")
        raise typer.Exit(code=1)

    if stage in ("tower", "all"):
        _train_tower_detector(
            config=config,
            region=region,
            max_epochs=max_epochs,
            use_wandb=not no_wandb,
            fast_dev_run=fast_dev_run,
        )

    if stage in ("line", "all"):
        typer.echo("Line segmentation training not yet implemented")


def _train_tower_detector(
    config: "Config",
    region: str,
    max_epochs: int,
    use_wandb: bool,
    fast_dev_run: bool,
) -> None:
    """Train the tower detection model.

    Args:
        config: Configuration instance.
        region: Region to train on.
        max_epochs: Maximum training epochs.
        use_wandb: Whether to use W&B logging.
        fast_dev_run: Whether to do a quick test run.
    """
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
    from pytorch_lightning.loggers import WandbLogger

    from detect_electrical_utility_from_satellite_images.training import (
        TowerDetectionDataModule,
        TowerDetectorModule,
    )

    log = get_logger()

    # Setup paths
    patches_dir = Path(config.paths.output_dir) / region
    if not patches_dir.exists():
        log.error("patches_not_found", path=str(patches_dir))
        typer.echo(f"Error: Preprocessed patches not found at {patches_dir}")
        typer.echo("Run 'uv run main preprocess <region>' first")
        raise typer.Exit(code=1)

    # Create data module
    data_module = TowerDetectionDataModule(
        config=config,
        patches_dir=patches_dir,
        val_split=config.training.val_split,
        num_workers=config.training.num_workers,
    )

    # Create model
    model = TowerDetectorModule(config=config)

    # Create run-specific checkpoint directory with timestamp
    from datetime import datetime
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(f"checkpoints/tower/{region}/{run_timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup callbacks - checkpoint saves best and last ONLY at end of training
    callbacks: list[pl.Callback] = [
        RichProgressBar(),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best",
            monitor="val/mAP",
            mode="max",
            save_top_k=1,
            save_last=True,  # saves as last.ckpt
            verbose=True,
            every_n_epochs=max_epochs,  # Only save at the very end
        ),
    ]

    # Setup W&B logger from config
    logger: pl.loggers.Logger | bool = False
    wandb_cfg = config.wandb
    if use_wandb and wandb_cfg.enabled:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        run_name = f"tower-{region}-{timestamp}"
        
        logger = WandbLogger(
            project=wandb_cfg.project,
            entity=wandb_cfg.entity,
            name=run_name,
            tags=wandb_cfg.tags + [region, "tower-detection"],
            notes=wandb_cfg.notes,
            offline=wandb_cfg.offline,
            config={
                "stage": "tower_detection",
                "region": region,
                "patch_size": config.preprocessing.patch_size,
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.initial_lr,
                "anchor_areas": config.tower_detection.anchor_areas,
                "anchor_ratios": config.tower_detection.anchor_ratios,
                "confidence_threshold": config.tower_detection.confidence_threshold,
                "nms_threshold": config.tower_detection.nms_threshold,
            },
        )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=fast_dev_run,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    log.info("starting_tower_training", patches_dir=str(patches_dir))

    # Train
    trainer.fit(model, data_module)

    log.info("tower_training_complete", checkpoint_dir=str(checkpoint_dir))


def _find_checkpoint(
    region: str,
    nth_last: int = 0,
    use_last: bool = False,
) -> Path | None:
    """Find checkpoint file in the checkpoints directory.

    Args:
        region: Region name to look for checkpoints.
        nth_last: 0 = most recent run, 1 = second most recent run, etc.
        use_last: If True, use 'last.ckpt' instead of 'best.ckpt'.

    Returns:
        Path to checkpoint or None if not found.
    """
    checkpoint_base = Path(f"checkpoints/tower/{region}")
    if not checkpoint_base.exists():
        return None

    # Find all run directories (timestamped folders)
    run_dirs = [d for d in checkpoint_base.iterdir() if d.is_dir()]
    if not run_dirs:
        return None

    # Sort by name (timestamp) descending - most recent first
    run_dirs.sort(key=lambda d: d.name, reverse=True)

    if nth_last >= len(run_dirs):
        nth_last = len(run_dirs) - 1  # Use oldest if out of range

    target_dir = run_dirs[nth_last]
    
    if use_last:
        ckpt = target_dir / "last.ckpt"
    else:
        ckpt = target_dir / "best.ckpt"
    
    return ckpt if ckpt.exists() else None


def _list_checkpoints(region: str) -> list[tuple[Path, str]]:
    """List all available checkpoints with metadata.

    Args:
        region: Region name.

    Returns:
        List of (path, info_string) tuples.
    """
    from datetime import datetime

    checkpoint_base = Path(f"checkpoints/tower/{region}")
    if not checkpoint_base.exists():
        return []

    # Find all run directories
    run_dirs = [d for d in checkpoint_base.iterdir() if d.is_dir()]
    run_dirs.sort(key=lambda d: d.name, reverse=True)

    result = []
    for i, run_dir in enumerate(run_dirs):
        best_ckpt = run_dir / "best.ckpt"
        last_ckpt = run_dir / "last.ckpt"
        
        marker = " (latest)" if i == 0 else f" (--prev {i})"
        
        if best_ckpt.exists():
            size_mb = best_ckpt.stat().st_size / (1024 * 1024)
            info = f"{run_dir.name}/best.ckpt | {size_mb:.1f}MB{marker}"
            result.append((best_ckpt, info))
        
        if last_ckpt.exists():
            size_mb = last_ckpt.stat().st_size / (1024 * 1024)
            info = f"{run_dir.name}/last.ckpt | {size_mb:.1f}MB{marker} --last"
            result.append((last_ckpt, info))

    return result


@app.command()
def evaluate(
    checkpoint: Annotated[
        Optional[Path],
        typer.Argument(help="Path to checkpoint (optional, defaults to latest)"),
    ] = None,
    region: Annotated[
        str,
        typer.Option("--region", "-r", help="Region to evaluate on"),
    ] = "NZ_Dunedin",
    prev: Annotated[
        int,
        typer.Option("--prev", "-p", help="Use Nth previous checkpoint (0=latest)"),
    ] = 0,
    last: Annotated[
        bool,
        typer.Option("--last", "-l", help="Use last.ckpt (last saved, not best)"),
    ] = False,
    list_all: Annotated[
        bool,
        typer.Option("--list", help="List all available checkpoints"),
    ] = False,
    show: Annotated[
        bool,
        typer.Option("--show/--no-show", "-s", help="Show visualizations interactively"),
    ] = True,
    output_dir: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Save visualizations to this directory"),
    ] = None,
    num_samples: Annotated[
        int,
        typer.Option("--samples", "-n", help="Number of samples to visualize"),
    ] = 16,
) -> None:
    """Evaluate a trained model and visualize predictions.

    By default, evaluates the most recent checkpoint for the region.
    Use --prev N to evaluate the Nth previous checkpoint.
    Use --list to see all available checkpoints.
    """
    import torch
    import matplotlib.pyplot as plt

    from detect_electrical_utility_from_satellite_images.training import (
        TowerDetectionDataModule,
        TowerDetectorModule,
    )
    from detect_electrical_utility_from_satellite_images.evaluation import (
        evaluate_model_on_dataset,
    )

    log = get_logger()
    config = load_config()

    # List checkpoints if requested
    if list_all:
        checkpoints = _list_checkpoints(region)
        if not checkpoints:
            typer.echo(f"No checkpoints found for region '{region}'")
            typer.echo(f"Run 'uv run main train tower -r {region}' first")
            raise typer.Exit(code=1)

        typer.echo(f"\n=== Checkpoints for {region} ===\n")
        for _, info in checkpoints:
            typer.echo(f"  {info}")
        typer.echo("")
        return

    # Resolve checkpoint path
    if checkpoint is None:
        checkpoint = _find_checkpoint(region, nth_last=prev, use_last=last)
        if checkpoint is None:
            typer.echo(f"No checkpoints found for region '{region}'")
            typer.echo(f"Run 'uv run main train tower -r {region}' first")
            raise typer.Exit(code=1)
        typer.echo(f"Auto-selected checkpoint: {checkpoint}")
    elif not checkpoint.exists():
        log.error("checkpoint_not_found", path=str(checkpoint))
        typer.echo(f"Error: Checkpoint not found at {checkpoint}")
        raise typer.Exit(code=1)

    log.info("evaluation_start", checkpoint=str(checkpoint), region=region)

    # Setup paths
    patches_dir = Path(config.paths.output_dir) / region
    if not patches_dir.exists():
        log.error("patches_not_found", path=str(patches_dir))
        typer.echo(f"Error: Preprocessed patches not found at {patches_dir}")
        raise typer.Exit(code=1)

    # Load model from checkpoint
    typer.echo(f"Loading model from {checkpoint}...")
    model = TowerDetectorModule.load_from_checkpoint(
        str(checkpoint),
        config=config,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create data module
    data_module = TowerDetectionDataModule(
        config=config,
        patches_dir=patches_dir,
        val_split=config.training.val_split,
        num_workers=config.training.num_workers,
    )
    data_module.setup(stage="validate")

    # Evaluate
    typer.echo("Running evaluation...")
    results = evaluate_model_on_dataset(
        model=model.model,
        dataloader=data_module.val_dataloader(),
        device=device,
        score_threshold=config.tower_detection.confidence_threshold,
        max_vis_samples=num_samples,
    )

    # Extract metrics
    metrics = results["metrics"]
    
    # Print results
    typer.echo("\n=== Evaluation Results ===\n")
    typer.echo(f"mAP:        {metrics['mAP']:.4f}")
    typer.echo(f"mAP@50:     {metrics['mAP_50']:.4f}")
    typer.echo(f"mAP@75:     {metrics['mAP_75']:.4f}")
    typer.echo(f"Precision:  {metrics['precision']:.4f}")
    typer.echo(f"Recall:     {metrics['recall']:.4f}")
    
    # Compute F1
    precision = metrics['precision']
    recall = metrics['recall']
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    typer.echo(f"F1 Score:   {f1:.4f}")

    log.info(
        "metrics_summary",
        mAP=metrics["mAP"],
        mAP_50=metrics["mAP_50"],
        precision=metrics["precision"],
        recall=metrics["recall"],
    )

    # Get visualization figure from results
    fig = results["visualization"]

    # Always save visualization next to checkpoint
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine save directory - use output_dir if provided, else checkpoint's directory
    if output_dir:
        save_dir = Path(output_dir)
    else:
        save_dir = checkpoint.parent  # Save next to the checkpoint
    
    save_dir.mkdir(parents=True, exist_ok=True)
    viz_filename = f"eval_{timestamp}_mAP{metrics['mAP']:.3f}.png"
    viz_path = save_dir / viz_filename
    fig.savefig(viz_path, dpi=150, bbox_inches="tight")
    typer.echo(f"\nVisualization saved: {viz_path}")

    if show:
        typer.echo("Displaying visualization... (close window to continue)")
        plt.show()
    else:
        plt.close(fig)


@app.command()
def infer(
    image: Annotated[
        Path,
        typer.Argument(help="Path to satellite image for inference"),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output path for results"),
    ] = None,
) -> None:
    """Run inference on a satellite image."""
    log = get_logger()

    if not image.exists():
        log.error("image_not_found", path=str(image))
        raise typer.Exit(code=1)

    log.info("inference_start", image=str(image))
    typer.echo(f"Running inference on: {image} (not yet implemented)")


@app.command()
def info() -> None:
    """Display configuration and system information."""
    import torch

    config = load_config()

    typer.echo("\n=== GridTracer Configuration ===\n")
    typer.echo(f"Patch size: {config.preprocessing.patch_size}px")
    typer.echo(f"Data directory: {config.paths.data_dir}")
    typer.echo(f"Output directory: {config.paths.output_dir}")
    typer.echo(f"Training iterations: {config.training.iterations}")
    typer.echo(f"Batch size: {config.training.batch_size}")
    typer.echo(f"Learning rate: {config.training.initial_lr}")

    typer.echo("\n=== System Info ===\n")
    typer.echo(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        typer.echo(f"GPU: {torch.cuda.get_device_name(0)}")
        typer.echo(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )


if __name__ == "__main__":
    app()
