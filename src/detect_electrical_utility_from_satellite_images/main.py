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
        Optional[str],
        typer.Option("--region", "-r", help="Region to train on (default: all preprocessed regions)"),
    ] = None,
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

    # Discover regions to train on
    regions = _discover_regions(config, region)
    if not regions:
        typer.echo("Error: No preprocessed regions found. Run 'uv run main preprocess' first.")
        raise typer.Exit(code=1)

    log.info(
        "training_start",
        stage=stage,
        regions=regions,
        seed=config.training.seed,
        max_epochs=max_epochs,
    )

    if stage not in ("tower", "line", "all"):
        typer.echo(f"Error: Unknown stage '{stage}'. Use 'tower', 'line', or 'all'")
        raise typer.Exit(code=1)

    if stage in ("tower", "all"):
        _train_tower_detector(
            config=config,
            regions=regions,
            max_epochs=max_epochs,
            use_wandb=not no_wandb,
            fast_dev_run=fast_dev_run,
        )

    if stage in ("line", "all"):
        _train_line_segmentor(
            config=config,
            regions=regions,
            max_epochs=max_epochs,
            use_wandb=not no_wandb,
            fast_dev_run=fast_dev_run,
        )


def _discover_regions(config: "Config", region: str | None) -> list[str]:
    """Discover available preprocessed regions.

    Args:
        config: Configuration instance.
        region: Specific region name, or None to discover all.

    Returns:
        List of region names.
    """
    output_dir = Path(config.paths.output_dir)

    if region:
        # Specific region requested
        region_path = output_dir / region
        if region_path.exists() and (region_path / "images").exists():
            return [region]
        return []

    # Auto-discover all preprocessed regions
    if not output_dir.exists():
        return []

    regions = []
    for path in sorted(output_dir.iterdir()):
        if path.is_dir() and (path / "images").exists():
            regions.append(path.name)

    return regions


def _train_tower_detector(
    config: "Config",
    regions: list[str],
    max_epochs: int,
    use_wandb: bool,
    fast_dev_run: bool,
) -> None:
    """Train the tower detection model.

    Args:
        config: Configuration instance.
        regions: List of regions to train on.
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

    # Collect patches directories for all regions
    patches_dirs = []
    for region in regions:
        patches_dir = Path(config.paths.output_dir) / region
        if patches_dir.exists():
            patches_dirs.append(patches_dir)
            log.info("region_added", region=region, path=str(patches_dir))
        else:
            log.warning("region_not_found", region=region)

    if not patches_dirs:
        typer.echo("Error: No valid preprocessed regions found")
        raise typer.Exit(code=1)

    # Create data module with all regions
    data_module = TowerDetectionDataModule(
        config=config,
        patches_dirs=patches_dirs,
        val_split=config.training.val_split,
        num_workers=config.training.num_workers,
    )

    # Create model
    model = TowerDetectorModule(config=config)

    # Create run-specific checkpoint directory with timestamp
    from datetime import datetime
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use 'combined' for multi-region or single region name
    region_label = regions[0] if len(regions) == 1 else "combined"
    checkpoint_dir = Path(f"checkpoints/tower/{region_label}/{run_timestamp}")
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

    # Setup W&B logger from config (separate project for tower detection)
    logger: pl.loggers.Logger | bool = False
    wandb_cfg = config.wandb
    if use_wandb and wandb_cfg.enabled:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        run_name = f"tower-{region_label}-{timestamp}"
        
        logger = WandbLogger(
            project=f"{wandb_cfg.project}-tower",
            entity=wandb_cfg.entity,
            name=run_name,
            tags=wandb_cfg.tags + regions + ["tower-detection"],
            notes=wandb_cfg.notes,
            offline=wandb_cfg.offline,
            config={
                "stage": "tower_detection",
                "regions": regions,
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

    log.info("starting_tower_training", regions=regions, num_regions=len(patches_dirs))

    # Train
    trainer.fit(model, data_module)

    log.info("tower_training_complete", checkpoint_dir=str(checkpoint_dir))


def _train_line_segmentor(
    config: "Config",
    regions: list[str],
    max_epochs: int,
    use_wandb: bool,
    fast_dev_run: bool,
) -> None:
    """Train the line segmentation model (U-Net).

    Args:
        config: Configuration instance.
        regions: List of regions to train on.
        max_epochs: Maximum training epochs.
        use_wandb: Whether to use W&B logging.
        fast_dev_run: Whether to do a quick test run.
    """
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
    from pytorch_lightning.loggers import WandbLogger

    from detect_electrical_utility_from_satellite_images.training.line_training import (
        LineSegmentationDataModule,
        LineSegmentorModule,
    )

    log = get_logger()

    # Collect patches directories for all regions
    patches_dirs = []
    for region in regions:
        patches_dir = Path(config.paths.output_dir) / region
        if patches_dir.exists():
            patches_dirs.append(patches_dir)
            log.info("region_added", region=region, path=str(patches_dir))
        else:
            log.warning("region_not_found", region=region)

    if not patches_dirs:
        typer.echo("Error: No valid preprocessed regions found")
        raise typer.Exit(code=1)

    # Create data module with all regions
    data_module = LineSegmentationDataModule(
        config=config,
        patches_dirs=patches_dirs,
        val_split=config.training.val_split,
        num_workers=config.training.num_workers,
    )

    # Create model
    model = LineSegmentorModule(config=config)

    # Create run-specific checkpoint directory with timestamp
    from datetime import datetime
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    region_label = regions[0] if len(regions) == 1 else "combined"
    checkpoint_dir = Path(f"checkpoints/line/{region_label}/{run_timestamp}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup callbacks - checkpoint saves best and last ONLY at end of training
    callbacks: list[pl.Callback] = [
        RichProgressBar(),
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best",
            monitor="val/IoU",
            mode="max",
            save_top_k=1,
            save_last=True,
            verbose=True,
            every_n_epochs=max_epochs,
        ),
    ]

    # Setup W&B logger from config (separate project for line segmentation)
    logger: pl.loggers.Logger | bool = False
    wandb_cfg = config.wandb
    if use_wandb and wandb_cfg.enabled:
        timestamp = datetime.now().strftime("%m%d_%H%M")
        run_name = f"line-{region_label}-{timestamp}"
        
        logger = WandbLogger(
            project=f"{wandb_cfg.project}-line",
            entity=wandb_cfg.entity,
            name=run_name,
            tags=wandb_cfg.tags + regions + ["line-segmentation"],
            notes=wandb_cfg.notes,
            offline=wandb_cfg.offline,
            config={
                "stage": "line_segmentation",
                "regions": regions,
                "patch_size": config.preprocessing.patch_size,
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.initial_lr,
                "line_width_train": config.line_segmentation.line_width_train,
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

    log.info("starting_line_training", regions=regions, num_regions=len(patches_dirs))

    # Train
    trainer.fit(model, data_module)

    log.info("line_training_complete", checkpoint_dir=str(checkpoint_dir))


def _find_checkpoint(
    stage: str,
    region: str,
    nth_last: int = 0,
    use_last: bool = False,
) -> Path | None:
    """Find checkpoint file in the checkpoints directory.

    Args:
        stage: Model stage ('tower' or 'line').
        region: Region name to look for checkpoints.
        nth_last: 0 = most recent run, 1 = second most recent run, etc.
        use_last: If True, use 'last.ckpt' instead of 'best.ckpt'.

    Returns:
        Path to checkpoint or None if not found.
    """
    checkpoint_base = Path(f"checkpoints/{stage}/{region}")
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
        if ckpt.exists():
            return ckpt
    else:
        # Try best.ckpt first, fall back to last.ckpt
        best_ckpt = target_dir / "best.ckpt"
        if best_ckpt.exists():
            return best_ckpt
        last_ckpt = target_dir / "last.ckpt"
        if last_ckpt.exists():
            return last_ckpt
    
    return None


def _list_checkpoints(stage: str, region: str) -> list[tuple[Path, str]]:
    """List all available checkpoints with metadata.

    Args:
        stage: Model stage ('tower' or 'line').
        region: Region name.

    Returns:
        List of (path, info_string) tuples.
    """
    from datetime import datetime

    checkpoint_base = Path(f"checkpoints/{stage}/{region}")
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
    stage: Annotated[
        str,
        typer.Argument(help="Model stage to evaluate: 'tower' or 'line'"),
    ] = "tower",
    checkpoint: Annotated[
        Optional[Path],
        typer.Option("--checkpoint", "-c", help="Path to checkpoint (defaults to latest)"),
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

    Examples:
        uv run main evaluate tower                  # Evaluate tower detection
        uv run main evaluate line                   # Evaluate line segmentation
        uv run main evaluate tower --list           # List tower checkpoints
    """
    import torch
    import matplotlib.pyplot as plt

    log = get_logger()
    config = load_config()

    if stage not in ("tower", "line"):
        typer.echo(f"Error: Unknown stage '{stage}'. Use 'tower' or 'line'")
        raise typer.Exit(code=1)

    # List checkpoints if requested
    if list_all:
        checkpoints = _list_checkpoints(stage, region)
        if not checkpoints:
            typer.echo(f"No {stage} checkpoints found for region '{region}'")
            typer.echo(f"Run 'uv run main train {stage} -r {region}' first")
            raise typer.Exit(code=1)

        typer.echo(f"\n=== {stage.title()} Checkpoints for {region} ===\n")
        for _, info in checkpoints:
            typer.echo(f"  {info}")
        typer.echo("")
        return

    # Resolve checkpoint path
    if checkpoint is None:
        checkpoint = _find_checkpoint(stage, region, nth_last=prev, use_last=last)
        if checkpoint is None:
            typer.echo(f"No {stage} checkpoints found for region '{region}'")
            typer.echo(f"Run 'uv run main train {stage} -r {region}' first")
            raise typer.Exit(code=1)
        typer.echo(f"Auto-selected checkpoint: {checkpoint}")
    elif not checkpoint.exists():
        log.error("checkpoint_not_found", path=str(checkpoint))
        typer.echo(f"Error: Checkpoint not found at {checkpoint}")
        raise typer.Exit(code=1)

    log.info("evaluation_start", stage=stage, checkpoint=str(checkpoint), region=region)

    # Setup paths
    patches_dir = Path(config.paths.output_dir) / region
    if not patches_dir.exists():
        log.error("patches_not_found", path=str(patches_dir))
        typer.echo(f"Error: Preprocessed patches not found at {patches_dir}")
        raise typer.Exit(code=1)

    # Route to appropriate evaluation function
    if stage == "tower":
        _evaluate_tower(
            config, checkpoint, patches_dir, num_samples, show, output_dir
        )
    else:
        _evaluate_line(
            config, checkpoint, patches_dir, num_samples, show, output_dir
        )


def _evaluate_tower(
    config: "Config",
    checkpoint: Path,
    patches_dir: Path,
    num_samples: int,
    show: bool,
    output_dir: Path | None,
) -> None:
    """Evaluate tower detection model."""
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
        patches_dirs=patches_dir,
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
    typer.echo("\n=== Tower Detection Results ===\n")
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
        "tower_metrics",
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
        save_dir = checkpoint.parent
    
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


def _evaluate_line(
    config: "Config",
    checkpoint: Path,
    patches_dir: Path,
    num_samples: int,
    show: bool,
    output_dir: Path | None,
) -> None:
    """Evaluate line segmentation model."""
    import torch
    import matplotlib.pyplot as plt

    from detect_electrical_utility_from_satellite_images.training.line_training import (
        LineSegmentationDataModule,
        LineSegmentorModule,
    )
    from detect_electrical_utility_from_satellite_images.evaluation.line_eval import (
        evaluate_line_model_on_dataset,
    )

    log = get_logger()

    # Load model from checkpoint
    typer.echo(f"Loading model from {checkpoint}...")
    model = LineSegmentorModule.load_from_checkpoint(
        str(checkpoint),
        config=config,
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Create data module
    data_module = LineSegmentationDataModule(
        config=config,
        patches_dirs=patches_dir,
        val_split=config.training.val_split,
        num_workers=config.training.num_workers,
    )
    data_module.setup(stage="validate")

    # Evaluate
    typer.echo("Running evaluation...")
    results = evaluate_line_model_on_dataset(
        model=model.model,
        dataloader=data_module.val_dataloader(),
        device=device,
        threshold=0.5,
        max_vis_samples=num_samples,
    )

    # Extract metrics
    metrics = results["metrics"]
    
    # Print results
    typer.echo("\n=== Line Segmentation Results ===\n")
    typer.echo(f"IoU:        {metrics['IoU']:.4f}")
    typer.echo(f"Dice:       {metrics['Dice']:.4f}")
    typer.echo(f"Precision:  {metrics['Precision']:.4f}")
    typer.echo(f"Recall:     {metrics['Recall']:.4f}")
    typer.echo(f"F1 Score:   {metrics['F1']:.4f}")

    log.info(
        "line_metrics",
        IoU=metrics["IoU"],
        Dice=metrics["Dice"],
        Precision=metrics["Precision"],
        Recall=metrics["Recall"],
    )

    # Get visualization figure from results
    fig = results["visualization"]

    # Always save visualization next to checkpoint
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine save directory
    if output_dir:
        save_dir = Path(output_dir)
    else:
        save_dir = checkpoint.parent
    
    save_dir.mkdir(parents=True, exist_ok=True)
    viz_filename = f"eval_{timestamp}_IoU{metrics['IoU']:.3f}.png"
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
    tower_checkpoint: Annotated[
        Optional[Path],
        typer.Option("--tower-ckpt", "-t", help="Tower detection checkpoint"),
    ] = None,
    line_checkpoint: Annotated[
        Optional[Path],
        typer.Option("--line-ckpt", "-l", help="Line segmentation checkpoint"),
    ] = None,
    region: Annotated[
        str,
        typer.Option("--region", "-r", help="Region name for auto-selecting checkpoints"),
    ] = "NZ_Dunedin",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory for results"),
    ] = None,
    show: Annotated[
        bool,
        typer.Option("--show/--no-show", "-s", help="Show visualization interactively"),
    ] = True,
    save_graph: Annotated[
        bool,
        typer.Option("--save-graph/--no-save-graph", help="Save graph as JSON"),
    ] = True,
) -> None:
    """Run full 3-stage inference on a satellite image.

    This command runs the complete GridTracer pipeline:
    1. Tower Detection (Faster R-CNN)
    2. Line Segmentation (U-Net)
    3. Graph Inference (connect towers using segmentation scores)

    Examples:
        uv run main infer raw_data/NZ_Dunedin/NZ_Dunedin_1.jpg
        uv run main infer image.jpg --tower-ckpt tower.ckpt --line-ckpt line.ckpt
    """
    import json

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image

    from detect_electrical_utility_from_satellite_images.evaluation.graph_eval import (
        create_inference_summary,
        visualize_graph,
    )
    from detect_electrical_utility_from_satellite_images.models.graph_inference import (
        infer_graph,
    )
    from detect_electrical_utility_from_satellite_images.training import (
        TowerDetectorModule,
    )
    from detect_electrical_utility_from_satellite_images.training.line_training import (
        LineSegmentorModule,
    )

    log = get_logger()
    config = load_config()

    if not image.exists():
        log.error("image_not_found", path=str(image))
        typer.echo(f"Error: Image not found: {image}")
        raise typer.Exit(code=1)

    # Auto-select checkpoints if not provided
    if tower_checkpoint is None:
        tower_checkpoint = _find_checkpoint("tower", region)
        if tower_checkpoint is None:
            typer.echo(f"Error: No tower checkpoint found for region '{region}'")
            typer.echo("Run 'uv run main train tower' first or provide --tower-ckpt")
            raise typer.Exit(code=1)
        typer.echo(f"Tower checkpoint: {tower_checkpoint}")

    if line_checkpoint is None:
        line_checkpoint = _find_checkpoint("line", region)
        if line_checkpoint is None:
            typer.echo(f"Error: No line checkpoint found for region '{region}'")
            typer.echo("Run 'uv run main train line' first or provide --line-ckpt")
            raise typer.Exit(code=1)
        typer.echo(f"Line checkpoint: {line_checkpoint}")

    log.info(
        "inference_start",
        image=str(image),
        tower_checkpoint=str(tower_checkpoint),
        line_checkpoint=str(line_checkpoint),
    )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    typer.echo(f"Using device: {device}")

    # Load image
    typer.echo(f"Loading image: {image}")
    pil_image = Image.open(image).convert("RGB")
    image_np = np.array(pil_image)
    
    # Convert to tensor (C, H, W)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_batch = image_tensor.unsqueeze(0).to(device)

    # --- Stage 1: Tower Detection ---
    typer.echo("\n=== Stage 1: Tower Detection ===")
    tower_model = TowerDetectorModule.load_from_checkpoint(
        str(tower_checkpoint), config=config
    )
    tower_model.eval()
    tower_model = tower_model.to(device)

    with torch.no_grad():
        tower_outputs = tower_model.model(image_batch)

    # Extract detections
    boxes = tower_outputs[0]["boxes"]
    scores = tower_outputs[0]["scores"]
    
    # Filter by confidence
    conf_thresh = config.tower_detection.confidence_threshold
    mask = scores >= conf_thresh
    boxes = boxes[mask]
    scores = scores[mask]

    # Compute centroids
    if len(boxes) > 0:
        centroids = torch.stack([
            (boxes[:, 0] + boxes[:, 2]) / 2,
            (boxes[:, 1] + boxes[:, 3]) / 2,
        ], dim=1)
    else:
        centroids = torch.zeros((0, 2), device=device)

    typer.echo(f"Detected {len(centroids)} towers")
    log.info("stage_1_complete", num_towers=len(centroids))

    # --- Stage 2: Line Segmentation ---
    typer.echo("\n=== Stage 2: Line Segmentation ===")
    line_model = LineSegmentorModule.load_from_checkpoint(
        str(line_checkpoint), config=config
    )
    line_model.eval()
    line_model = line_model.to(device)

    with torch.no_grad():
        seg_output = line_model.model(image_batch)
        seg_map = torch.sigmoid(seg_output).squeeze()

    typer.echo(f"Segmentation map shape: {seg_map.shape}")
    log.info("stage_2_complete", seg_shape=list(seg_map.shape))

    # --- Stage 3: Graph Inference ---
    typer.echo("\n=== Stage 3: Graph Inference ===")
    graph = infer_graph(
        tower_centroids=centroids,
        tower_scores=scores,
        segmentation_map=seg_map,
        max_distance_m=config.graph_inference.max_distance_m,
        connectivity_threshold=config.graph_inference.connectivity_threshold,
        line_width=config.line_segmentation.line_width_inference,
        resolution_m_per_px=config.graph_inference.resolution_m_per_px,
    )

    typer.echo(f"Graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
    log.info(
        "stage_3_complete",
        num_nodes=graph.num_nodes,
        num_edges=graph.num_edges,
    )

    # Print results summary
    typer.echo("\n=== Inference Results ===")
    typer.echo(f"Towers detected:     {graph.num_nodes}")
    typer.echo(f"Connections found:   {graph.num_edges}")
    typer.echo(f"Max distance:        {config.graph_inference.max_distance_m}m")
    typer.echo(f"Conn. threshold:     {config.graph_inference.connectivity_threshold}")

    # Setup output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output is None:
        output = Path(f"inference_results/{timestamp}")
    output.mkdir(parents=True, exist_ok=True)

    # Create and save visualizations
    typer.echo(f"\nSaving results to: {output}")

    # Summary figure
    fig_summary = create_inference_summary(
        image=image_np,
        graph=graph,
        boxes=boxes.cpu().numpy(),
        segmentation_map=seg_map.cpu().numpy(),
    )
    summary_path = output / "pipeline_summary.png"
    fig_summary.savefig(summary_path, dpi=150, bbox_inches="tight")
    typer.echo(f"  Pipeline summary: {summary_path}")

    # Graph visualization
    fig_graph = visualize_graph(
        image=image_np,
        graph=graph,
        segmentation_map=seg_map.cpu().numpy(),
    )
    graph_path = output / "graph_overlay.png"
    fig_graph.savefig(graph_path, dpi=150, bbox_inches="tight")
    typer.echo(f"  Graph overlay:    {graph_path}")

    # Save graph as JSON
    if save_graph:
        graph_json_path = output / "graph.json"
        with open(graph_json_path, "w") as f:
            json.dump(graph.to_dict(), f, indent=2)
        typer.echo(f"  Graph JSON:       {graph_json_path}")

    log.info("inference_complete", output_dir=str(output))

    if show:
        typer.echo("\nDisplaying visualizations... (close windows to continue)")
        plt.show()
    else:
        plt.close("all")


@app.command()
def info() -> None:
    """Display configuration and system information."""
    import torch

    config = load_config()

    typer.echo("\n=== GridTracer Configuration ===\n")
    typer.echo(f"Patch size: {config.preprocessing.patch_size}px")
    typer.echo(f"Data directory: {config.paths.data_dir}")
    typer.echo(f"Output directory: {config.paths.output_dir}")
    typer.echo(f"Max epochs: {config.training.max_epochs}")
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
