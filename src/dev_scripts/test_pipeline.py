"""
Test scripts for the training pipeline.
These functions can be invoked via `uv run` after adding to pyproject.toml.
"""

import subprocess
import sys
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def run_command(cmd: list[str]) -> bool:
    """Run a command and return True if successful."""
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        # Add src to PYTHONPATH for module discovery
        env = {**os.environ, "PYTHONPATH": "src"}
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
        logger.info(f"Command output:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"Command stderr:\n{result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(f"stdout:\n{e.stdout}")
        logger.error(f"stderr:\n{e.stderr}")
        return False


def test_cpu() -> None:
    """Test CPU training pipeline with mock data."""
    logger.info("Testing CPU training pipeline with mock data...")
    success = run_command([
        sys.executable, "-m", "detect_electrical_utility_from_satellite_images.main",
        "--config", "config_debug_cpu",
        "train"
    ])
    if success:
        logger.info("CPU pipeline test PASSED")
    else:
        logger.error("CPU pipeline test FAILED")
        sys.exit(1)


def test_gpu() -> None:
    """Test GPU training pipeline with mock data."""
    logger.info("Testing GPU training pipeline with mock data...")
    success = run_command([
        sys.executable, "-m", "detect_electrical_utility_from_satellite_images.main",
        "--config", "config_debug_gpu",
        "train"
    ])
    if success:
        logger.info("GPU pipeline test PASSED")
    else:
        logger.error("GPU pipeline test FAILED")
        sys.exit(1)


def test_dataset() -> None:
    """Test dataset loading and validation."""
    logger.info("Testing dataset loading...")
    success = run_command([
        sys.executable, "-m", "detect_electrical_utility_from_satellite_images.main",
        "--config", "config_debug_cpu",
        "test-dataset"
    ])
    if success:
        logger.info("Dataset test PASSED")
    else:
        logger.error("Dataset test FAILED")
        sys.exit(1)


def test_real_data() -> None:
    """Test training pipeline with real data (requires preprocessed patches)."""
    logger.info("Testing training pipeline with real data...")
    
    # Check if patches exist
    img_dir = Path("output_imgs/patches4/img_patches")
    mask_dir = Path("output_imgs/patches4/mask_patches")
    
    if not img_dir.exists() or not mask_dir.exists():
        logger.error(f"Image or mask directory not found: {img_dir}, {mask_dir}")
        logger.error("Please run preprocessing first: uv run main preprocess --config config_debug_cpu")
        sys.exit(1)
    
    img_files = list(img_dir.glob("*.png"))
    mask_files = list(mask_dir.glob("*.png"))
    
    if len(img_files) == 0 or len(mask_files) == 0:
        logger.error(f"No PNG files found in {img_dir} or {mask_dir}")
        logger.error("Please run preprocessing first: uv run main preprocess --config config_debug_cpu")
        sys.exit(1)
    
    logger.info(f"Found {len(img_files)} image patches and {len(mask_files)} mask patches")
    
    # Run training with real data
    success = run_command([
        sys.executable, "-m", "detect_electrical_utility_from_satellite_images.main",
        "--config", "config_debug_cpu",
        "train", "--real-data"
    ])
    
    if success:
        logger.info("Real data pipeline test PASSED")
    else:
        logger.error("Real data pipeline test FAILED")
        sys.exit(1)


def main() -> None:
    """Main entry point when script is run directly."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test training pipeline")
    parser.add_argument("--test", choices=["cpu", "gpu", "dataset", "real-data"],
                       required=True, help="Which test to run")
    
    args = parser.parse_args()
    
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if args.test == "cpu":
        test_cpu()
    elif args.test == "gpu":
        test_gpu()
    elif args.test == "dataset":
        test_dataset()
    elif args.test == "real-data":
        test_real_data()


if __name__ == "__main__":
    main()
