import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, random_split
import typing
import logging
import time
from pathlib import Path
import numpy as np

from .model import create_model, save_checkpoint, count_parameters
from .dataset import SatteliteImgsDataset
from .config import AppConfig

logger = logging.getLogger(__name__)

def create_mock_dataloader(
    batch_size: int,
    num_samples: int = 100,
    image_size: typing.Tuple[int, int] = (1024, 1024),
    num_classes: int = 5,
    device: str = "cpu"
) -> DataLoader:
    """Create a mock dataloader for testing training pipeline.
    
    Args:
        batch_size: Batch size
        num_samples: Number of mock samples to generate
        image_size: (height, width) of mock images
        num_classes: Number of classes for segmentation masks
        device: Device to place tensors on
        
    Returns:
        DataLoader yielding (images, masks) as torch tensors
    """
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples, image_size, num_classes, device):
            self.num_samples = num_samples
            self.image_size = image_size
            self.num_classes = num_classes
            self.device = device
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            # Generate random image (3 channels, normalized to [0, 1])
            image = torch.rand(3, self.image_size[0], self.image_size[1], device=self.device)
            # Generate random mask with class indices
            mask = torch.randint(0, self.num_classes, (1, self.image_size[0], self.image_size[1]), device=self.device)
            return image, mask.squeeze(0)  # Remove channel dimension for mask
    
    dataset = MockDataset(num_samples, image_size, num_classes, device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int
) -> float:
    """Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (images, masks) in enumerate(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        
        # Ensure masks have correct shape (batch_size, height, width)
        if masks.ndim == 4:  # If masks have channel dimension
            masks = masks.squeeze(1)  # Remove channel dimension
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            logger.debug(f"Epoch {epoch}, Batch {batch_idx}: loss = {loss.item():.4f}")
    
    avg_loss = total_loss / max(num_batches, 1)
    logger.info(f"Epoch {epoch} training complete: avg loss = {avg_loss:.4f}")
    return avg_loss

def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str
) -> float:
    """Validate model.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    logger.info(f"Validation complete: avg loss = {avg_loss:.4f}")
    return avg_loss

def train_model(cfg: AppConfig, use_mock_data: bool = True) -> None:
    """Main training function.
    
    Args:
        cfg: Application configuration
        use_mock_data: If True, use mock data; if False, use real dataset
    """
    start_time = time.perf_counter()
    
    # Setup device
    device = cfg.model.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    # Create model
    model = create_model(
        architecture=cfg.model.architecture,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=cfg.model.in_channels,
        classes=cfg.model.classes,
        device=device
    )
    
    logger.info(f"Model has {count_parameters(model):,} trainable parameters")
    
    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
    
    # Create dataloaders
    if use_mock_data:
        logger.info("Using mock data for training")
        train_loader = create_mock_dataloader(
            batch_size=cfg.model.batch_size,
            num_samples=100,
            image_size=(cfg.preprocessing.patch_size, cfg.preprocessing.patch_size),
            num_classes=cfg.model.classes,
            device=device
        )
        val_loader = create_mock_dataloader(
            batch_size=cfg.model.batch_size,
            num_samples=20,
            image_size=(cfg.preprocessing.patch_size, cfg.preprocessing.patch_size),
            num_classes=cfg.model.classes,
            device=device
        )
    else:
        logger.info("Using real data for training")
        # Load real dataset
        img_patch_paths = list(sorted((cfg.paths.output_dir / "img_patches").glob("*.png")))
        mask_patch_paths = list(sorted((cfg.paths.output_dir / "mask_patches").glob("*.png")))
        
        if len(img_patch_paths) == 0:
            raise ValueError(f"No image patches found in {cfg.paths.output_dir / 'img_patches'}")
        if len(mask_patch_paths) == 0:
            raise ValueError(f"No mask patches found in {cfg.paths.output_dir / 'mask_patches'}")
        
        logger.info(f"Found {len(img_patch_paths)} image patches and {len(mask_patch_paths)} mask patches")
        
        # Define transforms (same as in dataset.py)
        from torchvision.transforms import v2
        transforms = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            v2.GaussianBlur(kernel_size=3),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Lambda(lambda x: x.to(torch.long) if hasattr(x, '__class__') and x.__class__.__name__ == 'Mask' else x),
        ])
        
        # Create dataset
        from .dataset import SatteliteImgsDataset
        dataset = SatteliteImgsDataset(
            img_paths=img_patch_paths,
            mask_paths=mask_patch_paths,
            transforms=transforms
        )
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.model.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.model.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        logger.info(f"Created dataloaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Training loop
    best_loss = float('inf')
    checkpoint_dir = cfg.paths.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, cfg.model.epochs + 1):
        logger.info(f"Starting epoch {epoch}/{cfg.model.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Save checkpoint if best
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
        
        # Save periodic checkpoint
        if epoch % 5 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
    
    # Save final model
    final_path = checkpoint_dir / "final_model.pt"
    save_checkpoint(model, optimizer, cfg.model.epochs, val_loss, final_path)
    
    end_time = time.perf_counter()
    logger.info(f"Training completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Best validation loss: {best_loss:.4f}")

if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path
    from .utils.logging_config import setup_logger
    import sys
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", required=True, help="Path to config file")
    parser.add_argument("--real-data", action="store_true", help="Use real data instead of mock")
    args = parser.parse_args()
    
    # Load config
    cfg_path = Path(args.config).with_suffix(".yaml")
    with open(cfg_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    from .config import AppConfig
    import pydantic
    try:
        cfg = AppConfig(**cfg_dict)
    except pydantic.ValidationError as e:
        print(f"FATAL. CONFIG FAILED: {e}")
        sys.exit(1)
    
    # Setup logging
    setup_logger(
        cfg.paths.logging_dir_name,
        cfg.logging.logger_lvl,
        cfg.logging.console_handler_lvl,
        cfg.logging.file_handler_lvl
    )
    
    # Train
    train_model(cfg, use_mock_data=not args.real_data)
