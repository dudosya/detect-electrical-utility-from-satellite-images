import torch
import torch.nn as nn
import typing
import logging
from pathlib import Path
from segmentation_models_pytorch import Unet, DeepLabV3, FPN

logger = logging.getLogger(__name__)

def create_model(
    architecture: str,
    encoder_name: str,
    encoder_weights: typing.Optional[str],
    in_channels: int,
    classes: int,
    device: str = "cpu"
) -> nn.Module:
    """Create a segmentation model from configuration.
    
    Args:
        architecture: Model architecture ('unet', 'deeplabv3', 'fpn')
        encoder_name: Backbone encoder name
        encoder_weights: Pretrained weights ('imagenet' or None)
        in_channels: Number of input channels (3 for RGB, 4 for RGBA)
        classes: Number of output classes (including background)
        device: Device to place model on ('cpu' or 'cuda')
        
    Returns:
        Initialized PyTorch model
    """
    model_args = {
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights,
        "in_channels": in_channels,
        "classes": classes,
    }
    
    if architecture == "unet":
        model = Unet(**model_args)
    elif architecture == "deeplabv3":
        model = DeepLabV3(**model_args)
    elif architecture == "fpn":
        model = FPN(**model_args)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    model = model.to(device)
    logger.info(f"Created {architecture} model with {encoder_name} encoder on {device}")
    logger.debug(f"Model parameters: {model_args}")
    
    return model

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: typing.Union[str, Path, typing.IO]
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, str(path))
    logger.info(f"Checkpoint saved to {path} (epoch {epoch}, loss {loss:.4f})")

def load_checkpoint(
    model: nn.Module,
    optimizer: typing.Optional[torch.optim.Optimizer],
    path: typing.Union[str, Path, typing.IO]
) -> typing.Tuple[int, float]:
    """Load model checkpoint.
    
    Returns:
        Tuple of (epoch, loss)
    """
    checkpoint = torch.load(str(path), map_location=next(model.parameters()).device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    logger.info(f"Checkpoint loaded from {path} (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss
