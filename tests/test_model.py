"""
Tests for model creation and utilities.
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from detect_electrical_utility_from_satellite_images.model import (
    count_parameters,
    create_model,
    load_checkpoint,
    save_checkpoint,
)


def test_create_model_unet():
    """Test creating a UNet model."""
    model = create_model(
        architecture="unet",
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=5,
        device="cpu",
    )

    assert isinstance(model, nn.Module)
    # Check output channels
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 5, 256, 256)


def test_create_model_deeplabv3():
    """Test creating a DeepLabV3 model."""
    model = create_model(
        architecture="deeplabv3",
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=5,
        device="cpu",
    )

    assert isinstance(model, nn.Module)
    model.eval()  # Set to eval mode to avoid batch norm issues
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 5, 256, 256)


def test_create_model_fpn():
    """Test creating an FPN model."""
    model = create_model(
        architecture="fpn",
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=5,
        device="cpu",
    )

    assert isinstance(model, nn.Module)
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 5, 256, 256)


def test_create_model_invalid_architecture():
    """Test creating model with invalid architecture."""
    with pytest.raises(ValueError, match="Unknown architecture"):
        create_model(
            architecture="invalid",
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=5,
            device="cpu",
        )


def test_create_model_no_pretrained():
    """Test creating model without pretrained weights."""
    model = create_model(
        architecture="unet",
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=5,
        device="cpu",
    )

    assert isinstance(model, nn.Module)
    # Model should still work
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 5, 256, 256)


def test_create_model_different_channels():
    """Test creating model with different input channels."""
    # Test with 4 channels (RGBA)
    model = create_model(
        architecture="unet",
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=4,
        classes=5,
        device="cpu",
    )

    x = torch.randn(1, 4, 256, 256)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 5, 256, 256)


def test_count_parameters():
    """Test counting model parameters."""
    model = create_model(
        architecture="unet",
        encoder_name="resnet18",
        encoder_weights=None,  # Smaller without pretrained
        in_channels=3,
        classes=5,
        device="cpu",
    )

    param_count = count_parameters(model)
    assert isinstance(param_count, int)
    assert param_count > 0
    # ResNet18 UNet should have around 13-14M parameters
    assert 10_000_000 < param_count < 20_000_000


def test_save_load_checkpoint():
    """Test saving and loading model checkpoints."""
    # Create a simple model and optimizer
    model = create_model(
        architecture="unet",
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=5,
        device="cpu",
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint.pth"

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=10,
            loss=0.5,
            path=checkpoint_path,
        )

        assert checkpoint_path.exists()

        # Create new model and optimizer
        new_model = create_model(
            architecture="unet",
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=5,
            device="cpu",
        )
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

        # Load checkpoint
        epoch, loss = load_checkpoint(
            model=new_model,
            optimizer=new_optimizer,
            path=checkpoint_path,
        )

        assert epoch == 10
        assert loss == 0.5

        # Verify model parameters are loaded
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)


def test_save_load_checkpoint_no_optimizer():
    """Test saving and loading checkpoints without optimizer."""
    model = create_model(
        architecture="unet",
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=5,
        device="cpu",
    )

    # Create a dummy optimizer for saving
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint.pth"

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=5,
            loss=0.3,
            path=checkpoint_path,
        )

        assert checkpoint_path.exists()

        # Load without optimizer
        new_model = create_model(
            architecture="unet",
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=5,
            device="cpu",
        )

        epoch, loss = load_checkpoint(
            model=new_model,
            optimizer=None,
            path=checkpoint_path,
        )

        assert epoch == 5
        assert loss == 0.3


def test_model_device_placement():
    """Test model creation on different devices."""
    # Test CPU
    model_cpu = create_model(
        architecture="unet",
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=5,
        device="cpu",
    )
    assert next(model_cpu.parameters()).device.type == "cpu"

    # Test CUDA if available
    if torch.cuda.is_available():
        model_cuda = create_model(
            architecture="unet",
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=3,
            classes=5,
            device="cuda",
        )
        assert next(model_cuda.parameters()).device.type == "cuda"


def test_model_different_encoders():
    """Test creating models with different encoders."""
    encoders = ["resnet18", "resnet34", "resnet50", "efficientnet-b0", "mobilenet_v2"]

    for encoder in encoders:
        model = create_model(
            architecture="unet",
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=5,
            device="cpu",
        )

        assert isinstance(model, nn.Module)
        # Quick forward pass
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 5, 256, 256)


def test_model_gradient_flow():
    """Test that gradients flow through the model."""
    model = create_model(
        architecture="unet",
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=5,
        device="cpu",
    )

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Forward pass
    x = torch.randn(2, 3, 256, 256, requires_grad=False)
    target = torch.randint(0, 5, (2, 256, 256))

    output = model(x)
    loss = torch.nn.functional.cross_entropy(output, target)

    # Backward pass
    loss.backward()

    # Check gradients
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None:
            has_gradients = True
            break

    assert has_gradients, "No gradients found in model parameters"
