"""U-Net model for line segmentation (Stage 2).

This implements a standard U-Net architecture for semantic segmentation
of power lines. The paper uses StackNetMTL for larger receptive field,
but U-Net serves as a strong baseline with pretrained encoder support.
"""

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive conv-bn-relu blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize double convolution block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            mid_channels: Number of channels after first conv (default: out_channels).
            dropout_rate: Dropout rate for MC Dropout uncertainty.
        """
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))

        layers.extend(
            [
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        )
        if dropout_rate > 0:
            layers.append(nn.Dropout2d(p=dropout_rate))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.0) -> None:
        """Initialize downsampling block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            dropout_rate: Dropout rate for MC Dropout uncertainty.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bilinear: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize upsampling block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            bilinear: Use bilinear upsampling instead of transposed conv.
            dropout_rate: Dropout rate for MC Dropout uncertainty.
        """
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2, dropout_rate=dropout_rate
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels, dropout_rate=dropout_rate)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection.

        Args:
            x1: Features from decoder (to be upsampled).
            x2: Skip connection features from encoder.

        Returns:
            Concatenated and convolved features.
        """
        x1 = self.up(x1)

        # Handle size mismatch due to pooling
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution to output channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize output convolution.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (classes).
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.conv(x)


class UNet(nn.Module):
    """Standard U-Net architecture for semantic segmentation.

    Architecture:
        - Encoder: 4 downsampling blocks (64 -> 128 -> 256 -> 512 -> 1024)
        - Decoder: 4 upsampling blocks with skip connections
        - Output: 1x1 conv to single channel probability map

    The model outputs logits; apply sigmoid for probabilities.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_features: int = 64,
        bilinear: bool = True,
        dropout_rate: float = 0.0,
    ) -> None:
        """Initialize U-Net.

        Args:
            in_channels: Number of input channels (3 for RGB).
            out_channels: Number of output channels (1 for binary segmentation).
            base_features: Number of features in first layer (doubles each level).
            bilinear: Use bilinear upsampling (faster) or transposed conv.
            dropout_rate: Dropout rate for MC Dropout uncertainty.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Feature channel progression
        f = base_features  # 64

        # Encoder
        self.inc = DoubleConv(in_channels, f, dropout_rate=dropout_rate)  # 64
        self.down1 = Down(f, f * 2, dropout_rate=dropout_rate)  # 128
        self.down2 = Down(f * 2, f * 4, dropout_rate=dropout_rate)  # 256
        self.down3 = Down(f * 4, f * 8, dropout_rate=dropout_rate)  # 512

        factor = 2 if bilinear else 1
        self.down4 = Down(f * 8, f * 16 // factor, dropout_rate=dropout_rate)  # 1024 or 512

        # Decoder
        self.up1 = Up(f * 16, f * 8 // factor, bilinear, dropout_rate=dropout_rate)  # 512 or 256
        self.up2 = Up(f * 8, f * 4 // factor, bilinear, dropout_rate=dropout_rate)  # 256 or 128
        self.up3 = Up(f * 4, f * 2 // factor, bilinear, dropout_rate=dropout_rate)  # 128 or 64
        self.up4 = Up(f * 2, f, bilinear, dropout_rate=dropout_rate)  # 64

        # Output
        self.outc = OutConv(f, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Logits tensor (B, out_channels, H, W). Apply sigmoid for probabilities.
        """
        # Encoder (save for skip connections)
        x1 = self.inc(x)  # 64
        x2 = self.down1(x1)  # 128
        x3 = self.down2(x2)  # 256
        x4 = self.down3(x3)  # 512
        x5 = self.down4(x4)  # 1024

        # Decoder (with skip connections)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Output logits
        logits = self.outc(x)
        return logits


def create_line_segmentor(
    in_channels: int = 3,
    base_features: int = 64,
    bilinear: bool = True,
    pretrained_encoder: bool = False,
    dropout_rate: float = 0.0,
) -> UNet:
    """Factory function to create line segmentation model.

    Args:
        in_channels: Number of input channels (3 for RGB).
        base_features: Base feature channels (64 default).
        bilinear: Use bilinear upsampling.
        pretrained_encoder: Whether to use pretrained encoder (not implemented yet).
        dropout_rate: Dropout rate for MC Dropout uncertainty.

    Returns:
        Configured UNet model.
    """
    model = UNet(
        in_channels=in_channels,
        out_channels=1,  # Binary segmentation
        base_features=base_features,
        bilinear=bilinear,
        dropout_rate=dropout_rate,
    )

    return model
