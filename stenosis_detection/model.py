
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )

        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )

        self.norm2 = nn.GroupNorm(8, out_channels)

        # Adjust shortcut dimensions when necessary
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, bias=False
                ),
                nn.GroupNorm(8, out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))

        out = out + identity

        return self.relu(out)


class ResidualUNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1):

        super().__init__()

        # Encoder
        self.enc1 = ResidualBlock(in_channels, 16)
        self.enc2 = ResidualBlock(16, 32)
        self.enc3 = ResidualBlock(32, 64)
        self.enc4 = ResidualBlock(64, 128)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(128, 256)

        # Decoder
        self.up4 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec4 = ResidualBlock(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = ResidualBlock(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = ResidualBlock(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = ResidualBlock(32, 16)

        # Final segmentation layer
        self.output = nn.Conv2d(
            16, out_channels, kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Return raw logits (no sigmoid here)
        return self.output(d1)
