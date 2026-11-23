
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================================================
# RESIDUAL BLOCK
# Same architecture as the original model
# ==================================================

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm1 = nn.GroupNorm(8, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    bias=False
                ),
                nn.GroupNorm(8, out_channels)
            )

        else:

            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.relu(
            self.norm1(self.conv1(x))
        )

        out = self.norm2(self.conv2(out))

        out = out + identity

        return self.relu(out)


# ==================================================
# MULTI-SCALE FEATURE FUSION
# ==================================================

class MultiScaleFusion(nn.Module):

    def __init__(self):

        super().__init__()

        # Convert every decoder feature map
        # into 16 channels.

        self.proj4 = nn.Sequential(
            nn.Conv2d(128, 16, kernel_size=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True)
        )

        self.proj3 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True)
        )

        self.proj2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True)
        )

        self.proj1 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True)
        )

        # Combine the four feature maps.

        self.fusion = nn.Sequential(

            nn.Conv2d(
                64,
                16,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.GroupNorm(8, 16)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, d4, d3, d2, d1):

        target_size = d1.shape[-2:]

        # Project features to a common channel size.

        f4 = self.proj4(d4)
        f3 = self.proj3(d3)
        f2 = self.proj2(d2)
        f1 = self.proj1(d1)

        # Resize all features to the finest resolution.

        f4 = F.interpolate(
            f4,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        f3 = F.interpolate(
            f3,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        f2 = F.interpolate(
            f2,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        # Concatenate multi-scale features.

        combined = torch.cat(
            [f4, f3, f2, f1],
            dim=1
        )

        # Fuse the combined information.

        fused = self.fusion(combined)

        # Preserve the original fine-scale features.

        return self.relu(fused + d1)


# ==================================================
# MULTI-SCALE RESIDUAL U-NET
# ==================================================

class MultiScaleResidualUNet(nn.Module):

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
        self.up4 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )
        self.dec4 = ResidualBlock(256, 128)

        self.up3 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )
        self.dec3 = ResidualBlock(128, 64)

        self.up2 = nn.ConvTranspose2d(
            64, 32, 2, stride=2
        )
        self.dec2 = ResidualBlock(64, 32)

        self.up1 = nn.ConvTranspose2d(
            32, 16, 2, stride=2
        )
        self.dec1 = ResidualBlock(32, 16)

        # NEW: Multi-scale feature fusion
        self.multiscale_fusion = MultiScaleFusion()

        # Final segmentation layer
        self.output = nn.Conv2d(
            16,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder
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

        # NEW: Combine features from all decoder levels.
        fused_features = self.multiscale_fusion(
            d4,
            d3,
            d2,
            d1
        )

        # Raw segmentation logits.
        return self.output(fused_features)


# ==================================================
# SAFE PRETRAINED WEIGHT TRANSFER
# ==================================================

def load_pretrained_weights(model, checkpoint_path, device):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    pretrained_state = checkpoint["model_state_dict"]

    model_state = model.state_dict()

    compatible_weights = {}
    skipped_weights = []

    for key, value in pretrained_state.items():

        if (
            key in model_state
            and model_state[key].shape == value.shape
        ):

            compatible_weights[key] = value

        else:

            skipped_weights.append(key)

    model_state.update(compatible_weights)

    model.load_state_dict(model_state)

    total_old_parameters = len(pretrained_state)
    loaded_parameters = len(compatible_weights)

    print("\nWEIGHT TRANSFER SUMMARY")
    print("-" * 45)

    print(
        f"Transferred tensors: "
        f"{loaded_parameters}/{total_old_parameters}"
    )

    print(
        f"New or skipped tensors: "
        f"{len(model_state) - loaded_parameters}"
    )

    print(
        "New multi-scale layers initialized independently."
    )

    if skipped_weights:
        print("\nSkipped old tensors:")
        for key in skipped_weights:
            print(key)

    return checkpoint
