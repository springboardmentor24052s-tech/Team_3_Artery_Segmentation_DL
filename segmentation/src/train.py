
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.dataset import StenosisDataset
from src.unet import UNet


BATCH_SIZE = 2
ADDITIONAL_EPOCHS = 3
LEARNING_RATE = 1e-4

DEVICE = torch.device("cpu")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Resume from the latest extended best checkpoint (Epoch 10).
START_CHECKPOINT = OUTPUT_DIR / "unet_baseline_extended_best.pth"

# New filenames: previous checkpoints will not be overwritten.
LATEST_CHECKPOINT = OUTPUT_DIR / "unet_baseline_round2_latest.pth"
BEST_CHECKPOINT = OUTPUT_DIR / "unet_baseline_round2_best.pth"


def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)

    intersection = (probs * targets).sum(1)

    dice = (2 * intersection + smooth) / (
        probs.sum(1) + targets.sum(1) + smooth
    )

    return 1 - dice.mean()


def calculate_global_dice(model, loader):
    model.eval()

    total_intersection = 0.0
    total_denominator = 0.0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            total_intersection += (
                predictions * masks
            ).sum().item()

            total_denominator += (
                predictions.sum().item()
                + masks.sum().item()
            )

    return (
        2 * total_intersection + 1.0
    ) / (total_denominator + 1.0)


def main():
    print("Device:", DEVICE)
    print("Loading datasets...")

    train_dataset = StenosisDataset("train")
    val_dataset = StenosisDataset("val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    if not START_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Starting checkpoint not found: {START_CHECKPOINT}"
        )

    model = UNet().to(DEVICE)

    bce_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([10.0], device=DEVICE)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    checkpoint = torch.load(
        START_CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        print("Optimizer state restored.")
    else:
        print("Using a new optimizer.")

    start_epoch = checkpoint["epoch"]
    best_dice = checkpoint["val_dice"]

    # Preserve the starting best model as the initial round-2 best.
    if not BEST_CHECKPOINT.exists():
        torch.save(checkpoint, BEST_CHECKPOINT)

    print("Loaded checkpoint:", START_CHECKPOINT)
    print("Starting epoch:", start_epoch + 1)
    print("Starting validation Dice:", best_dice)
    print("Learning rate:", optimizer.param_groups[0]["lr"])

    for epoch in range(
        start_epoch,
        start_epoch + ADDITIONAL_EPOCHS
    ):
        model.train()
        total_train_loss = 0.0

        print(f"\nEpoch {epoch + 1}")

        for batch_index, (images, masks) in enumerate(
            train_loader, start=1
        ):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)

            loss = (
                bce_loss(logits, masks)
                + dice_loss(logits, masks)
            )

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if (
                batch_index % 25 == 0
                or batch_index == len(train_loader)
            ):
                print(
                    f"Batch {batch_index}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        average_train_loss = (
            total_train_loss / len(train_loader)
        )

        val_dice = calculate_global_dice(
            model, val_loader
        )

        print(f"Average training loss: {average_train_loss:.4f}")
        print(f"Validation global Dice: {val_dice:.4f}")

        save_data = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": val_dice,
        }

        torch.save(save_data, LATEST_CHECKPOINT)
        print("Latest round-2 checkpoint saved.")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(save_data, BEST_CHECKPOINT)
            print("New round-2 best model saved.")

    print("\nBaseline U-Net round-2 training completed.")
    print("Best validation Dice:", best_dice)


if __name__ == "__main__":
    main()
