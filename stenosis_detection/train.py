
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model import ResidualUNet


# ---------------- CONFIGURATION ----------------

PATCH_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 0.0001

POSITIVE_PROBABILITY = 0.70
PATIENCE = 8

DEVICE = torch.device("cpu")

CHECKPOINT_DIR = Path("stenosis_detection/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = (
    CHECKPOINT_DIR / "resunet_stenosis_best.pt"
)


# ---------------- RANDOM SEED ----------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(4)


# ---------------- PATCH DATASET ----------------

class PatchDataset(Dataset):

    def __init__(self, base_dataset, patches_per_image=2):

        self.base_dataset = base_dataset
        self.patches_per_image = patches_per_image

        self.positive_indices = []
        self.negative_indices = []

        for index in range(len(base_dataset)):

            _, mask = base_dataset[index]

            if mask.sum().item() > 0:
                self.positive_indices.append(index)
            else:
                self.negative_indices.append(index)

        if not self.positive_indices:
            raise ValueError("No positive images found!")

        if not self.negative_indices:
            raise ValueError("No negative images found!")

    def __len__(self):
        return len(self.base_dataset) * self.patches_per_image

    def __getitem__(self, index):

        positive = random.random() < POSITIVE_PROBABILITY

        # -------- POSITIVE PATCH --------

        if positive:

            image_index = random.choice(
                self.positive_indices
            )

            image, mask = self.base_dataset[image_index]

            coordinates = torch.nonzero(mask[0] > 0)

            selected = coordinates[
                random.randrange(len(coordinates))
            ]

            y, x = selected.tolist()

            top = random.randint(
                max(0, y - PATCH_SIZE + 1),
                min(y, mask.shape[-2] - PATCH_SIZE)
            )

            left = random.randint(
                max(0, x - PATCH_SIZE + 1),
                min(x, mask.shape[-1] - PATCH_SIZE)
            )

        # -------- NEGATIVE PATCH --------

        else:

            found = False

            for _ in range(50):

                image_index = random.randrange(
                    len(self.base_dataset)
                )

                image, mask = self.base_dataset[image_index]

                top = random.randint(
                    0, mask.shape[-2] - PATCH_SIZE
                )

                left = random.randint(
                    0, mask.shape[-1] - PATCH_SIZE
                )

                candidate = mask[
                    :,
                    top:top + PATCH_SIZE,
                    left:left + PATCH_SIZE
                ]

                if candidate.sum().item() == 0:
                    found = True
                    break

            if not found:

                image_index = random.choice(
                    self.negative_indices
                )

                image, mask = self.base_dataset[image_index]

                top = random.randint(
                    0, mask.shape[-2] - PATCH_SIZE
                )

                left = random.randint(
                    0, mask.shape[-1] - PATCH_SIZE
                )

        # -------- EXTRACT PATCH --------

        image_patch = image[
            :,
            top:top + PATCH_SIZE,
            left:left + PATCH_SIZE
        ]

        mask_patch = mask[
            :,
            top:top + PATCH_SIZE,
            left:left + PATCH_SIZE
        ]

        # -------- DATA AUGMENTATION --------

        if random.random() < 0.5:

            image_patch = torch.flip(
                image_patch, dims=[2]
            )

            mask_patch = torch.flip(
                mask_patch, dims=[2]
            )

        if random.random() < 0.5:

            image_patch = torch.flip(
                image_patch, dims=[1]
            )

            mask_patch = torch.flip(
                mask_patch, dims=[1]
            )

        if random.random() < 0.5:

            rotations = random.randint(1, 3)

            image_patch = torch.rot90(
                image_patch, rotations, dims=[1, 2]
            )

            mask_patch = torch.rot90(
                mask_patch, rotations, dims=[1, 2]
            )

        return image_patch, mask_patch


# ---------------- DICE LOSS ----------------

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):

        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.flatten(1)
        targets = targets.flatten(1)

        intersection = (
            probabilities * targets
        ).sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probabilities.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


# ---------------- COMBINED LOSS ----------------

class CombinedLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):

        return (
            0.5 * self.bce(logits, targets)
            + 0.5 * self.dice(logits, targets)
        )


# ---------------- VALIDATION ----------------

def validate(model, loader):

    model.eval()

    total_soft_dice = 0.0
    total_hard_dice = 0.0
    total_iou = 0.0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)

            probabilities = torch.sigmoid(logits)

            # -------- SOFT DICE --------

            soft_intersection = (
                probabilities * masks
            ).sum(dim=(1, 2, 3))

            soft_pred_sum = probabilities.sum(
                dim=(1, 2, 3)
            )

            mask_sum = masks.sum(
                dim=(1, 2, 3)
            )

            soft_dice = (
                2 * soft_intersection + 1e-6
            ) / (
                soft_pred_sum + mask_sum + 1e-6
            )

            # -------- HARD DICE AND IOU --------

            predictions = (
                probabilities >= 0.5
            ).float()

            intersection = (
                predictions * masks
            ).sum(dim=(1, 2, 3))

            pred_sum = predictions.sum(
                dim=(1, 2, 3)
            )

            hard_dice = (
                2 * intersection + 1e-6
            ) / (
                pred_sum + mask_sum + 1e-6
            )

            union = pred_sum + mask_sum - intersection

            iou = (
                intersection + 1e-6
            ) / (
                union + 1e-6
            )

            total_soft_dice += soft_dice.sum().item()
            total_hard_dice += hard_dice.sum().item()
            total_iou += iou.sum().item()

    number_of_images = len(loader.dataset)

    return (
        total_soft_dice / number_of_images,
        total_hard_dice / number_of_images,
        total_iou / number_of_images
    )


# ---------------- MAIN TRAINING ----------------

def main():

    print("Loading datasets...")

    train_base = StenosisDataset("train")
    val_dataset = StenosisDataset("val")

    train_dataset = PatchDataset(
        train_base,
        patches_per_image=2
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = ResidualUNet().to(DEVICE)

    criterion = CombinedLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    best_soft_dice = -1.0
    epochs_without_improvement = 0

    print("\nStarting Residual U-Net training")
    print("Device:", DEVICE)
    print("Training patches:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Maximum epochs:", EPOCHS)

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for images, masks in train_loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(logits, masks)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = (
            running_loss / len(train_loader)
        )

        (
            val_soft_dice,
            val_hard_dice,
            val_iou
        ) = validate(model, val_loader)

        scheduler.step(val_soft_dice)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Soft Dice: {val_soft_dice:.4f} "
            f"Hard Dice: {val_hard_dice:.4f} "
            f"IoU: {val_iou:.4f} "
            f"LR: {current_lr:.6f}"
        )

        if val_soft_dice > best_soft_dice:

            best_soft_dice = val_soft_dice
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_soft_dice": best_soft_dice,
                    "val_hard_dice": val_hard_dice,
                    "val_iou": val_iou,
                },
                BEST_MODEL_PATH
            )

            print("Best model checkpoint saved!")

        else:

            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:

            print("Early stopping triggered.")
            break

    print("\nTraining completed!")
    print("Best validation Soft Dice:", best_soft_dice)
    print("Checkpoint:", BEST_MODEL_PATH)


if __name__ == "__main__":
    main()
