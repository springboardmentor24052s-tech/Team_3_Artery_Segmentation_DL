
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model import ResidualUNet


# ==================================================
# SETTINGS
# ==================================================

CHECKPOINT_PATH = Path(
    "stenosis_detection/checkpoints/"
    "resunet_hardnegative_best.pt"
)

CANDIDATES_PATH = Path(
    "stenosis_detection/hard_negative_candidates.json"
)

OUTPUT_PATH = Path(
    "stenosis_detection/checkpoints/"
    "resunet_boundary_aware_best.pt"
)

PATCH_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 8

LEARNING_RATE = 0.00001
EARLY_STOPPING_PATIENCE = 4

HARD_NEGATIVE_WEIGHT = 0.5
VALIDATION_THRESHOLD = 0.80

DEVICE = torch.device("cpu")

random.seed(42)
torch.manual_seed(42)

torch.set_num_threads(4)


# ==================================================
# SAFETY CHECKS
# ==================================================

if OUTPUT_PATH.exists():
    raise FileExistsError(
        f"\nOutput checkpoint already exists:\n{OUTPUT_PATH}\n"
        "Rename or back up the existing file before running again."
    )

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(CHECKPOINT_PATH)

if not CANDIDATES_PATH.exists():
    raise FileNotFoundError(CANDIDATES_PATH)


# ==================================================
# PATCH DATASET
# ==================================================

class HardNegativePatchDataset(Dataset):

    def __init__(self, base_dataset, candidates):

        self.base = base_dataset
        self.candidates = candidates

        self.positive_indices = [
            i for i in range(len(self.base))
        ]

    def __len__(self):
        return len(self.base)

    def crop(self, image, mask, x, y):

        image_patch = image[
            :,
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

        mask_patch = mask[
            :,
            y:y + PATCH_SIZE,
            x:x + PATCH_SIZE
        ]

        return image_patch, mask_patch

    def augment(self, image, mask):

        if random.random() < 0.5:
            image = torch.flip(image, [-1])
            mask = torch.flip(mask, [-1])

        if random.random() < 0.5:
            image = torch.flip(image, [-2])
            mask = torch.flip(mask, [-2])

        rotations = random.randint(0, 3)

        if rotations:
            image = torch.rot90(image, rotations, [-2, -1])
            mask = torch.rot90(mask, rotations, [-2, -1])

        return image.contiguous(), mask.contiguous()

    def positive_patch(self):

        for _ in range(100):

            index = random.randrange(len(self.base))

            image, mask = self.base[index]

            coordinates = torch.nonzero(mask[0] > 0.5)

            if len(coordinates) == 0:
                continue

            selected = coordinates[
                random.randrange(len(coordinates))
            ]

            cy = int(selected[0])
            cx = int(selected[1])

            x = max(
                0,
                min(cx - PATCH_SIZE // 2, 512 - PATCH_SIZE)
            )

            y = max(
                0,
                min(cy - PATCH_SIZE // 2, 512 - PATCH_SIZE)
            )

            return self.crop(image, mask, x, y), 1.0

        raise RuntimeError(
            "Could not find a positive training patch."
        )

    def hard_negative_patch(self):

        candidate = random.choice(self.candidates)

        index = candidate["image_index"]

        image, mask = self.base[index]

        patch = self.crop(
            image,
            mask,
            candidate["x"],
            candidate["y"]
        )

        return patch, HARD_NEGATIVE_WEIGHT

    def random_negative_patch(self):

        for _ in range(80):

            index = random.randrange(len(self.base))

            image, mask = self.base[index]

            x = random.randrange(512 - PATCH_SIZE + 1)
            y = random.randrange(512 - PATCH_SIZE + 1)

            patch = self.crop(image, mask, x, y)

            if torch.count_nonzero(patch[1]).item() == 0:
                return patch, 1.0

        # Safe fallback if an empty random patch is not found.
        return self.hard_negative_patch()

    def __getitem__(self, index):

        choice = random.random()

        if choice < 0.50:
            (image, mask), weight = self.positive_patch()

        elif choice < 0.75:
            (image, mask), weight = self.hard_negative_patch()

        else:
            (image, mask), weight = self.random_negative_patch()

        image, mask = self.augment(image, mask)

        return image, mask, torch.tensor(weight)


# ==================================================
# LOSS
# ==================================================


def calculate_loss(logits, targets, weights):

    # 1. Binary cross-entropy loss
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    bce = bce.mean(dim=(1, 2, 3))

    # 2. Dice loss
    probabilities = torch.sigmoid(logits)

    intersection = (
        probabilities * targets
    ).sum(dim=(1, 2, 3))

    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        denominator + 1e-6
    )

    dice_loss = 1.0 - dice

    # 3. Boundary extraction using morphological gradients
    def extract_boundary(tensor):

        dilated = nn.functional.max_pool2d(
            tensor,
            kernel_size=3,
            stride=1,
            padding=1
        )

        eroded = -nn.functional.max_pool2d(
            -tensor,
            kernel_size=3,
            stride=1,
            padding=1
        )

        return (dilated - eroded).clamp(0, 1)

    predicted_boundary = extract_boundary(probabilities)
    target_boundary = extract_boundary(targets)

    # 4. Boundary loss
    boundary_loss = torch.abs(
        predicted_boundary - target_boundary
    ).mean(dim=(1, 2, 3))

    # 5. Combined loss
    sample_loss = (
        0.45 * bce
        + 0.45 * dice_loss
        + 0.10 * boundary_loss
    )

    return (sample_loss * weights).mean()



# ==================================================
# VALIDATION
# ==================================================

def evaluate(model, dataset):

    model.eval()

    soft_dices = []
    hard_dices = []
    ious = []

    with torch.no_grad():

        for index in range(len(dataset)):

            image, mask = dataset[index]

            image = image.unsqueeze(0).to(DEVICE)
            mask = mask.to(DEVICE)

            logits = model(image)

            probabilities = torch.sigmoid(logits)[0, 0]

            prediction = (
                probabilities >= VALIDATION_THRESHOLD
            ).float()

            intersection_soft = (
                probabilities * mask[0]
            ).sum()

            soft_dice = (
                2 * intersection_soft + 1e-6
            ) / (
                probabilities.sum()
                + mask.sum()
                + 1e-6
            )

            intersection = (
                prediction * mask[0]
            ).sum()

            union = (
                prediction + mask[0]
            ).clamp(0, 1).sum()

            hard_dice = (
                2 * intersection + 1e-6
            ) / (
                prediction.sum()
                + mask.sum()
                + 1e-6
            )

            iou = (
                intersection + 1e-6
            ) / (
                union + 1e-6
            )

            soft_dices.append(soft_dice.item())
            hard_dices.append(hard_dice.item())
            ious.append(iou.item())

    return (
        sum(soft_dices) / len(soft_dices),
        sum(hard_dices) / len(hard_dices),
        sum(ious) / len(ious)
    )


# ==================================================
# LOAD CHECKPOINT
# ==================================================

print("=" * 60)
print("HARD-NEGATIVE FINE-TUNING")
print("=" * 60)

print("\nLoading original epoch-17 checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

model = ResidualUNet(
    in_channels=1,
    out_channels=1
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])

print("Starting epoch:", checkpoint["epoch"])
print(
    "Original best validation Dice:",
    checkpoint["best_val_soft_dice"]
)


# ==================================================
# LOAD DATA
# ==================================================

print("\nLoading training data...")

train_base = StenosisDataset("train")
val_dataset = StenosisDataset("val")

with open(CANDIDATES_PATH, "r") as file:
    candidates = json.load(file)

train_dataset = HardNegativePatchDataset(
    train_base,
    candidates
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("Training images:", len(train_base))
print("Validation images:", len(val_dataset))
print("Hard-negative candidates:", len(candidates))
print("Training patches per epoch:", len(train_dataset))


# ==================================================
# OPTIMIZER
# ==================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# ==================================================
# TRAINING
# ==================================================

best_dice = -1.0
epochs_without_improvement = 0

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0

    print("\n" + "=" * 60)
    print(f"EPOCH {epoch}/{EPOCHS}")
    print("=" * 60)

    for batch_index, (images, masks, weights) in enumerate(
        train_loader
    ):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        weights = weights.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)

        loss = calculate_loss(
            logits,
            masks,
            weights
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        if (batch_index + 1) % 100 == 0:

            print(
                f"Batch {batch_index + 1}/"
                f"{len(train_loader)} | "
                f"Loss: {loss.item():.5f}"
            )

    average_loss = total_loss / len(train_loader)

    print("\nTraining loss:", round(average_loss, 6))

    # Validation
    print("Evaluating validation dataset...")

    soft_dice, hard_dice, iou = evaluate(
        model,
        val_dataset
    )

    scheduler.step(soft_dice)

    current_lr = optimizer.param_groups[0]["lr"]

    print("\nVALIDATION RESULTS")
    print("-------------------------")
    print("Soft Dice:", round(soft_dice, 6))
    print(
        f"Hard Dice (threshold {VALIDATION_THRESHOLD}):",
        round(hard_dice, 6)
    )
    print("IoU:", round(iou, 6))
    print("Learning rate:", current_lr)

    # Save only the best checkpoint.
    if soft_dice > best_dice:

        best_dice = soft_dice
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_soft_dice": best_dice,
                "val_hard_dice": hard_dice,
                "val_iou": iou,
                "validation_threshold": VALIDATION_THRESHOLD,
                "source_checkpoint": str(CHECKPOINT_PATH),
                "training_method": "boundary_aware_finetuning"
            },
            OUTPUT_PATH
        )

        print("\nBEST CHECKPOINT SAVED!")

    else:

        epochs_without_improvement += 1

        print(
            "No improvement:",
            epochs_without_improvement,
            "/",
            EARLY_STOPPING_PATIENCE
        )

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

        print("\nEarly stopping triggered.")
        break


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("FINE-TUNING COMPLETED")
print("=" * 60)

print("Best validation soft Dice:", best_dice)
print("Checkpoint saved at:", OUTPUT_PATH)

print("Original checkpoint remains untouched.")
