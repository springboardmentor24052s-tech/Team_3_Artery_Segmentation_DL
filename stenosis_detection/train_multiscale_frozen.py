
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model_multiscale import MultiScaleResidualUNet


# ==================================================
# SETTINGS
# ==================================================

CHECKPOINT_PATH = Path(
    "stenosis_detection/checkpoints/"
    "resunet_multiscale_best.pt"
)

CANDIDATES_PATH = Path(
    "stenosis_detection/hard_negative_candidates.json"
)

OUTPUT_PATH = Path(
    "stenosis_detection/checkpoints/"
    "resunet_multiscale_frozen_finetuned_best.pt"
)

PATCH_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 2

LEARNING_RATE = 0.00001
EARLY_STOPPING_PATIENCE = 2

HARD_NEGATIVE_WEIGHT = 0.5
VALIDATION_THRESHOLD = 0.90

DEVICE = torch.device("cpu")

random.seed(42)
torch.manual_seed(42)
torch.set_num_threads(4)


# ==================================================
# SAFETY CHECKS
# ==================================================

if OUTPUT_PATH.exists():
    raise FileExistsError(
        f"Output checkpoint already exists:\n{OUTPUT_PATH}\n"
        "Do not overwrite it. Rename it before running again."
    )

for path in [CHECKPOINT_PATH, CANDIDATES_PATH]:
    if not path.exists():
        raise FileNotFoundError(path)


# ==================================================
# PATCH DATASET
# Same sampling and augmentation as original training
# ==================================================

class HardNegativePatchDataset(Dataset):

    def __init__(self, base_dataset, candidates):
        self.base = base_dataset
        self.candidates = candidates

    def __len__(self):
        return len(self.base)

    def crop(self, image, mask, x, y):
        return (
            image[:, y:y + PATCH_SIZE, x:x + PATCH_SIZE],
            mask[:, y:y + PATCH_SIZE, x:x + PATCH_SIZE]
        )

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

            selected = coordinates[random.randrange(len(coordinates))]
            cy, cx = int(selected[0]), int(selected[1])

            x = max(0, min(cx - PATCH_SIZE // 2, 512 - PATCH_SIZE))
            y = max(0, min(cy - PATCH_SIZE // 2, 512 - PATCH_SIZE))

            return self.crop(image, mask, x, y), 1.0

        raise RuntimeError("Could not find a positive training patch.")

    def hard_negative_patch(self):

        candidate = random.choice(self.candidates)
        index = candidate["image_index"]

        image, mask = self.base[index]

        patch = self.crop(
            image, mask, candidate["x"], candidate["y"]
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

        return image, mask, torch.tensor(weight, dtype=torch.float32)


# ==================================================
# SAME BOUNDARY-AWARE LOSS
# ==================================================

def calculate_loss(logits, targets, weights):

    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    ).mean(dim=(1, 2, 3))

    probabilities = torch.sigmoid(logits)

    intersection = (probabilities * targets).sum(dim=(1, 2, 3))

    denominator = (
        probabilities.sum(dim=(1, 2, 3))
        + targets.sum(dim=(1, 2, 3))
    )

    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    dice_loss = 1.0 - dice

    def extract_boundary(tensor):

        dilated = F.max_pool2d(
            tensor, kernel_size=3, stride=1, padding=1
        )

        eroded = -F.max_pool2d(
            -tensor, kernel_size=3, stride=1, padding=1
        )

        return (dilated - eroded).clamp(0, 1)

    predicted_boundary = extract_boundary(probabilities)
    target_boundary = extract_boundary(targets)

    boundary_loss = torch.abs(
        predicted_boundary - target_boundary
    ).mean(dim=(1, 2, 3))

    sample_loss = (
        0.45 * bce
        + 0.45 * dice_loss
        + 0.10 * boundary_loss
    )

    return (sample_loss * weights).mean()


# ==================================================
# GLOBAL + MACRO VALIDATION
# Same pixel-level calculation as comparison script
# ==================================================

def evaluate(model, dataset):

    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    macro_dice = []

    with torch.no_grad():

        for index in range(len(dataset)):

            image, mask = dataset[index]

            image = image.unsqueeze(0).to(DEVICE)
            mask = mask[0].to(DEVICE)

            probabilities = torch.sigmoid(
                model(image)[0, 0]
            )

            prediction = (
                probabilities >= VALIDATION_THRESHOLD
            ).float()

            tp = (prediction * mask).sum().item()
            fp = (prediction * (1 - mask)).sum().item()
            fn = ((1 - prediction) * mask).sum().item()

            dice = (
                2 * tp + 1e-6
            ) / (
                2 * tp + fp + fn + 1e-6
            )

            macro_dice.append(dice)

            total_tp += tp
            total_fp += fp
            total_fn += fn

    global_dice = (
        2 * total_tp + 1e-6
    ) / (
        2 * total_tp + total_fp + total_fn + 1e-6
    )

    global_iou = (
        total_tp + 1e-6
    ) / (
        total_tp + total_fp + total_fn + 1e-6
    )

    precision = (
        total_tp + 1e-6
    ) / (
        total_tp + total_fp + 1e-6
    )

    recall = (
        total_tp + 1e-6
    ) / (
        total_tp + total_fn + 1e-6
    )

    return {
        "global_dice": global_dice,
        "global_iou": global_iou,
        "precision": precision,
        "recall": recall,
        "macro_dice": sum(macro_dice) / len(macro_dice)
    }


# ==================================================
# LOAD MODEL AND FREEZE SELECTED LAYERS
# ==================================================

print("=" * 65)
print("MULTISCALE SELECTIVE-FREEZE FINE-TUNING")
print("=" * 65)

model = MultiScaleResidualUNet(
    in_channels=1,
    out_channels=1
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])

print("\nLoaded existing MultiScale checkpoint.")
print("Source epoch:", checkpoint.get("epoch", "Unknown"))

# Freeze the first two encoder blocks.
for parameter in model.enc1.parameters():
    parameter.requires_grad = False

for parameter in model.enc2.parameters():
    parameter.requires_grad = False

# All other layers remain trainable.
trainable_parameters = [
    parameter for parameter in model.parameters()
    if parameter.requires_grad
]

print("\nFROZEN LAYERS")
print("-" * 40)
print("enc1: Frozen")
print("enc2: Frozen")

print("\nTRAINABLE LAYERS")
print("-" * 40)

for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        print(name)

print(
    "\nTrainable parameters:",
    sum(p.numel() for p in trainable_parameters)
)


# ==================================================
# LOAD DATA
# ==================================================

print("\nLoading datasets...")

train_base = StenosisDataset("train")
val_dataset = StenosisDataset("val")

with open(CANDIDATES_PATH, "r") as file:
    candidates = json.load(file)

train_dataset = HardNegativePatchDataset(
    train_base, candidates
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print("\nTraining images:", len(train_base))
print("Validation images:", len(val_dataset))
print("Hard-negative candidates:", len(candidates))


# ==================================================
# BASELINE VALIDATION
# ==================================================

print("\nEvaluating original MultiScale checkpoint...")

baseline = evaluate(model, val_dataset)

print("\nBASELINE RESULTS")
print("-" * 45)

for key, value in baseline.items():
    print(f"{key}: {value:.6f}")

best_dice = baseline["global_dice"]
best_epoch = 0
epochs_without_improvement = 0


# ==================================================
# OPTIMIZER
# ==================================================

optimizer = torch.optim.Adam(
    trainable_parameters,
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)


# ==================================================
# TRAINING
# ==================================================

for epoch in range(1, EPOCHS + 1):

    model.train()

    total_loss = 0.0

    print("\n" + "=" * 65)
    print(f"EPOCH {epoch}/{EPOCHS}")
    print("=" * 65)

    for batch_index, (images, masks, weights) in enumerate(train_loader):

        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        weights = weights.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)

        loss = calculate_loss(logits, masks, weights)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        if (batch_index + 1) % 100 == 0:
            print(
                f"Batch {batch_index + 1}/{len(train_loader)} "
                f"| Loss: {loss.item():.5f}"
            )

    average_loss = total_loss / len(train_loader)

    print("\nAverage training loss:", round(average_loss, 6))
    print("Evaluating validation dataset...")

    metrics = evaluate(model, val_dataset)

    scheduler.step(metrics["global_dice"])

    print("\nVALIDATION RESULTS")
    print("-" * 45)

    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    print("Learning rate:", optimizer.param_groups[0]["lr"])

    # Save only if global Dice beats the original baseline.
    if metrics["global_dice"] > best_dice:

        best_dice = metrics["global_dice"]
        best_epoch = epoch
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_global_dice": best_dice,
                "validation_metrics": metrics,
                "validation_threshold": VALIDATION_THRESHOLD,
                "source_checkpoint": str(CHECKPOINT_PATH),
                "frozen_layers": ["enc1", "enc2"],
                "training_method": "selective_freeze_multiscale_finetuning"
            },
            OUTPUT_PATH
        )

        print("\nIMPROVED MODEL! New checkpoint saved.")

    else:

        epochs_without_improvement += 1

        print(
            "\nNo improvement over baseline.",
            epochs_without_improvement,
            "/", EARLY_STOPPING_PATIENCE
        )

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
        print("\nEarly stopping triggered.")
        break


# ==================================================
# FINAL SUMMARY
# ==================================================

print("\n" + "=" * 65)
print("FINE-TUNING COMPLETED")
print("=" * 65)

print(f"Original global Dice: {baseline['global_dice']:.6f}")
print(f"Best global Dice:     {best_dice:.6f}")
print("Best epoch:", best_epoch)

if best_epoch > 0:
    print("\nImprovement:", round(best_dice - baseline["global_dice"], 6))
    print("New checkpoint:", OUTPUT_PATH)
else:
    print("\nNo improvement over the original model.")
    print("Keep using the original MultiScale checkpoint.")

print("\nOriginal checkpoint remains untouched.")
