
import torch
from torch.utils.data import DataLoader, Subset

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model import ResidualUNet
from stenosis_detection.train import (
    PatchDataset,
    CombinedLoss,
    validate
)


print("1. Loading datasets...")

train_base = StenosisDataset("train")
val_dataset = StenosisDataset("val")

train_dataset = PatchDataset(
    train_base,
    patches_per_image=2
)

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0
)

print("2. Creating model and loss...")

model = ResidualUNet()
criterion = CombinedLoss()

print("3. Testing one training batch...")

images, masks = next(iter(train_loader))

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

optimizer.zero_grad()

outputs = model(images)

loss = criterion(outputs, masks)

loss.backward()

optimizer.step()

assert torch.isfinite(loss).item()

print("Training batch passed!")
print("Image shape:", images.shape)
print("Mask shape:", masks.shape)
print("Output shape:", outputs.shape)
print("Loss:", round(loss.item(), 4))

print("4. Testing validation...")

# Use a small subset for the smoke test
val_subset = Subset(
    val_dataset,
    range(2)
)

val_loader = DataLoader(
    val_subset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

soft_dice, hard_dice, iou = validate(
    model,
    val_loader
)

print("Validation passed!")
print("Validation Soft Dice:", round(soft_dice, 4))
print("Validation Hard Dice:", round(hard_dice, 4))
print("Validation IoU:", round(iou, 4))

print("\nALL TRAINING PIPELINE TESTS PASSED!")
