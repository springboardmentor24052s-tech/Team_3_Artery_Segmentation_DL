
import random
import torch
import matplotlib.pyplot as plt

from stenosis_detection.dataset import StenosisDataset


PATCH_SIZE = 256

# Load training dataset
dataset = StenosisDataset("train")

# Find an image containing stenosis
positive_indices = []

for i in range(len(dataset)):
    image, mask = dataset[i]

    if mask.sum() > 0:
        positive_indices.append(i)

# Select one positive image
index = random.choice(positive_indices)
image, mask = dataset[index]

# Find a positive pixel
positive_pixels = torch.nonzero(mask[0] > 0)

selected = positive_pixels[
    random.randint(0, len(positive_pixels) - 1)
]

y, x = selected.tolist()

# Crop a patch containing the positive pixel
top = max(0, min(y - PATCH_SIZE // 2, 512 - PATCH_SIZE))
left = max(0, min(x - PATCH_SIZE // 2, 512 - PATCH_SIZE))

image_patch = image[
    :, top:top + PATCH_SIZE, left:left + PATCH_SIZE
].clone()

mask_patch = mask[
    :, top:top + PATCH_SIZE, left:left + PATCH_SIZE
].clone()

# Apply the same augmentation operations used in training
aug_image = image_patch.clone()
aug_mask = mask_patch.clone()

if random.random() < 0.5:
    aug_image = torch.flip(aug_image, dims=[2])
    aug_mask = torch.flip(aug_mask, dims=[2])

if random.random() < 0.5:
    aug_image = torch.flip(aug_image, dims=[1])
    aug_mask = torch.flip(aug_mask, dims=[1])

if random.random() < 0.5:
    k = random.choice([1, 2, 3])
    aug_image = torch.rot90(aug_image, k, dims=[1, 2])
    aug_mask = torch.rot90(aug_mask, k, dims=[1, 2])

# Display original and augmented patches
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

def show_patch(ax, img, title):
    ax.imshow(img[0].numpy(), cmap="gray")
    ax.set_title(title)
    ax.axis("off")

def show_overlay(ax, img, msk, title):
    ax.imshow(img[0].numpy(), cmap="gray")
    ax.imshow(msk[0].numpy(), cmap="Reds", alpha=0.45)
    ax.set_title(title)
    ax.axis("off")

show_patch(axes[0, 0], image_patch, "Original Image")
show_patch(axes[0, 1], mask_patch, "Original Mask")
show_overlay(axes[0, 2], image_patch, mask_patch, "Original Overlay")

show_patch(axes[1, 0], aug_image, "Augmented Image")
show_patch(axes[1, 1], aug_mask, "Augmented Mask")
show_overlay(axes[1, 2], aug_image, aug_mask, "Augmented Overlay")

plt.tight_layout()
plt.show()

print("Selected training image:", dataset.images[index]["file_name"])
print("Original mask pixels:", int(mask_patch.sum()))
print("Augmented mask pixels:", int(aug_mask.sum()))
print("\nAUGMENTATION VISUALIZATION COMPLETED!")
