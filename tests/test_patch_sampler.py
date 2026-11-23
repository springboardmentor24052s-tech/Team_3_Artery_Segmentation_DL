
import torch

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.train import PatchDataset, PATCH_SIZE

# Load training dataset
base_dataset = StenosisDataset("train")

# Create patch dataset
patch_dataset = PatchDataset(base_dataset)

print("Total available patches:", len(patch_dataset))

positive_count = 0
negative_count = 0
total_patches = 100

for i in range(total_patches):

    image, mask = patch_dataset[i]

    # Check dimensions
    assert image.shape == (1, PATCH_SIZE, PATCH_SIZE)
    assert mask.shape == (1, PATCH_SIZE, PATCH_SIZE)

    # Check valid values
    assert torch.isfinite(image).all()
    assert torch.isfinite(mask).all()

    if mask.sum().item() > 0:
        positive_count += 1
    else:
        negative_count += 1

print("\nPATCH SAMPLING RESULTS")
print("----------------------")
print("Positive patches:", positive_count)
print("Negative patches:", negative_count)

print(
    "Actual positive percentage:",
    round(positive_count / total_patches * 100, 2), "%"
)

print("\nPATCH SAMPLER TEST COMPLETED!")
