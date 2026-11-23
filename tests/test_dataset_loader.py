
import torch
from stenosis_detection.dataset import StenosisDataset

for split in ["train", "val", "test"]:

    print("\n" + "=" * 40)
    print("TESTING:", split.upper())
    print("=" * 40)

    dataset = StenosisDataset(split=split)

    print("Total images:", len(dataset))

    # Test the first image
    image, mask = dataset[0]

    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)

    print("Image minimum:", image.min().item())
    print("Image maximum:", image.max().item())

    print("Mask unique values:", torch.unique(mask).tolist())

    print("Image dtype:", image.dtype)
    print("Mask dtype:", mask.dtype)

    print("Stenosis pixels:", mask.sum().item())

    assert image.shape == (1, 512, 512)
    assert mask.shape == (1, 512, 512)
    assert image.min() >= 0 and image.max() <= 1
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})

    print("Dataset test PASSED!")

# Check an image without stenosis annotations
train_dataset = StenosisDataset(split="train")

empty_index = next(
    i for i, img in enumerate(train_dataset.images)
    if not train_dataset.annotations_by_image[img["id"]]
)

empty_image, empty_mask = train_dataset[empty_index]

print("\nEmpty-mask test:")
print("Image:", train_dataset.images[empty_index]["file_name"])
print("Mask pixels:", empty_mask.sum().item())

assert empty_mask.sum().item() == 0

print("Empty-mask test PASSED!")
print("\nALL TESTS COMPLETED!")
