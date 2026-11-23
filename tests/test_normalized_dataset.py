from stenosis_detection.dataset import StenosisDataset
import torch

for split in ["train", "val", "test"]:

    dataset = StenosisDataset(split)

    image, mask = dataset[0]

    print("\n", split.upper())
    print("-------------------------")
    print("Dataset size:", len(dataset))
    print("Image shape:", image.shape)
    print("Mask shape:", mask.shape)
    print("Image range:", image.min().item(), image.max().item())
    print("Mask values:", torch.unique(mask).tolist())