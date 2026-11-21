
import torch
from torch.utils.data import DataLoader

from src.dataset import StenosisDataset
from src.unet import UNet


def main():
    device = torch.device("cpu")

    dataset = StenosisDataset("val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = UNet().to(device)
    model.load_state_dict(
        torch.load(
            "outputs/overfit_test.pth",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    total_dice = 0.0

    with torch.no_grad():
        for i, (images, masks) in enumerate(loader, start=1):
            logits = model(images.to(device))
            predictions = (torch.sigmoid(logits) >= 0.5).float()

            intersection = (predictions * masks).sum().item()
            denominator = predictions.sum().item() + masks.sum().item()
            dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
            total_dice += dice

            if i % 50 == 0:
                print(f"Processed {i}/{len(loader)} validation images")

    print(f"Validation images: {len(loader)}")
    print(f"Mean validation Dice: {total_dice / len(loader):.4f}")


if __name__ == "__main__":
    main()
