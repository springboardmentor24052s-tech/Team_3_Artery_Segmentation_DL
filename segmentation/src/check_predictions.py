
import torch
from torch.utils.data import DataLoader

from src.dataset import StenosisDataset
from src.unet import UNet


def main():
    device = torch.device("cpu")

    checkpoint = torch.load(
        "outputs/unet_baseline_best.pth",
        map_location=device,
    )

    model = UNet().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = StenosisDataset("val")
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    thresholds = [0.1, 0.2, 0.3, 0.5]
    dice_sums = {t: 0.0 for t in thresholds}
    predicted_pixels = {t: 0.0 for t in thresholds}

    total_images = 0
    actual_foreground = 0.0
    mean_probability = 0.0

    with torch.no_grad():
        for images, masks in loader:
            logits = model(images.to(device))
            probabilities = torch.sigmoid(logits)

            actual_foreground += masks.mean().item() * len(images)
            mean_probability += probabilities.mean().item() * len(images)
            total_images += len(images)

            for threshold in thresholds:
                predictions = (probabilities >= threshold).float()

                pred_flat = predictions.flatten(start_dim=1)
                mask_flat = masks.flatten(start_dim=1)

                intersection = (pred_flat * mask_flat).sum(dim=1)
                denominator = pred_flat.sum(dim=1) + mask_flat.sum(dim=1)

                dice = (2 * intersection + 1.0) / (denominator + 1.0)

                dice_sums[threshold] += dice.sum().item()
                predicted_pixels[threshold] += (
                    predictions.mean().item() * len(images)
                )

    print("\nValidation prediction diagnostics")
    print("---------------------------------")
    print("Validation images:", total_images)
    print(f"Actual foreground fraction: {actual_foreground / total_images:.6f}")
    print(f"Mean predicted probability: {mean_probability / total_images:.6f}")
    print("\nThreshold | Predicted foreground | Mean Dice")

    for threshold in thresholds:
        foreground = predicted_pixels[threshold] / total_images
        dice = dice_sums[threshold] / total_images

        print(f"{threshold:9.2f} | {foreground:20.6f} | {dice:.6f}")


if __name__ == "__main__":
    main()
