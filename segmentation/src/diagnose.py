
import torch

from src.dataset import StenosisDataset
from src.unet import UNet


DEVICE = torch.device("cpu")
CHECKPOINT = "outputs/best_stenosis_unet.pth"

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def main():
    dataset = StenosisDataset("val")

    model = UNet().to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dice_sums = {t: 0.0 for t in THRESHOLDS}
    foreground_sums = {t: 0.0 for t in THRESHOLDS}
    actual_foreground_sum = 0.0
    probability_sum = 0.0

    print(f"Validation images: {len(dataset)}")
    print("Evaluating thresholds...\n")

    with torch.no_grad():
        for index in range(len(dataset)):
            image, mask = dataset[index]

            image = image.unsqueeze(0).to(DEVICE)
            mask = mask.unsqueeze(0).to(DEVICE)

            probabilities = torch.sigmoid(model(image))

            actual_foreground_sum += mask.mean().item()
            probability_sum += probabilities.mean().item()

            for threshold in THRESHOLDS:
                prediction = (
                    probabilities >= threshold
                ).float()

                intersection = (prediction * mask).sum().item()
                pred_count = prediction.sum().item()
                mask_count = mask.sum().item()

                denominator = pred_count + mask_count

                dice = (
                    (2.0 * intersection) / denominator
                    if denominator > 0
                    else 1.0
                )

                dice_sums[threshold] += dice
                foreground_sums[threshold] += (
                    pred_count / prediction.numel()
                )

            if (index + 1) % 50 == 0:
                print(f"Processed {index + 1}/{len(dataset)}")

    count = len(dataset)

    print("\nValidation summary")
    print(f"Mean actual foreground: {100 * actual_foreground_sum / count:.4f}%")
    print(f"Mean probability:        {probability_sum / count:.6f}")

    print("\nThreshold | Mean Dice | Predicted foreground")
    print("-" * 48)

    for threshold in THRESHOLDS:
        mean_dice = dice_sums[threshold] / count
        mean_foreground = 100 * foreground_sums[threshold] / count

        print(
            f"{threshold:9.2f} | "
            f"{mean_dice:9.4f} | "
            f"{mean_foreground:18.4f}%"
        )


if __name__ == "__main__":
    main()
