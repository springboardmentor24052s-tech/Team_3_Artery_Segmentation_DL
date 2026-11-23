
import torch
import numpy as np
from torch.utils.data import DataLoader

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model import ResidualUNet


DEVICE = torch.device("cpu")

CHECKPOINT_PATH = (
    "stenosis_detection/checkpoints/resunet_stenosis_best.pt"
)

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main():

    print("Loading validation dataset...")

    dataset = StenosisDataset("val")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = ResidualUNet().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    results = {
        threshold: {
            "dice": [],
            "iou": [],
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "predicted_pixels": 0
        }
        for threshold in THRESHOLDS
    }

    print("Evaluating all validation images...")

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            probabilities = torch.sigmoid(model(images))

            for threshold in THRESHOLDS:

                predictions = (
                    probabilities >= threshold
                ).float()

                intersection = (
                    predictions * masks
                ).sum().item()

                pred_sum = predictions.sum().item()
                mask_sum = masks.sum().item()

                union = pred_sum + mask_sum - intersection

                dice = (
                    2 * intersection + 1e-6
                ) / (
                    pred_sum + mask_sum + 1e-6
                )

                iou = (
                    intersection + 1e-6
                ) / (
                    union + 1e-6
                )

                results[threshold]["dice"].append(dice)
                results[threshold]["iou"].append(iou)

                results[threshold]["tp"] += intersection
                results[threshold]["fp"] += pred_sum - intersection
                results[threshold]["fn"] += mask_sum - intersection

                results[threshold]["predicted_pixels"] += pred_sum

    print("\nTHRESHOLD COMPARISON")
    print("=" * 85)

    print(
        f"{'Threshold':<12}"
        f"{'Dice':<14}"
        f"{'IoU':<14}"
        f"{'Precision':<14}"
        f"{'Recall':<14}"
        f"{'Predicted %':<14}"
    )

    best_threshold = None
    best_dice = -1

    total_pixels = len(dataset) * 512 * 512

    for threshold in THRESHOLDS:

        data = results[threshold]

        mean_dice = np.mean(data["dice"])
        mean_iou = np.mean(data["iou"])

        precision = (
            data["tp"] / (data["tp"] + data["fp"] + 1e-6)
        )

        recall = (
            data["tp"] / (data["tp"] + data["fn"] + 1e-6)
        )

        predicted_percentage = (
            data["predicted_pixels"] / total_pixels
        ) * 100

        print(
            f"{threshold:<12.2f}"
            f"{mean_dice:<14.4f}"
            f"{mean_iou:<14.4f}"
            f"{precision:<14.4f}"
            f"{recall:<14.4f}"
            f"{predicted_percentage:<14.3f}"
        )

        if mean_dice > best_dice:
            best_dice = mean_dice
            best_threshold = threshold

    print("\nBEST THRESHOLD:", best_threshold)
    print("BEST HARD DICE:", round(best_dice, 6))


if __name__ == "__main__":
    main()
