
import torch
from torch.utils.data import DataLoader

from src.dataset import StenosisDataset
from src.unet import UNet
from src.attention_unet import AttentionUNet


THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]

MODELS = {
    "Baseline U-Net": {
        "class": UNet,
        "checkpoint": "outputs/unet_baseline_round2_best.pth",
    },
    "Attention U-Net": {
        "class": AttentionUNet,
        "checkpoint": "outputs/attention_unet_round2_best.pth",
    },
}


def evaluate_model(model, loader, dataset_size, device):
    totals = {
        t: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "dice": 0.0}
        for t in THRESHOLDS
    }

    model.eval()

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            targets = masks.to(device) >= 0.5

            probabilities = torch.sigmoid(model(images))

            for threshold in THRESHOLDS:
                predictions = probabilities >= threshold
                row = totals[threshold]

                row["tp"] += (predictions & targets).sum().item()
                row["fp"] += (predictions & ~targets).sum().item()
                row["fn"] += (~predictions & targets).sum().item()
                row["tn"] += (~predictions & ~targets).sum().item()

                for i in range(images.shape[0]):
                    pred = predictions[i]
                    target = targets[i]

                    intersection = (pred & target).sum().item()
                    denominator = pred.sum().item() + target.sum().item()

                    row["dice"] += (
                        (2 * intersection + 1) / (denominator + 1)
                    )

    results = {}

    for threshold, row in totals.items():
        results[threshold] = {
            "dice": row["dice"] / dataset_size,
            "precision": row["tp"] / max(row["tp"] + row["fp"], 1),
            "recall": row["tp"] / max(row["tp"] + row["fn"], 1),
            "specificity": row["tn"] / max(row["tn"] + row["fp"], 1),
        }

    return results


def main():
    device = torch.device("cpu")

    print("Device:", device)
    print("Loading validation dataset...")

    dataset = StenosisDataset("val")

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    print("Validation images:", len(dataset))

    all_results = {}

    for name, config in MODELS.items():
        checkpoint_path = config["checkpoint"]

        print(f"\nEvaluating {name}...")
        print("Checkpoint:", checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

        print("Training epoch:", checkpoint.get("epoch", "Unknown"))
        print("Checkpoint validation Dice:", checkpoint.get("val_dice", "Unknown"))

        model = config["class"]().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        results = evaluate_model(
            model, loader, len(dataset), device
        )

        all_results[name] = results

        print("\nThreshold | Mean Dice | Precision | Recall | Specificity")

        for threshold, metrics in results.items():
            print(
                f"{threshold:9.2f} | "
                f"{metrics['dice']:9.4f} | "
                f"{metrics['precision']:9.4f} | "
                f"{metrics['recall']:6.4f} | "
                f"{metrics['specificity']:11.4f}"
            )

        del model

    print("\n" + "=" * 85)
    print("FINAL MODEL COMPARISON")
    print("=" * 85)

    print(
        "\nModel             | Threshold | Mean Dice | Precision | Recall | Specificity"
    )
    print("-" * 85)

    for name, results in all_results.items():
        best_threshold = max(
            results,
            key=lambda t: results[t]["dice"]
        )

        metrics = results[best_threshold]

        print(
            f"{name:17} | {best_threshold:9.2f} | "
            f"{metrics['dice']:.4f}    | "
            f"{metrics['precision']:.4f}    | "
            f"{metrics['recall']:.4f} | "
            f"{metrics['specificity']:.4f}"
        )

    print("\nComparison completed.")


if __name__ == "__main__":
    main()
