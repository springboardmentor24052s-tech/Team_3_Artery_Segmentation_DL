
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import distance_transform_edt, binary_erosion

from src.dataset import StenosisDataset
from src.unet import UNet
from src.attention_unet import AttentionUNet


MODELS = [
    {
        "name": "Baseline U-Net",
        "class": UNet,
        "checkpoint": "outputs/unet_baseline_round2_best.pth",
        "threshold": 0.40,
    },
    {
        "name": "Attention U-Net",
        "class": AttentionUNet,
        "checkpoint": "outputs/attention_unet_round2_best.pth",
        "threshold": 0.30,
    },
]


def load_model(model_class, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = model_class().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Training epoch: {checkpoint['epoch']}")
    print(f"Saved validation Dice: {checkpoint['val_dice']:.4f}")

    return model


def surface_distances(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)

    if not pred.any() and not target.any():
        return 0.0, 0.0

    if not pred.any() or not target.any():
        penalty = float(np.hypot(*pred.shape))
        return penalty, penalty

    pred_surface = pred ^ binary_erosion(pred)
    target_surface = target ^ binary_erosion(target)

    distance_to_target = distance_transform_edt(~target_surface)
    distance_to_pred = distance_transform_edt(~pred_surface)

    pred_to_target = distance_to_target[pred_surface]
    target_to_pred = distance_to_pred[target_surface]

    hausdorff = max(
        float(pred_to_target.max()),
        float(target_to_pred.max()),
    )

    average_surface = (
        float(pred_to_target.mean())
        + float(target_to_pred.mean())
    ) / 2.0

    return hausdorff, average_surface


def evaluate(model, loader, device, threshold):
    totals = {
        "tp": 0,
        "tn": 0,
        "fp": 0,
        "fn": 0,
        "dice_sum": 0.0,
        "image_count": 0,
        "hausdorff_sum": 0.0,
        "surface_sum": 0.0,
        "distance_count": 0,
    }

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            targets = masks.to(device) >= 0.5

            probabilities = torch.sigmoid(model(images))
            predictions = probabilities >= threshold

            totals["tp"] += (predictions & targets).sum().item()
            totals["tn"] += (~predictions & ~targets).sum().item()
            totals["fp"] += (predictions & ~targets).sum().item()
            totals["fn"] += (~predictions & targets).sum().item()

            for i in range(images.shape[0]):
                pred_np = predictions[i, 0].cpu().numpy()
                target_np = targets[i, 0].cpu().numpy()

                intersection = np.logical_and(
                    pred_np, target_np
                ).sum()

                denominator = pred_np.sum() + target_np.sum()

                dice = (
                    2.0 * intersection + 1.0
                ) / (denominator + 1.0)

                totals["dice_sum"] += dice
                totals["image_count"] += 1

                hd, asd = surface_distances(
                    pred_np, target_np
                )

                totals["hausdorff_sum"] += hd
                totals["surface_sum"] += asd
                totals["distance_count"] += 1

    tp = totals["tp"]
    tn = totals["tn"]
    fp = totals["fp"]
    fn = totals["fn"]

    pixel_accuracy = (
        tp + tn
    ) / max(tp + tn + fp + fn, 1)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    return {
        "Pixel accuracy": pixel_accuracy,
        "Precision": precision,
        "Recall / Sensitivity": recall,
        "Specificity": specificity,
        "Mean image Dice": totals["dice_sum"] / max(
            totals["image_count"], 1
        ),
        "Mean Hausdorff (pixels)": totals["hausdorff_sum"] / max(
            totals["distance_count"], 1
        ),
        "Mean surface distance (pixels)": totals["surface_sum"] / max(
            totals["distance_count"], 1
        ),
    }


def main():
    device = torch.device("cpu")

    dataset = StenosisDataset("val")

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )

    print("Device:", device)
    print("Validation images:", len(dataset))

    for config in MODELS:
        print(f"\n{'=' * 18} {config['name']} {'=' * 18}")
        print("Prediction threshold:", config["threshold"])

        model = load_model(
            config["class"],
            config["checkpoint"],
            device,
        )

        metrics = evaluate(
            model,
            loader,
            device,
            config["threshold"],
        )

        for metric, value in metrics.items():
            print(f"{metric:34s}: {value:.6f}")

        del model

    print("\nComparison completed.")


if __name__ == "__main__":
    main()
