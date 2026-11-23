
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model import ResidualUNet


DEVICE = torch.device("cpu")

CHECKPOINT_PATH = (
    "stenosis_detection/checkpoints/resunet_stenosis_best.pt"
)

THRESHOLD = 0.80


def main():

    print("Loading validation dataset...")

    dataset = StenosisDataset("val")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print("Loading Epoch 17 checkpoint...")

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

    results = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    print("Analyzing all validation images...")

    with torch.no_grad():

        for index, (image, mask) in enumerate(loader):

            image = image.to(DEVICE)
            mask = mask.to(DEVICE)

            probabilities = torch.sigmoid(model(image))

            prediction = (
                probabilities >= THRESHOLD
            ).float()

            tp = int(((prediction == 1) & (mask == 1)).sum())
            fp = int(((prediction == 1) & (mask == 0)).sum())
            fn = int(((prediction == 0) & (mask == 1)).sum())

            total_tp += tp
            total_fp += fp
            total_fn += fn

            results.append({
                "index": index,
                "image": image[0, 0].cpu().numpy(),
                "mask": mask[0, 0].cpu().numpy(),
                "prediction": prediction[0, 0].cpu().numpy(),
                "tp": tp,
                "fp": fp,
                "fn": fn
            })

    print("\nOVERALL VALIDATION ERROR ANALYSIS")
    print("---------------------------------")
    print("Images:", len(dataset))
    print("True Positives:", total_tp)
    print("False Positives:", total_fp)
    print("False Negatives:", total_fn)

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Select three images with the most false positives
    # and three with the most false negatives.

    highest_fp = sorted(
        results,
        key=lambda x: x["fp"],
        reverse=True
    )[:3]

    highest_fn = sorted(
        results,
        key=lambda x: x["fn"],
        reverse=True
    )[:3]

    selected = []

    for item in highest_fp + highest_fn:
        if item["index"] not in [
            existing["index"] for existing in selected
        ]:
            selected.append(item)

    print("\nSelected images for visualization:")

    for item in selected:
        print(
            f"Image {item['index']}: "
            f"TP={item['tp']}, "
            f"FP={item['fp']}, "
            f"FN={item['fn']}"
        )

    # Generate TP / FP / FN overlays.

    fig, axes = plt.subplots(
        len(selected),
        2,
        figsize=(12, 4 * len(selected))
    )

    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, item in enumerate(selected):

        image = item["image"]
        mask = item["mask"]
        prediction = item["prediction"]

        tp_mask = (prediction == 1) & (mask == 1)
        fp_mask = (prediction == 1) & (mask == 0)
        fn_mask = (prediction == 0) & (mask == 1)

        overlay = np.zeros(
            (*image.shape, 3),
            dtype=np.float32
        )

        overlay[tp_mask] = [0, 1, 0]
        overlay[fp_mask] = [1, 0, 0]
        overlay[fn_mask] = [1, 1, 0]

        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(
            f"Original Image {item['index']}"
        )

        axes[row, 1].imshow(image, cmap="gray")
        axes[row, 1].imshow(overlay, alpha=0.65)
        axes[row, 1].set_title(
            f"TP: {item['tp']} | "
            f"FP: {item['fp']} | "
            f"FN: {item['fn']}"
        )

        axes[row, 0].axis("off")
        axes[row, 1].axis("off")

    plt.tight_layout()

    output_path = "stenosis_error_analysis.png"

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    print("\nVisualization saved:", output_path)

    plt.show()


if __name__ == "__main__":
    main()
