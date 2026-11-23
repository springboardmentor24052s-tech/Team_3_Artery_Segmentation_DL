
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from scipy import ndimage

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model import ResidualUNet


DEVICE = torch.device("cpu")

CHECKPOINT_PATH = (
    "stenosis_detection/checkpoints/resunet_stenosis_best.pt"
)

THRESHOLD = 0.80

# Test different minimum connected-component sizes.
MIN_SIZES = [1, 5, 10, 20, 40, 80, 120]


def calculate_metrics(prediction, mask):

    tp = np.logical_and(prediction, mask).sum()
    fp = np.logical_and(prediction, ~mask).sum()
    fn = np.logical_and(~prediction, mask).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)

    return tp, fp, fn, precision, recall, dice, iou


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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    structure = np.ones((3, 3), dtype=np.uint8)

    total_components = 0
    isolated_components = 0
    mixed_components = 0

    component_sizes = []

    # Store predictions for testing different filters.
    image_results = []

    print("Analyzing connected components...")

    with torch.no_grad():

        for index, (image, mask) in enumerate(loader):

            image = image.to(DEVICE)

            probabilities = torch.sigmoid(model(image))

            prediction = (
                probabilities >= THRESHOLD
            ).cpu().numpy()[0, 0].astype(bool)

            ground_truth = (
                mask.numpy()[0, 0] >= 0.5
            )

            # Label separate predicted regions.
            labels, num_components = ndimage.label(
                prediction,
                structure=structure
            )

            component_sizes_for_image = []

            for component_id in range(1, num_components + 1):

                component = labels == component_id

                size = int(component.sum())

                overlap = np.logical_and(
                    component,
                    ground_truth
                ).sum()

                component_sizes.append(size)

                component_sizes_for_image.append(
                    (size, int(overlap))
                )

                total_components += 1

                if overlap == 0:
                    isolated_components += 1
                else:
                    mixed_components += 1

            image_results.append({
                "prediction": prediction,
                "mask": ground_truth,
                "labels": labels,
                "components": component_sizes_for_image
            })

    print("\nCONNECTED COMPONENT SUMMARY")
    print("---------------------------------")

    print("Validation images:", len(dataset))
    print("Total predicted components:", total_components)
    print("Components with no GT overlap:", isolated_components)
    print("Components with GT overlap:", mixed_components)

    sizes = np.array(component_sizes)

    print("\nCOMPONENT SIZE DISTRIBUTION")
    print("---------------------------------")

    print("Smallest component:", sizes.min())
    print("Median component size:", np.median(sizes))
    print("Mean component size:", sizes.mean())
    print("Largest component:", sizes.max())

    for limit in [5, 10, 20, 40, 80, 120]:

        count = int((sizes < limit).sum())

        percentage = (
            count / max(len(sizes), 1)
        ) * 100

        print(
            f"Components smaller than {limit:3d} pixels: "
            f"{count:6d} ({percentage:.2f}%)"
        )

    # Simulate removing small components.
    print("\nFILTERING EXPERIMENT")
    print("=" * 85)

    print(
        f"{'Min Size':<12}"
        f"{'Dice':<12}"
        f"{'IoU':<12}"
        f"{'Precision':<14}"
        f"{'Recall':<12}"
        f"{'FP Pixels':<14}"
        f"{'FN Pixels':<12}"
    )

    for min_size in MIN_SIZES:

        total_tp = 0
        total_fp = 0
        total_fn = 0

        image_dices = []
        image_ious = []

        for item in image_results:

            labels = item["labels"]
            mask = item["mask"]

            # Keep components that meet the minimum size.
            counts = np.bincount(labels.ravel())

            keep = counts >= min_size
            keep[0] = False

            filtered_prediction = keep[labels]

            tp, fp, fn, precision, recall, dice, iou = (
                calculate_metrics(filtered_prediction, mask)
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn

            image_dices.append(dice)
            image_ious.append(iou)

        global_precision = (
            total_tp / (total_tp + total_fp + 1e-6)
        )

        global_recall = (
            total_tp / (total_tp + total_fn + 1e-6)
        )

        mean_dice = np.mean(image_dices)
        mean_iou = np.mean(image_ious)

        print(
            f"{min_size:<12}"
            f"{mean_dice:<12.4f}"
            f"{mean_iou:<12.4f}"
            f"{global_precision:<14.4f}"
            f"{global_recall:<12.4f}"
            f"{total_fp:<14}"
            f"{total_fn:<12}"
        )

    # Save component-size distribution.
    plt.figure(figsize=(9, 5))

    plt.hist(
        sizes,
        bins=50,
        log=True
    )

    plt.xlabel("Connected component size (pixels)")
    plt.ylabel("Number of components (log scale)")
    plt.title("Predicted Stenosis Component Size Distribution")

    plt.tight_layout()

    plt.savefig(
        "stenosis_component_distribution.png",
        dpi=200
    )

    print("\nSaved: stenosis_component_distribution.png")


if __name__ == "__main__":
    main()
