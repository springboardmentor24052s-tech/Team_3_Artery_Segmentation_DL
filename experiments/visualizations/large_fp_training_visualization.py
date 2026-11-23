
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from torch.utils.data import DataLoader

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model import ResidualUNet


# ==========================================
# CONFIGURATION
# ==========================================

DEVICE = torch.device("cpu")

CHECKPOINT_PATH = (
    "stenosis_detection/checkpoints/resunet_stenosis_best.pt"
)

THRESHOLD = 0.80

MIN_COMPONENT_SIZE = 100

MIN_DISTANCE_FROM_GT = 15

MAX_REGIONS = 12

CROP_PADDING = 80

CROP_FOLDER = "training_fp_crops"


# ==========================================
# SAVE INDIVIDUAL FALSE-POSITIVE CROPS
# ==========================================

def save_individual_crops(selected):

    os.makedirs(CROP_FOLDER, exist_ok=True)

    print("\nSaving individual false-positive crops...")

    for rank, item in enumerate(selected, start=1):

        image = item["image"]
        component = item["component"]
        mask = item["mask"]

        ys, xs = np.where(component)

        y_min = max(0, int(ys.min()) - CROP_PADDING)
        y_max = min(
            image.shape[0],
            int(ys.max()) + CROP_PADDING + 1
        )

        x_min = max(0, int(xs.min()) - CROP_PADDING)
        x_max = min(
            image.shape[1],
            int(xs.max()) + CROP_PADDING + 1
        )

        cropped_image = image[y_min:y_max, x_min:x_max]

        cropped_component = component[
            y_min:y_max,
            x_min:x_max
        ]

        cropped_mask = mask[
            y_min:y_max,
            x_min:x_max
        ]

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 5)
        )

        # Original image crop
        axes[0].imshow(
            cropped_image,
            cmap="gray"
        )

        axes[0].set_title("Original Image")

        # Ground truth crop
        axes[1].imshow(
            cropped_mask,
            cmap="gray"
        )

        axes[1].set_title("Ground Truth")

        # False-positive overlay
        axes[2].imshow(
            cropped_image,
            cmap="gray"
        )

        overlay = np.zeros(
            (*cropped_image.shape, 3),
            dtype=np.float32
        )

        overlay[cropped_component] = [1, 0, 0]

        axes[2].imshow(
            overlay,
            alpha=0.75
        )

        axes[2].set_title(
            f"False Positive: {item['size']} pixels"
        )

        for ax in axes:
            ax.axis("off")

        fig.suptitle(
            f"Rank {rank} | Training Image {item['index']}",
            fontsize=14
        )

        plt.tight_layout()

        filename = os.path.join(
            CROP_FOLDER,
            f"fp_rank_{rank}_image_{item['index']}.png"
        )

        plt.savefig(
            filename,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(f"Saved: {filename}")

    print("\nAll individual crops saved successfully!")


# ==========================================
# MAIN
# ==========================================

def main():

    print("Loading training dataset...")

    dataset = StenosisDataset("train")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print("Loading checkpoint...")

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

    print("Checkpoint epoch:", checkpoint.get("epoch"))

    structure = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    false_positive_regions = []

    print("\nFinding large distant false-positive regions...")

    with torch.no_grad():

        for index, (image, mask) in enumerate(loader):

            image = image.to(DEVICE)

            probabilities = torch.sigmoid(
                model(image)
            )

            prediction = (
                probabilities >= THRESHOLD
            ).cpu().numpy()[0, 0].astype(bool)

            ground_truth = (
                mask.numpy()[0, 0] >= 0.5
            )

            if not ground_truth.any():
                continue

            distance_map = ndimage.distance_transform_edt(
                ~ground_truth
            )

            distant_fp = (
                prediction
                & ~ground_truth
                & (distance_map > MIN_DISTANCE_FROM_GT)
            )

            labels, count = ndimage.label(
                distant_fp,
                structure=structure
            )

            original = image[0, 0].cpu().numpy()

            for component_id in range(1, count + 1):

                component = labels == component_id

                size = int(component.sum())

                if size < MIN_COMPONENT_SIZE:
                    continue

                false_positive_regions.append({

                    "index": index,

                    "size": size,

                    "image": original.copy(),

                    "mask": ground_truth.copy(),

                    "component": component.copy()

                })

            if (index + 1) % 100 == 0:

                print(
                    f"Processed {index + 1}/{len(dataset)} images"
                )

    # ======================================
    # RANK FALSE-POSITIVE REGIONS
    # ======================================

    false_positive_regions.sort(
        key=lambda item: item["size"],
        reverse=True
    )

    selected = false_positive_regions[:MAX_REGIONS]

    print("\nTRAINING FALSE-POSITIVE REGIONS")
    print("---------------------------------")

    print(
        "Total regions found:",
        len(false_positive_regions)
    )

    print(
        "Regions selected:",
        len(selected)
    )

    for rank, item in enumerate(selected, start=1):

        print(
            f"Rank {rank}: "
            f"Image {item['index']}, "
            f"Region size: {item['size']} pixels"
        )

    if not selected:

        print("No regions found.")

        return

    # ======================================
    # SAVE INDIVIDUAL CROPS
    # ======================================

    save_individual_crops(selected)

    # ======================================
    # SAVE COMBINED VISUALIZATION
    # ======================================

    print("\nCreating combined visualization...")

    fig, axes = plt.subplots(
        len(selected),
        3,
        figsize=(15, 4 * len(selected))
    )

    if len(selected) == 1:

        axes = np.expand_dims(
            axes,
            axis=0
        )

    for row, item in enumerate(selected):

        image = item["image"]

        axes[row, 0].imshow(
            image,
            cmap="gray"
        )

        axes[row, 0].set_title(
            f"Original - Image {item['index']}"
        )

        axes[row, 1].imshow(
            item["mask"],
            cmap="gray"
        )

        axes[row, 1].set_title(
            "Ground Truth"
        )

        axes[row, 2].imshow(
            image,
            cmap="gray"
        )

        overlay = np.zeros(
            (*image.shape, 3),
            dtype=np.float32
        )

        overlay[item["component"]] = [1, 0, 0]

        axes[row, 2].imshow(
            overlay,
            alpha=0.75
        )

        axes[row, 2].set_title(
            f"Distant FP: {item['size']} pixels"
        )

        for col in range(3):

            axes[row, col].axis("off")

    plt.tight_layout()

    combined_path = "large_fp_training_visualization.png"

    plt.savefig(
        combined_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nCombined visualization saved:")
    print(combined_path)

    print("\n=================================")
    print("ANALYSIS COMPLETED")
    print("=================================")

    print("Individual crops:", CROP_FOLDER)

    print("Combined image:", combined_path)


if __name__ == "__main__":

    main()
