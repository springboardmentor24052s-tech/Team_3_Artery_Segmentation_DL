
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from torch.utils.data import DataLoader

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model import ResidualUNet


DEVICE = torch.device("cpu")

CHECKPOINT_PATH = (
    "stenosis_detection/checkpoints/resunet_stenosis_best.pt"
)

THRESHOLD = 0.80
MIN_COMPONENT_SIZE = 100


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

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    structure = np.ones((3, 3), dtype=np.uint8)

    false_positive_regions = []

    print("Finding large distant false-positive regions...")

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

            if not ground_truth.any():
                continue

            distance_map = ndimage.distance_transform_edt(
                ~ground_truth
            )

            distant_fp = (
                prediction
                & ~ground_truth
                & (distance_map > 15)
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
                    "prediction": prediction.copy(),
                    "component": component.copy()
                })

    false_positive_regions.sort(
        key=lambda item: item["size"],
        reverse=True
    )

    selected = false_positive_regions[:12]

    print("\nFALSE-POSITIVE REGIONS")
    print("---------------------------------")
    print("Total regions found:", len(false_positive_regions))
    print("Regions visualized:", len(selected))

    for rank, item in enumerate(selected, start=1):

        print(
            f"Rank {rank}: "
            f"Image {item['index']}, "
            f"Region size: {item['size']} pixels"
        )

    if not selected:
        print("No regions found.")
        return

    fig, axes = plt.subplots(
        len(selected),
        3,
        figsize=(15, 4 * len(selected))
    )

    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, item in enumerate(selected):

        image = item["image"]

        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(
            f"Original - Image {item['index']}"
        )

        axes[row, 1].imshow(
            item["mask"],
            cmap="gray"
        )
        axes[row, 1].set_title("Ground Truth")

        axes[row, 2].imshow(image, cmap="gray")

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

    plt.savefig(
        "large_fp_visualization.png",
        dpi=200,
        bbox_inches="tight"
    )

    print("\nSaved: large_fp_visualization.png")

    plt.show()


if __name__ == "__main__":
    main()
