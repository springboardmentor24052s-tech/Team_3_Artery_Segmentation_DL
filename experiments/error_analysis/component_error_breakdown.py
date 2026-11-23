
import torch
import numpy as np
from scipy import ndimage
from torch.utils.data import DataLoader

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model import ResidualUNet


DEVICE = torch.device("cpu")

CHECKPOINT_PATH = (
    "stenosis_detection/checkpoints/resunet_stenosis_best.pt"
)

THRESHOLD = 0.80

SIZE_BINS = [
    ("1-10", 1, 10),
    ("10-40", 10, 40),
    ("40-80", 40, 80),
    ("80-120", 80, 120),
    ("120-200", 120, 200),
    ("200-400", 200, 400),
    ("400-800", 400, 800),
    ("800-1600", 800, 1600),
    ("1600+", 1600, float("inf"))
]


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

    stats = {
        label: {
            "components": 0,
            "component_pixels": 0,
            "fp_pixels": 0,
            "tp_pixels": 0
        }
        for label, _, _ in SIZE_BINS
    }

    total_fp = 0
    total_tp = 0

    print("Analyzing component-level errors...")

    with torch.no_grad():

        for image, mask in loader:

            image = image.to(DEVICE)

            probabilities = torch.sigmoid(model(image))

            prediction = (
                probabilities >= THRESHOLD
            ).cpu().numpy()[0, 0].astype(bool)

            ground_truth = (
                mask.numpy()[0, 0] >= 0.5
            )

            labels, count = ndimage.label(
                prediction,
                structure=structure
            )

            for component_id in range(1, count + 1):

                component = labels == component_id

                size = int(component.sum())

                tp = int(
                    np.logical_and(
                        component,
                        ground_truth
                    ).sum()
                )

                fp = size - tp

                total_tp += tp
                total_fp += fp

                for label, lower, upper in SIZE_BINS:

                    if lower <= size < upper:

                        stats[label]["components"] += 1
                        stats[label]["component_pixels"] += size
                        stats[label]["tp_pixels"] += tp
                        stats[label]["fp_pixels"] += fp

                        break

    print("\nCOMPONENT SIZE ERROR BREAKDOWN")
    print("=" * 100)

    print(
        f"{'Size Range':<18}"
        f"{'Components':<14}"
        f"{'Total Pixels':<16}"
        f"{'TP Pixels':<14}"
        f"{'FP Pixels':<14}"
        f"{'FP %':<10}"
    )

    for label, lower, upper in SIZE_BINS:

        item = stats[label]

        fp_percentage = (
            item["fp_pixels"] /
            max(total_fp, 1)
        ) * 100

        upper_text = (
            "inf" if upper == float("inf")
            else str(int(upper))
        )

        size_range = f"{lower}-{upper_text}"

        print(
            f"{size_range:<18}"
            f"{item['components']:<14}"
            f"{item['component_pixels']:<16}"
            f"{item['tp_pixels']:<14}"
            f"{item['fp_pixels']:<14}"
            f"{fp_percentage:<10.2f}"
        )

    print("\nOVERALL CHECK")
    print("---------------------------------")
    print("Total TP:", total_tp)
    print("Total FP:", total_fp)

    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()
