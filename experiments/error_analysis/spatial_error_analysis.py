
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

DISTANCE_BINS = [
    ("Near (0-5 px)", 0, 5),
    ("Medium (5-15 px)", 5, 15),
    ("Far (>15 px)", 15, float("inf"))
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

    distance_counts = {
        label: 0 for label, _, _ in DISTANCE_BINS
    }

    total_fp = 0
    total_tp = 0
    images_without_gt = 0

    print("Analyzing spatial false-positive distribution...")

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

            tp_mask = prediction & ground_truth
            fp_mask = prediction & ~ground_truth

            total_tp += int(tp_mask.sum())
            total_fp += int(fp_mask.sum())

            if not ground_truth.any():

                images_without_gt += 1

                distance_counts["Far (>15 px)"] += int(
                    fp_mask.sum()
                )

                continue

            # Distance of every pixel to nearest GT pixel.
            distance_map = ndimage.distance_transform_edt(
                ~ground_truth
            )

            fp_distances = distance_map[fp_mask]

            for label, lower, upper in DISTANCE_BINS:

                if upper == float("inf"):
                    count = np.sum(fp_distances > lower)
                else:
                    count = np.sum(
                        (fp_distances >= lower) &
                        (fp_distances < upper)
                    )

                distance_counts[label] += int(count)

    print("\nSPATIAL FALSE-POSITIVE ANALYSIS")
    print("=" * 65)

    print("Validation images:", len(dataset))
    print("Images without GT stenosis:", images_without_gt)
    print("True-positive pixels:", total_tp)
    print("False-positive pixels:", total_fp)

    print("\nFALSE POSITIVE DISTANCE DISTRIBUTION")
    print("-" * 65)

    for label, _, _ in DISTANCE_BINS:

        count = distance_counts[label]

        percentage = (
            count / max(total_fp, 1)
        ) * 100

        print(
            f"{label:<20} "
            f"{count:>10} pixels "
            f"({percentage:.2f}%)"
        )

    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()
