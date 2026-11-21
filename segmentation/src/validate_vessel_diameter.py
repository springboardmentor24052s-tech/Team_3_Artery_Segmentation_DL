
from pathlib import Path
import csv

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

MASK_DIR = Path("outputs/extracted_vessel_structures")
CENTERLINE_DIR = Path("outputs/vessel_centerlines")
DIAMETER_DIR = Path("outputs/vessel_diameter/diameter_maps")

OUTPUT_DIR = Path("outputs/diameter_validation")
VISUAL_DIR = OUTPUT_DIR / "visualizations"


def main():
    for folder in [MASK_DIR, CENTERLINE_DIR, DIAMETER_DIR]:
        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

    VISUAL_DIR.mkdir(parents=True, exist_ok=True)

    mask_files = sorted(MASK_DIR.glob("*.npy"))

    results = []
    all_values = []

    for mask_path in mask_files:
        name = mask_path.name

        centerline_path = CENTERLINE_DIR / name
        diameter_path = DIAMETER_DIR / name

        if not centerline_path.exists() or not diameter_path.exists():
            print(f"Skipping {name}: missing output")
            continue

        mask = np.load(mask_path) > 0
        centerline = np.load(centerline_path) > 0
        diameter_map = np.load(diameter_path)

        if not (
            mask.shape == centerline.shape == diameter_map.shape
        ):
            print(f"Skipping {name}: shape mismatch")
            continue

        expected_centerline = centerline & mask

        # Independently recalculate expected diameters
        distance_map = distance_transform_edt(mask)
        expected_diameters = 2.0 * distance_map[expected_centerline]

        actual_diameters = diameter_map[expected_centerline]

        if len(actual_diameters) == 0:
            print(f"Skipping {name}: no centerline points")
            continue

        positive = bool(np.all(actual_diameters > 0))
        finite = bool(np.all(np.isfinite(actual_diameters)))

        correct_locations = bool(
            np.all(diameter_map[~expected_centerline] == 0)
        )

        values_match = bool(
            np.allclose(
                actual_diameters,
                expected_diameters,
                atol=1e-4
            )
        )

        # Generate RGB visualization:
        # Grey = vessel mask
        # Blue-to-red = local diameter along centerline
        visual = np.zeros((*mask.shape, 3), dtype=np.uint8)
        visual[mask] = [100, 100, 100]

        max_d = float(actual_diameters.max())

        if max_d > 0:
            normalized = np.clip(
                actual_diameters / max_d, 0, 1
            )

            colors = np.zeros((len(normalized), 3), dtype=np.uint8)
            colors[:, 0] = (normalized * 255).astype(np.uint8)
            colors[:, 2] = ((1 - normalized) * 255).astype(np.uint8)

            visual[expected_centerline] = colors

        Image.fromarray(visual).save(
            VISUAL_DIR / f"{mask_path.stem}.png"
        )

        results.append([
            name,
            len(actual_diameters),
            positive,
            finite,
            correct_locations,
            values_match,
            round(float(actual_diameters.mean()), 4),
            round(float(np.median(actual_diameters)), 4),
            round(float(actual_diameters.min()), 4),
            round(float(actual_diameters.max()), 4)
        ])

        all_values.extend(actual_diameters.tolist())

    report_path = OUTPUT_DIR / "diameter_validation_report.csv"

    with open(report_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image",
            "centerline_points",
            "all_positive",
            "all_finite",
            "correct_locations",
            "matches_distance_transform",
            "mean_diameter_px",
            "median_diameter_px",
            "minimum_diameter_px",
            "maximum_diameter_px"
        ])
        writer.writerows(results)

    print("\nDIAMETER VALIDATION SUMMARY")
    print("---------------------------")
    print("Images checked:", len(results))

    if results:
        print(
            "All positive:",
            all(row[2] for row in results)
        )
        print(
            "All finite:",
            all(row[3] for row in results)
        )
        print(
            "Correct locations:",
            all(row[4] for row in results)
        )
        print(
            "Matches recalculation:",
            all(row[5] for row in results)
        )

        values = np.asarray(all_values)

        print("\nOverall diameter statistics:")
        print("Mean:", round(float(values.mean()), 3), "pixels")
        print("Median:", round(float(np.median(values)), 3), "pixels")
        print("Minimum:", round(float(values.min()), 3), "pixels")
        print("Maximum:", round(float(values.max()), 3), "pixels")

    print("\nCSV report:", report_path)
    print("Visualizations:", VISUAL_DIR)


if __name__ == "__main__":
    main()
