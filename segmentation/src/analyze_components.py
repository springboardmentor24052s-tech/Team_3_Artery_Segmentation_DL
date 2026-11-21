

from pathlib import Path

import numpy as np
from scipy import ndimage


FOLDERS = {
    "Baseline U-Net": Path("outputs/baseline_binary_masks"),
    "Attention U-Net": Path("outputs/attention_binary_masks"),
}

# 8-connectivity: considers diagonal neighbors connected.
STRUCTURE = np.ones((3, 3), dtype=np.uint8)


def analyze_mask(mask_path):
    mask = np.load(mask_path).astype(bool)

    labels, component_count = ndimage.label(
        mask,
        structure=STRUCTURE
    )

    component_sizes = np.bincount(labels.ravel())[1:]
    component_sizes = np.sort(component_sizes)[::-1]

    total_pixels = int(mask.sum())

    print(f"\nFile: {mask_path.name}")
    print(f"Total foreground pixels: {total_pixels}")
    print(f"Connected components: {component_count}")

    print("Largest 10 components:")

    for index, size in enumerate(component_sizes[:10], start=1):
        print(f"  Component {index}: {size} pixels")

    print("\nPotential noise analysis:")

    for threshold in [5, 10, 20, 50]:
        small_pixels = int(
            component_sizes[component_sizes <= threshold].sum()
        )

        print(
            f"Components <= {threshold} pixels: "
            f"{int(np.sum(component_sizes <= threshold))}, "
            f"pixels affected: {small_pixels}"
        )


def main():
    for model_name, folder in FOLDERS.items():
        print("\n" + "=" * 55)
        print(model_name)
        print("=" * 55)

        for mask_path in sorted(folder.glob("*.npy")):
            analyze_mask(mask_path)


if __name__ == "__main__":
    main()
