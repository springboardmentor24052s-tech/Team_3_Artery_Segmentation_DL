import numpy as np
from scipy import ndimage

from stenosis_detection.dataset import StenosisDataset


# ==========================================
# SETTINGS
# ==========================================

MIN_COMPONENT_SIZE = 1

SMALL_LIMIT = 100
MEDIUM_LIMIT = 500

# ==========================================
# LOAD TRAINING DATA
# ==========================================

print("=" * 60)
print("STENOSIS COMPONENT ANALYSIS")
print("=" * 60)

dataset = StenosisDataset("train")

print("Training images:", len(dataset))

component_sizes = []

images_with_components = 0
images_without_components = 0

# ==========================================
# ANALYSE MASKS
# ==========================================

for index in range(len(dataset)):

    image, mask = dataset[index]

    mask_array = mask[0].numpy() > 0.5

    # Identify connected stenosis regions.
    labels, number_of_components = ndimage.label(
        mask_array
    )

    if number_of_components == 0:
        images_without_components += 1
        continue

    images_with_components += 1

    sizes = np.bincount(
        labels.ravel()
    )[1:]

    for size in sizes:

        if size >= MIN_COMPONENT_SIZE:
            component_sizes.append(int(size))

    if (index + 1) % 100 == 0:
        print(
            f"Processed {index + 1}/{len(dataset)} images"
        )


# ==========================================
# STATISTICS
# ==========================================

component_sizes = np.array(component_sizes)

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print("Images with stenosis:", images_with_components)
print("Images without stenosis:", images_without_components)

print("Total connected components:", len(component_sizes))

if len(component_sizes) == 0:
    raise RuntimeError("No stenosis components found.")

print("\nCOMPONENT SIZE STATISTICS")
print("-" * 60)

print("Minimum size:", component_sizes.min(), "pixels")
print("Maximum size:", component_sizes.max(), "pixels")
print("Mean size:", round(component_sizes.mean(), 2), "pixels")
print("Median size:", round(np.median(component_sizes), 2), "pixels")


# ==========================================
# SIZE GROUPS
# ==========================================

small = component_sizes[component_sizes < SMALL_LIMIT]

medium = component_sizes[
    (component_sizes >= SMALL_LIMIT) &
    (component_sizes < MEDIUM_LIMIT)
]

large = component_sizes[
    component_sizes >= MEDIUM_LIMIT
]

print("\nCOMPONENT SIZE DISTRIBUTION")
print("-" * 60)

groups = {
    "Small (<100 pixels)": small,
    "Medium (100-499 pixels)": medium,
    "Large (500+ pixels)": large
}

for name, group in groups.items():

    percentage = (
        len(group) / len(component_sizes)
    ) * 100

    print(
        f"{name}: {len(group)} "
        f"({percentage:.2f}%)"
    )


# ==========================================
# PIXEL DISTRIBUTION
# ==========================================

print("\nTOTAL STENOSIS PIXELS BY GROUP")
print("-" * 60)

total_pixels = component_sizes.sum()

for name, group in groups.items():

    pixels = group.sum()

    percentage = (
        pixels / total_pixels
    ) * 100

    print(
        f"{name}: {pixels} pixels "
        f"({percentage:.2f}%)"
    )

# ==========================================
# COMPONENT DISTRIBUTION PER IMAGE
# ==========================================

small_images = set()
medium_images = set()
large_images = set()

small_component_counts = {}
medium_component_counts = {}
large_component_counts = {}

for index in range(len(dataset)):

    _, mask = dataset[index]

    mask_array = mask[0].numpy() > 0.5

    labels, number_of_components = ndimage.label(
        mask_array
    )

    sizes = np.bincount(labels.ravel())[1:]

    for size in sizes:

        if size < SMALL_LIMIT:
            small_images.add(index)
            small_component_counts[index] = (
                small_component_counts.get(index, 0) + 1
            )

        elif size < MEDIUM_LIMIT:
            medium_images.add(index)
            medium_component_counts[index] = (
                medium_component_counts.get(index, 0) + 1
            )

        else:
            large_images.add(index)
            large_component_counts[index] = (
                large_component_counts.get(index, 0) + 1
            )

print("\nCOMPONENT DISTRIBUTION ACROSS IMAGES")
print("-" * 60)

print("Images containing small components:", len(small_images))
print("Images containing medium components:", len(medium_images))
print("Images containing large components:", len(large_images))

print("\nCOMPONENT COUNTS PER IMAGE")
print("-" * 60)

print(
    "Maximum small components in one image:",
    max(small_component_counts.values(), default=0)
)

print(
    "Maximum medium components in one image:",
    max(medium_component_counts.values(), default=0)
)

print(
    "Maximum large components in one image:",
    max(large_component_counts.values(), default=0)
)

# ==========================================
# FINAL SUMMARY
# ==========================================

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED")
print("=" * 60)