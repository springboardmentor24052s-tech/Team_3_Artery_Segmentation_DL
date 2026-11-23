
import os
import pandas as pd
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

# Update these paths if your project structure is different.

ANNOTATION_FILE = "annotations.csv"

IMAGE_DIR = "arcade_lesion_review_v3"

# Expected severity classes
EXPECTED_CLASSES = ["Mild", "Moderate", "Severe"]

# Common column names used for image paths and labels.
IMAGE_COLUMNS = [
    "image_path",
    "crop_path",
    "filename",
    "image",
    "crop",
    "file_name"
]

LABEL_COLUMNS = [
    "severity",
    "label",
    "class",
    "severity_label"
]

ID_COLUMNS = [
    "lesion_id",
    "lesionid",
    "id"
]

SPLIT_COLUMNS = [
    "split",
    "set",
    "dataset_split"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, possible_columns):

    for col in possible_columns:
        if col in df.columns:
            return col

    return None


def resolve_image_path(image_value):

    if pd.isna(image_value):
        return None

    image_value = str(image_value).strip()

    if not image_value:
        return None

    # Try the path as provided
    if os.path.isfile(image_value):
        return image_value

    # Try relative to image directory
    candidate = os.path.join(IMAGE_DIR, image_value)

    if os.path.isfile(candidate):
        return candidate

    # Try basename inside image directory
    candidate = os.path.join(
        IMAGE_DIR,
        os.path.basename(image_value)
    )

    if os.path.isfile(candidate):
        return candidate

    return None


# ============================================================
# MAIN DATASET INSPECTION
# ============================================================

def inspect_dataset():

    print("\n" + "=" * 65)
    print("       ARCADE SEVERITY DATASET INSPECTION")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Check annotation file
    # --------------------------------------------------------

    if not os.path.isfile(ANNOTATION_FILE):

        print("\nERROR: Annotation file not found!")
        print("Expected:", os.path.abspath(ANNOTATION_FILE))
        return

    df = pd.read_csv(ANNOTATION_FILE)

    print("\n[1] BASIC DATASET INFORMATION")
    print("-" * 65)

    print("Annotation file:", os.path.abspath(ANNOTATION_FILE))
    print("Total annotation rows:", len(df))
    print("Columns:", list(df.columns))

    # --------------------------------------------------------
    # 2. Identify important columns
    # --------------------------------------------------------

    image_col = find_column(df, IMAGE_COLUMNS)
    label_col = find_column(df, LABEL_COLUMNS)
    id_col = find_column(df, ID_COLUMNS)
    split_col = find_column(df, SPLIT_COLUMNS)

    print("\n[2] IDENTIFIED COLUMNS")
    print("-" * 65)

    print("Image column:", image_col)
    print("Severity column:", label_col)
    print("Lesion ID column:", id_col)
    print("Split column:", split_col)

    if image_col is None or label_col is None:

        print("\nERROR: Could not identify image or severity column.")
        print("Please check the column names printed above.")
        return

    # --------------------------------------------------------
    # 3. Inspect labels
    # --------------------------------------------------------

    print("\n[3] LABEL DISTRIBUTION")
    print("-" * 65)

    labels = df[label_col].dropna().astype(str).str.strip()

    print("Label counts:")

    for label, count in labels.value_counts().items():
        print(f"{label:20s}: {count}")

    print("\nExpected classes:")

    for cls in EXPECTED_CLASSES:

        count = (labels.str.lower() == cls.lower()).sum()

        print(f"{cls:20s}: {count}")

    # --------------------------------------------------------
    # 4. Check unlabelled samples
    # --------------------------------------------------------

    print("\n[4] UNLABELLED SAMPLES")
    print("-" * 65)

    missing_labels = df[label_col].isna().sum()

    empty_labels = (
        df[label_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    print("Missing labels:", missing_labels)
    print("Empty labels:", empty_labels)

    # --------------------------------------------------------
    # 5. Check unexpected labels
    # --------------------------------------------------------

    print("\n[5] UNEXPECTED LABELS")
    print("-" * 65)

    expected_lower = {
        x.lower() for x in EXPECTED_CLASSES
    }

    unexpected = labels[
        ~labels.str.lower().isin(expected_lower)
    ]

    if len(unexpected) == 0:
        print("No unexpected labels found.")
    else:
        print(unexpected.value_counts().to_string())

    # --------------------------------------------------------
    # 6. Check image files
    # --------------------------------------------------------

    print("\n[6] IMAGE FILE VALIDATION")
    print("-" * 65)

    resolved_paths = df[image_col].apply(resolve_image_path)

    missing_images = resolved_paths.isna()

    print("Total image references:", len(df))
    print("Existing image files:", (~missing_images).sum())
    print("Missing image files:", missing_images.sum())

    if missing_images.any():

        print("\nMissing image references:")

        for value in df.loc[missing_images, image_col].head(20):
            print(" -", value)

    # --------------------------------------------------------
    # 7. Check duplicate lesion IDs
    # --------------------------------------------------------

    print("\n[7] DUPLICATE LESION IDs")
    print("-" * 65)

    if id_col is not None:

        valid_ids = df[id_col].dropna()

        duplicate_count = valid_ids.duplicated().sum()

        print("Total valid lesion IDs:", len(valid_ids))
        print("Duplicate ID rows:", duplicate_count)

        if duplicate_count > 0:
            print("\nDuplicate IDs:")
            print(
                df.loc[
                    df[id_col].duplicated(keep=False),
                    id_col
                ].value_counts().head(20)
            )

    else:

        print("No lesion ID column detected.")

    # --------------------------------------------------------
    # 8. Check train/validation/test distribution
    # --------------------------------------------------------

    print("\n[8] DATASET SPLIT DISTRIBUTION")
    print("-" * 65)

    if split_col is not None:

        print("\nSplit counts:")
        print(df[split_col].value_counts(dropna=False))

        print("\nSeverity distribution by split:")

        print(
            pd.crosstab(
                df[split_col],
                df[label_col],
                margins=True,
                dropna=False
            )
        )

    else:

        print("No split column detected.")
        print("Check whether splits are defined inside ArcadeSeverityDataset.")

    # --------------------------------------------------------
    # 9. Check image directory
    # --------------------------------------------------------

    print("\n[9] IMAGE DIRECTORY")
    print("-" * 65)

    if os.path.isdir(IMAGE_DIR):

        print("Image directory exists:", os.path.abspath(IMAGE_DIR))

        image_extensions = (
            ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
        )

        image_files = [
            f for f in os.listdir(IMAGE_DIR)
            if f.lower().endswith(image_extensions)
        ]

        print("Number of image files:", len(image_files))

    else:

        print("WARNING: Image directory not found!")
        print("Expected:", os.path.abspath(IMAGE_DIR))

    # --------------------------------------------------------
    # 10. Final summary
    # --------------------------------------------------------

    print("\n[10] INSPECTION SUMMARY")
    print("-" * 65)

    valid_labels = labels[
        labels.str.lower().isin(expected_lower)
    ]

    print("Total rows:", len(df))
    print("Valid severity-labelled rows:", len(valid_labels))
    print("Missing labels:", missing_labels)
    print("Missing images:", missing_images.sum())

    print("\nClass counts:")

    for cls in EXPECTED_CLASSES:

        count = (
            valid_labels.str.lower() == cls.lower()
        ).sum()

        print(f"{cls:20s}: {count}")

    print("\n" + "=" * 65)
    print("       DATASET INSPECTION COMPLETED")
    print("=" * 65)


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    inspect_dataset()
