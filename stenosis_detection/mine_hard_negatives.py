
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model import ResidualUNet


# ==================================================
# SETTINGS
# ==================================================

CHECKPOINT_PATH = Path(
    "stenosis_detection/checkpoints/"
    "resunet_hardnegative_best.pt"
)

OUTPUT_PATH = Path(
    "stenosis_detection/refined_hard_negative_candidates.json"
)

PATCH_SIZE = 256
STRIDE = 128

PREDICTION_THRESHOLD = 0.80
MIN_PREDICTED_PIXELS = 20

# Exclude patches too close to annotated stenosis.
MIN_DISTANCE_FROM_STENOSIS = 16

MAX_PATCHES_PER_IMAGE = 3

DEVICE = torch.device("cpu")

torch.set_num_threads(4)


# ==================================================
# SAFETY CHECKS
# ==================================================

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

if OUTPUT_PATH.exists():
    raise FileExistsError(
        f"\nOutput already exists: {OUTPUT_PATH}\n"
        "Rename or back up the existing file before running."
    )


# ==================================================
# LOAD MODEL
# ==================================================

print("=" * 60)
print("REFINED HARD-NEGATIVE MINING")
print("=" * 60)

print("\nLoading hard-negative best checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

model = ResidualUNet(
    in_channels=1,
    out_channels=1
).to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Checkpoint epoch:", checkpoint["epoch"])
print(
    "Best validation Dice:",
    checkpoint["best_val_soft_dice"]
)

print("Model loaded successfully.")


# ==================================================
# LOAD TRAINING DATA ONLY
# ==================================================

print("\nLoading training dataset...")

dataset = StenosisDataset(split="train")

print("Training images:", len(dataset))


# ==================================================
# REFINED HARD-NEGATIVE MINING
# ==================================================

candidates = []

total_images_with_candidates = 0
total_candidate_patches = 0

rejected_overlap = 0
rejected_near_stenosis = 0

with torch.no_grad():

    for image_index in range(len(dataset)):

        image, mask = dataset[image_index]

        image = image.unsqueeze(0).to(DEVICE)

        probabilities = torch.sigmoid(
            model(image)
        )[0, 0]

        # Convert annotation mask to NumPy.
        ground_truth = (
            mask[0].cpu().numpy() > 0.5
        )

        # Calculate distance from every pixel to
        # the nearest annotated stenosis pixel.
        distance_map = distance_transform_edt(
            ~ground_truth
        )

        image_candidates = []

        height, width = ground_truth.shape

        for y in range(
            0,
            height - PATCH_SIZE + 1,
            STRIDE
        ):

            for x in range(
                0,
                width - PATCH_SIZE + 1,
                STRIDE
            ):

                ground_truth_patch = ground_truth[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                # Reject patches containing annotated stenosis.
                if ground_truth_patch.any():
                    rejected_overlap += 1
                    continue

                # Reject patches too close to annotated stenosis.
                distance_patch = distance_map[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                if (
                    distance_patch.min()
                    < MIN_DISTANCE_FROM_STENOSIS
                ):
                    rejected_near_stenosis += 1
                    continue

                predicted_patch = probabilities[
                    y:y + PATCH_SIZE,
                    x:x + PATCH_SIZE
                ]

                # Identify high-confidence predictions.
                positive_predictions = (
                    predicted_patch >= PREDICTION_THRESHOLD
                )

                predicted_pixels = int(
                    positive_predictions.sum().item()
                )

                if predicted_pixels < MIN_PREDICTED_PIXELS:
                    continue

                # Average confidence among predicted pixels.
                mean_confidence = (
                    predicted_patch[positive_predictions]
                    .mean()
                    .item()
                )

                max_confidence = (
                    predicted_patch.max().item()
                )

                image_candidates.append({

                    "image_index": image_index,
                    "x": x,
                    "y": y,

                    "predicted_pixels": predicted_pixels,

                    "mean_confidence": round(
                        mean_confidence, 5
                    ),

                    "max_probability": round(
                        max_confidence, 5
                    ),

                    "distance_from_stenosis": round(
                        float(distance_patch.min()), 2
                    )
                })

        # Rank candidates using predicted pixel count
        # and mean confidence.
        image_candidates.sort(
            key=lambda item: (
                item["predicted_pixels"],
                item["mean_confidence"]
            ),
            reverse=True
        )

        # Retain strongest candidates per image.
        image_candidates = image_candidates[
            :MAX_PATCHES_PER_IMAGE
        ]

        if image_candidates:

            total_images_with_candidates += 1

            total_candidate_patches += len(
                image_candidates
            )

            candidates.extend(image_candidates)

        if (image_index + 1) % 100 == 0:

            print(
                f"Processed {image_index + 1}/"
                f"{len(dataset)} images | "
                f"Candidates: {total_candidate_patches}"
            )


# ==================================================
# SAVE RESULTS
# ==================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        candidates,
        file,
        indent=4
    )


# ==================================================
# SUMMARY
# ==================================================

print("\n" + "=" * 60)
print("REFINED HARD-NEGATIVE MINING COMPLETED")
print("=" * 60)

print("Training images:", len(dataset))

print(
    "Images with candidates:",
    total_images_with_candidates
)

print(
    "Refined candidate patches:",
    total_candidate_patches
)

print("\nFILTERING SUMMARY")
print("----------------------------------------")

print("Rejected: annotated overlap:", rejected_overlap)

print(
    "Rejected: too close to annotations:",
    rejected_near_stenosis
)

print("\nMINING SETTINGS")
print("----------------------------------------")

print("Prediction threshold:", PREDICTION_THRESHOLD)
print("Minimum predicted pixels:", MIN_PREDICTED_PIXELS)
print("Patch size:", PATCH_SIZE)
print("Minimum annotation distance:", MIN_DISTANCE_FROM_STENOSIS)
print("Maximum patches per image:", MAX_PATCHES_PER_IMAGE)

print("\nSaved candidates to:")
print(OUTPUT_PATH)

print("\nIMPORTANT:")
print(
    "These are potentially confusing unannotated regions, "
    "not confirmed healthy tissue."
)
