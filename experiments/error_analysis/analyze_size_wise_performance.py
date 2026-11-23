
import torch
import numpy as np
from scipy import ndimage
from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model import ResidualUNet

# -----------------------------------------
# SETTINGS
# -----------------------------------------

CHECKPOINT = (
    "stenosis_detection/checkpoints/"
    "resunet_hardnegative_best.pt"
)

THRESHOLD = 0.80

SMALL_LIMIT = 100
MEDIUM_LIMIT = 500

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# -----------------------------------------
# LOAD MODEL
# -----------------------------------------

print("Loading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

model = ResidualUNet().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Checkpoint epoch:", checkpoint["epoch"])
print("Device:", DEVICE)

# -----------------------------------------
# LOAD VALIDATION DATA
# -----------------------------------------

dataset = StenosisDataset("val")

# Each category tracks:
# TP = correctly predicted stenosis pixels
# FN = missed annotated stenosis pixels

stats = {
    "small": {"tp": 0, "fn": 0, "components": 0},
    "medium": {"tp": 0, "fn": 0, "components": 0},
    "large": {"tp": 0, "fn": 0, "components": 0},
}

print("\nEvaluating validation images...")

# -----------------------------------------
# EVALUATION
# -----------------------------------------

with torch.no_grad():

    for index in range(len(dataset)):

        image, mask = dataset[index]

        image = image.unsqueeze(0).to(DEVICE)

        logits = model(image)

        probabilities = torch.sigmoid(logits)

        prediction = (
            probabilities[0, 0].cpu().numpy()
            >= THRESHOLD
        )

        ground_truth = (
            mask[0].numpy() > 0.5
        )

        labels, number_of_components = ndimage.label(
            ground_truth
        )

        for component_id in range(
            1, number_of_components + 1
        ):

            component = labels == component_id

            size = int(component.sum())

            if size < SMALL_LIMIT:
                category = "small"

            elif size < MEDIUM_LIMIT:
                category = "medium"

            else:
                category = "large"

            true_positive = int(
                np.logical_and(
                    prediction, component
                ).sum()
            )

            false_negative = size - true_positive

            stats[category]["tp"] += true_positive
            stats[category]["fn"] += false_negative
            stats[category]["components"] += 1

        if (index + 1) % 50 == 0:
            print(
                f"Processed {index + 1}/"
                f"{len(dataset)} images"
            )

# -----------------------------------------
# RESULTS
# -----------------------------------------

print("\n" + "=" * 65)
print("SIZE-WISE STENOSIS DETECTION PERFORMANCE")
print("=" * 65)

print(
    f"{'Category':<12}"
    f"{'Components':>12}"
    f"{'GT Pixels':>14}"
    f"{'Detected':>14}"
    f"{'Missed':>12}"
    f"{'Recall':>12}"
)

print("-" * 76)

for category in ["small", "medium", "large"]:

    values = stats[category]

    tp = values["tp"]
    fn = values["fn"]
    total = tp + fn

    recall = tp / total if total > 0 else 0

    print(
        f"{category:<12}"
        f"{values['components']:>12}"
        f"{total:>14}"
        f"{tp:>14}"
        f"{fn:>12}"
        f"{recall:>11.4f}"
    )

print("=" * 76)
print("Analysis completed.")
