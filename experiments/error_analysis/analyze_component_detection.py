
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

# -----------------------------------------
# LOAD DATASET
# -----------------------------------------

dataset = StenosisDataset("val")

categories = {
    "small": [],
    "medium": [],
    "large": []
}

# -----------------------------------------
# EVALUATE COMPONENTS
# -----------------------------------------

print("\nEvaluating individual components...")

with torch.no_grad():

    for index in range(len(dataset)):

        image, mask = dataset[index]

        image = image.unsqueeze(0).to(DEVICE)

        probabilities = torch.sigmoid(model(image))

        prediction = (
            probabilities[0, 0].cpu().numpy()
            >= THRESHOLD
        )

        ground_truth = mask[0].numpy() > 0.5

        labels, number_of_components = ndimage.label(
            ground_truth
        )

        for component_id in range(
            1, number_of_components + 1
        ):

            component = labels == component_id

            size = int(component.sum())

            detected = int(
                np.logical_and(
                    prediction, component
                ).sum()
            )

            recall = detected / size

            if size < 100:
                category = "small"

            elif size < 500:
                category = "medium"

            else:
                category = "large"

            categories[category].append(recall)

        if (index + 1) % 50 == 0:
            print(
                f"Processed {index + 1}/"
                f"{len(dataset)} images"
            )

# -----------------------------------------
# RESULTS
# -----------------------------------------

print("\n" + "=" * 65)
print("COMPONENT-LEVEL DETECTION ANALYSIS")
print("=" * 65)

for category in ["small", "medium", "large"]:

    recalls = np.array(categories[category])

    if len(recalls) == 0:
        continue

    print(f"\n{category.upper()} COMPONENTS")
    print("-" * 45)

    print("Total components:", len(recalls))

    print(
        "Detected at least 25%:",
        np.sum(recalls >= 0.25)
    )

    print(
        "Detected at least 50%:",
        np.sum(recalls >= 0.50)
    )

    print(
        "Detected at least 75%:",
        np.sum(recalls >= 0.75)
    )

    print(
        "Completely missed:",
        np.sum(recalls == 0)
    )

    print(
        "Mean component recall:",
        round(float(np.mean(recalls)), 4)
    )

print("\nAnalysis completed.")
