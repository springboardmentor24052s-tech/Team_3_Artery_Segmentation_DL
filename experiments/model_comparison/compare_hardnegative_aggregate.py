import torch
import torch.nn as nn
from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model import ResidualUNet

# -----------------------------------------
# CONFIGURATION
# -----------------------------------------

DEVICE = torch.device("cpu")

ORIGINAL_CHECKPOINT = (
    "stenosis_detection/checkpoints/"
    "resunet_stenosis_epoch17_backup.pt"
)

HARDNEGATIVE_CHECKPOINT = (
    "stenosis_detection/checkpoints/"
    "resunet_hardnegative_best.pt"
)

THRESHOLDS = [0.70, 0.80]

# -----------------------------------------
# LOAD VALIDATION DATASET
# -----------------------------------------

print("Loading validation dataset...")

dataset = StenosisDataset(split="val")

print("Validation images:", len(dataset))


# -----------------------------------------
# LOAD MODEL
# -----------------------------------------

def load_model(checkpoint_path):

    model = ResidualUNet(
        in_channels=1,
        out_channels=1
    ).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# -----------------------------------------
# EVALUATE MODEL
# -----------------------------------------

def evaluate_model(model, threshold):

    tp_total = 0
    fp_total = 0
    fn_total = 0

    with torch.no_grad():

        for index in range(len(dataset)):

            image, mask = dataset[index]

            image = image.unsqueeze(0).to(DEVICE)
            mask = mask.to(DEVICE)

            logits = model(image)

            probabilities = torch.sigmoid(logits)

            prediction = (
                probabilities[0, 0] >= threshold
            ).float()

            actual = mask[0]

            tp_total += (
                (prediction * actual).sum().item()
            )

            fp_total += (
                (prediction * (1 - actual)).sum().item()
            )

            fn_total += (
                ((1 - prediction) * actual).sum().item()
            )

            if (index + 1) % 50 == 0:
                print(
                    f"Processed {index + 1}/"
                    f"{len(dataset)} images"
                )

    precision = tp_total / (tp_total + fp_total + 1e-8)

    recall = tp_total / (tp_total + fn_total + 1e-8)

    dice = (
        2 * tp_total /
        (2 * tp_total + fp_total + fn_total + 1e-8)
    )

    iou = (
        tp_total /
        (tp_total + fp_total + fn_total + 1e-8)
    )

    return {
        "TP": tp_total,
        "FP": fp_total,
        "FN": fn_total,
        "Dice": dice,
        "IoU": iou,
        "Precision": precision,
        "Recall": recall
    }


# -----------------------------------------
# MAIN COMPARISON
# -----------------------------------------

print("\nLoading original model...")
original_model = load_model(ORIGINAL_CHECKPOINT)

print("\nLoading hard-negative model...")
hardnegative_model = load_model(HARDNEGATIVE_CHECKPOINT)

results = {}

for threshold in THRESHOLDS:

    print("\n" + "=" * 65)
    print(f"THRESHOLD: {threshold}")
    print("=" * 65)

    print("\nEvaluating original model...")

    original = evaluate_model(
        original_model,
        threshold
    )

    print("\nEvaluating hard-negative model...")

    hardnegative = evaluate_model(
        hardnegative_model,
        threshold
    )

    results[threshold] = {
        "Original": original,
        "Hard-negative": hardnegative
    }

    print("\nRESULTS")
    print("-" * 65)

    print(
        f"{'Metric':<15}"
        f"{'Original':>18}"
        f"{'Hard-negative':>20}"
    )

    for metric in [
        "TP",
        "FP",
        "FN",
        "Dice",
        "IoU",
        "Precision",
        "Recall"
    ]:

        print(
            f"{metric:<15}"
            f"{original[metric]:>18.6f}"
            f"{hardnegative[metric]:>20.6f}"
        )

    fp_reduction = (
        (original["FP"] - hardnegative["FP"])
        / (original["FP"] + 1e-8)
    ) * 100

    print("\nFALSE POSITIVE REDUCTION:")
    print(f"{fp_reduction:.2f}%")

print("\nComparison completed!")