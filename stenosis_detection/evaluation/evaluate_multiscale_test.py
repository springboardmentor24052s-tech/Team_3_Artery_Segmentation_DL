
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model_multiscale import MultiScaleResidualUNet


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

CHECKPOINT_PATH = Path(
    "stenosis_detection/checkpoints/"
    "resunet_multiscale_frozen_finetuned_best.pt"
)

THRESHOLD = 0.90
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# LOAD CHECKPOINT
# --------------------------------------------------

def load_checkpoint(model):

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)

    print("FrozenFineTuned checkpoint loaded successfully.")


# --------------------------------------------------
# METRIC CALCULATION
# --------------------------------------------------

def calculate_metrics(tp, fp, fn):

    epsilon = 1e-7

    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    return dice, iou, precision, recall


# --------------------------------------------------
# TEST EVALUATION
# --------------------------------------------------

def evaluate():

    print("=" * 65)
    print("FINAL FROZEN FINETUNED TEST-SET EVALUATION")
    print("=" * 65)

    print(f"Device: {DEVICE}")
    print(f"Threshold: {THRESHOLD}")

    test_dataset = StenosisDataset(split="test")

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(f"Test images: {len(test_dataset)}")

    model = MultiScaleResidualUNet().to(DEVICE)

    load_checkpoint(model)

    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    image_metrics = []

    with torch.no_grad():

        for index, (images, masks) in enumerate(test_loader):

            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            logits = model(images)
            probabilities = torch.sigmoid(logits)

            predictions = (probabilities >= THRESHOLD).float()
            ground_truth = (masks >= 0.5).float()

            tp = torch.sum(
                predictions * ground_truth
            ).item()

            fp = torch.sum(
                predictions * (1 - ground_truth)
            ).item()

            fn = torch.sum(
                (1 - predictions) * ground_truth
            ).item()

            total_tp += tp
            total_fp += fp
            total_fn += fn

            metrics = calculate_metrics(tp, fp, fn)
            image_metrics.append(metrics)

            if (index + 1) % 50 == 0:
                print(
                    f"Processed {index + 1}/{len(test_dataset)} images"
                )

    # GLOBAL METRICS

    global_dice, global_iou, global_precision, global_recall = (
        calculate_metrics(total_tp, total_fp, total_fn)
    )

    # MACRO METRICS

    macro_metrics = np.mean(
        np.array(image_metrics),
        axis=0
    )

    macro_dice, macro_iou, macro_precision, macro_recall = (
        macro_metrics
    )

    # RESULTS

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS - GLOBAL PIXEL-LEVEL")
    print("=" * 65)

    print(f"Threshold : {THRESHOLD:.2f}")
    print(f"Dice      : {global_dice:.6f}")
    print(f"IoU       : {global_iou:.6f}")
    print(f"Precision : {global_precision:.6f}")
    print(f"Recall    : {global_recall:.6f}")

    print("\n" + "=" * 65)
    print("FINAL TEST RESULTS - MACRO AVERAGE")
    print("=" * 65)

    print(f"Dice      : {macro_dice:.6f}")
    print(f"IoU       : {macro_iou:.6f}")
    print(f"Precision : {macro_precision:.6f}")
    print(f"Recall    : {macro_recall:.6f}")

    print("\n" + "=" * 65)
    print("PIXEL COUNTS")
    print("=" * 65)

    print(f"True Positives  : {int(total_tp)}")
    print(f"False Positives : {int(total_fp)}")
    print(f"False Negatives : {int(total_fn)}")

    print("\n" + "=" * 65)
    print("TEST EVALUATION COMPLETED")
    print("=" * 65)

    print(f"Images evaluated: {len(test_dataset)}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Threshold: {THRESHOLD}")

    print("\nNo model weights or checkpoints were modified.")


if __name__ == "__main__":
    evaluate()

