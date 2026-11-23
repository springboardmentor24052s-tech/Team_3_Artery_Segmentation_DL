
from pathlib import Path

import torch

from stenosis_detection.dataset import StenosisDataset
from stenosis_detection.model import ResidualUNet
from stenosis_detection.model_multiscale import MultiScaleResidualUNet


# ==================================================
# SETTINGS
# ==================================================

DEVICE = torch.device("cpu")

CHECKPOINTS = {
    "Original": (
        "stenosis_detection/checkpoints/"
        "resunet_stenosis_epoch17_backup.pt"
    ),

    "HardNegative": (
        "stenosis_detection/checkpoints/"
        "resunet_hardnegative_best.pt"
    ),

    "BoundaryAware": (
        "stenosis_detection/checkpoints/"
        "resunet_boundary_aware_best.pt"
    ),

    "RefinedHardNegative": (
        "stenosis_detection/checkpoints/"
        "resunet_refined_hardnegative_best.pt"
    ),

    "MultiScale": (
        "stenosis_detection/checkpoints/"
        "resunet_multiscale_best.pt"
    ),

    "FrozenFineTuned": (
        "stenosis_detection/checkpoints/"
        "resunet_multiscale_frozen_finetuned_best.pt"
    ),
}

THRESHOLDS = [
    0.30, 0.40, 0.50, 0.60,
    0.70, 0.80, 0.90
]


# ==================================================
# SAFETY CHECKS
# ==================================================

def check_checkpoints():

    print("\nChecking checkpoint files...")

    for name, path in CHECKPOINTS.items():

        if not Path(path).exists():

            raise FileNotFoundError(
                f"{name} checkpoint not found:\n{path}"
            )

        print(f"{name}: Found")


# ==================================================
# LOAD MODEL
# ==================================================

def load_model(name, path):

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    # Both MultiScale checkpoints use the same architecture.
    if name in ("MultiScale", "FrozenFineTuned"):

        model = MultiScaleResidualUNet(
            in_channels=1,
            out_channels=1
        ).to(DEVICE)

    else:

        model = ResidualUNet(
            in_channels=1,
            out_channels=1
        ).to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


# ==================================================
# EVALUATION
# ==================================================

def evaluate_model(model, dataset):

    results = {
        threshold: {
            "dice": [],
            "iou": [],
            "precision": [],
            "recall": []
        }
        for threshold in THRESHOLDS
    }

    global_counts = {
        threshold: {
            "tp": 0,
            "fp": 0,
            "fn": 0
        }
        for threshold in THRESHOLDS
    }

    with torch.no_grad():

        for index in range(len(dataset)):

            image, mask = dataset[index]

            image = image.unsqueeze(0).to(DEVICE)
            mask = mask[0].to(DEVICE)

            probabilities = torch.sigmoid(
                model(image)[0, 0]
            )

            for threshold in THRESHOLDS:

                prediction = (
                    probabilities >= threshold
                ).float()

                tp = (
                    prediction * mask
                ).sum().item()

                fp = (
                    prediction * (1 - mask)
                ).sum().item()

                fn = (
                    (1 - prediction) * mask
                ).sum().item()

                dice = (
                    2 * tp + 1e-6
                ) / (
                    2 * tp + fp + fn + 1e-6
                )

                iou = (
                    tp + 1e-6
                ) / (
                    tp + fp + fn + 1e-6
                )

                precision = (
                    tp + 1e-6
                ) / (
                    tp + fp + 1e-6
                )

                recall = (
                    tp + 1e-6
                ) / (
                    tp + fn + 1e-6
                )

                # Macro-average metrics
                results[threshold]["dice"].append(dice)
                results[threshold]["iou"].append(iou)
                results[threshold]["precision"].append(precision)
                results[threshold]["recall"].append(recall)

                # Global pixel-level counts
                global_counts[threshold]["tp"] += tp
                global_counts[threshold]["fp"] += fp
                global_counts[threshold]["fn"] += fn

    return results, global_counts


# ==================================================
# PRINT MACRO-AVERAGE RESULTS
# ==================================================

def print_results(name, results):

    print("\n" + "=" * 85)
    print(name.upper() + " - MACRO AVERAGE")
    print("=" * 85)

    print(
        f"{'Threshold':<12}"
        f"{'Dice':<12}"
        f"{'IoU':<12}"
        f"{'Precision':<14}"
        f"{'Recall':<12}"
    )

    for threshold in THRESHOLDS:

        values = results[threshold]

        dice = sum(values["dice"]) / len(values["dice"])
        iou = sum(values["iou"]) / len(values["iou"])
        precision = sum(values["precision"]) / len(values["precision"])
        recall = sum(values["recall"]) / len(values["recall"])

        print(
            f"{threshold:<12.2f}"
            f"{dice:<12.4f}"
            f"{iou:<12.4f}"
            f"{precision:<14.4f}"
            f"{recall:<12.4f}"
        )


# ==================================================
# GLOBAL PIXEL-LEVEL METRICS
# ==================================================

def calculate_global_metrics(counts):

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]

    dice = (
        2 * tp + 1e-6
    ) / (
        2 * tp + fp + fn + 1e-6
    )

    iou = (
        tp + 1e-6
    ) / (
        tp + fp + fn + 1e-6
    )

    precision = (
        tp + 1e-6
    ) / (
        tp + fp + 1e-6
    )

    recall = (
        tp + 1e-6
    ) / (
        tp + fn + 1e-6
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall
    }


# ==================================================
# PRINT GLOBAL RESULTS
# ==================================================

def print_global_results(name, global_counts):

    print("\n" + "=" * 85)
    print(name.upper() + " - GLOBAL PIXEL-LEVEL")
    print("=" * 85)

    print(
        f"{'Threshold':<12}"
        f"{'Dice':<12}"
        f"{'IoU':<12}"
        f"{'Precision':<14}"
        f"{'Recall':<12}"
    )

    for threshold in THRESHOLDS:

        metrics = calculate_global_metrics(
            global_counts[threshold]
        )

        print(
            f"{threshold:<12.2f}"
            f"{metrics['dice']:<12.4f}"
            f"{metrics['iou']:<12.4f}"
            f"{metrics['precision']:<14.4f}"
            f"{metrics['recall']:<12.4f}"
        )


# ==================================================
# FINAL MODEL RANKING
# ==================================================

def print_model_ranking(all_results):

    print("\n" + "=" * 85)
    print("MODEL RANKING - GLOBAL DICE")
    print("=" * 85)

    print(
        f"{'Threshold':<12}"
        f"{'Best Model':<28}"
        f"{'Global Dice':<15}"
    )

    for threshold in THRESHOLDS:

        rankings = []

        for name, model_results in all_results.items():

            metrics = calculate_global_metrics(
                model_results["global"][threshold]
            )

            rankings.append(
                (name, metrics["dice"])
            )

        rankings.sort(
            key=lambda item: item[1],
            reverse=True
        )

        best_name, best_dice = rankings[0]

        print(
            f"{threshold:<12.2f}"
            f"{best_name:<28}"
            f"{best_dice:<15.4f}"
        )


# ==================================================
# OVERALL BEST MODEL
# ==================================================

def print_overall_ranking(all_results):

    print("\n" + "=" * 85)
    print("BEST GLOBAL DICE FOR EACH MODEL")
    print("=" * 85)

    rankings = []

    for name, model_results in all_results.items():

        best_threshold = None
        best_dice = -1.0

        for threshold in THRESHOLDS:

            metrics = calculate_global_metrics(
                model_results["global"][threshold]
            )

            if metrics["dice"] > best_dice:

                best_dice = metrics["dice"]
                best_threshold = threshold

        rankings.append(
            (name, best_dice, best_threshold)
        )

    rankings.sort(
        key=lambda item: item[1],
        reverse=True
    )

    print(
        f"{'Model':<28}"
        f"{'Best Dice':<15}"
        f"{'Threshold':<15}"
    )

    for name, dice, threshold in rankings:

        print(
            f"{name:<28}"
            f"{dice:<15.4f}"
            f"{threshold:<15.2f}"
        )


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 85)
    print("SIX-MODEL STENOSIS COMPARISON")
    print("=" * 85)

    check_checkpoints()

    print("\nLoading validation dataset...")

    dataset = StenosisDataset("val")

    print("Validation images:", len(dataset))

    all_results = {}

    for name, path in CHECKPOINTS.items():

        print("\n" + "#" * 85)
        print("EVALUATING:", name)
        print("#" * 85)

        model = load_model(name, path)

        results, global_counts = evaluate_model(
            model,
            dataset
        )

        all_results[name] = {
            "macro": results,
            "global": global_counts
        }

        print_results(name, results)

        print_global_results(
            name,
            global_counts
        )

        del model

    # Compare all six models.
    print_model_ranking(all_results)

    print_overall_ranking(all_results)

    print("\n" + "=" * 85)
    print("COMPARISON COMPLETED")
    print("=" * 85)

    print("Models evaluated:", len(CHECKPOINTS))
    print("Validation images:", len(dataset))
    print("Thresholds tested:", THRESHOLDS)

    print("\nAll checkpoints remain untouched.")


if __name__ == "__main__":

    main()
