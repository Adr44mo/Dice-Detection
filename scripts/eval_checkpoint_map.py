#!/usr/bin/env python3
"""Compute mAP on the test split for a saved checkpoint.

Update the configuration constants below to point to the checkpoint and dataset
you want to evaluate.
"""

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path when running from scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DiceDetectionDataset, collate_fn
from src.model import get_fasterrcnn_model, load_model_checkpoint
from src.metrics import evaluate_map, print_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "checkpoints/baseline.pth"
DATASET_ROOT = os.path.join("dice-2", "test")
ANNOTATION_FILE = "_annotations.coco.json"
OUTPUT_JSON = None  # e.g. "results/test_metrics.json"
USE_CUSTOM_ANNOTATION = True
CUSTOM_ANNOTATION_PATH = os.path.join("Annotations", "test_balanced.coco.json")

BATCH_SIZE = 8
NUM_WORKERS = 4
TRAINABLE_BACKBONE_LAYERS = 3
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.05
DEVICE = None  # Set to "cuda"/"cpu" to override auto-selection


def infer_num_classes_from_checkpoint(checkpoint_path: str) -> int:
    """Return the classifier output dimension (including background)."""

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    weight_key = "roi_heads.box_predictor.cls_score.weight"

    if weight_key not in state_dict:
        raise KeyError(
            f"Missing {weight_key} in checkpoint: cannot infer class count"
        )

    return state_dict[weight_key].shape[0]


def resolve_device(requested: Optional[str]) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not os.path.exists(DATASET_ROOT):
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    if USE_CUSTOM_ANNOTATION:
        if not os.path.exists(CUSTOM_ANNOTATION_PATH):
            raise FileNotFoundError(
                "Custom annotation file not found: "
                f"{CUSTOM_ANNOTATION_PATH}"
            )
        target_annotation = os.path.join(DATASET_ROOT, ANNOTATION_FILE)
        os.makedirs(DATASET_ROOT, exist_ok=True)
        shutil.copy(CUSTOM_ANNOTATION_PATH, target_annotation)

    device = resolve_device(DEVICE)

    dataset = DiceDetectionDataset(
        root_dir=DATASET_ROOT,
        annotation_file=ANNOTATION_FILE,
        split="test"
    )
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    checkpoint_num_classes = infer_num_classes_from_checkpoint(CHECKPOINT_PATH)
    if checkpoint_num_classes != dataset.num_classes:
        print(
            "[info] dataset reports"
            f" {dataset.num_classes} classes (incl. background)"
            f" but checkpoint was trained with {checkpoint_num_classes}."
            " Using checkpoint value to build the model."
        )

    model = get_fasterrcnn_model(
        num_classes=checkpoint_num_classes,
        pretrained=False,
        trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS
    )
    model = load_model_checkpoint(model, CHECKPOINT_PATH, device=str(device))
    model.to(device)

    metrics = evaluate_map(
        model,
        data_loader,
        device=str(device),
        iou_threshold=IOU_THRESHOLD,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    print_metrics(metrics, class_names=dataset.categories)

    if OUTPUT_JSON:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as fp:
            json.dump(metrics, fp, indent=2)
        print(f"Metrics saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
