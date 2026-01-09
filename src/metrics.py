"""
Evaluation metrics for object detection
Implements mAP (mean Average Precision) calculation
"""

import torch
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU (Intersection over Union) between two boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def calculate_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Calculate Average Precision (AP) using 11-point interpolation
    
    Args:
        recalls: Array of recall values
        precisions: Array of precision values
        
    Returns:
        Average Precision
    """
    # Append sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Compute the precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
    
    # Calculate AP using 11-point interpolation
    recall_thresholds = np.linspace(0, 1, 11)
    ap = 0.0
    for threshold in recall_thresholds:
        idx = np.where(recalls >= threshold)[0]
        if len(idx) > 0:
            ap += precisions[idx[0]]
    
    return ap / 11.0


@torch.no_grad()
def evaluate_map(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.05
) -> Dict[str, float]:
    """
    Evaluate model using mAP metric
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device
        iou_threshold: IoU threshold for considering a detection as correct
        confidence_threshold: Minimum confidence for predictions
        
    Returns:
        Dictionary with mAP and per-class AP
    """
    model.eval()
    
    # Store all predictions and ground truths
    all_predictions = defaultdict(list)  # class_id -> [(image_id, confidence, box), ...]
    all_ground_truths = defaultdict(list)  # class_id -> [(image_id, box), ...]
    num_gt_per_class = defaultdict(int)
    
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        
        # Get predictions
        predictions = model(images)
        
        # Process each image
        for i, (pred, target) in enumerate(zip(predictions, targets)):
            image_id = target['image_id'].item()
            
            # Store ground truths
            gt_boxes = target['boxes'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            
            for box, label in zip(gt_boxes, gt_labels):
                all_ground_truths[label].append((image_id, box))
                num_gt_per_class[label] += 1
            
            # Store predictions
            pred_boxes = pred['boxes'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            
            # Filter by confidence
            keep = pred_scores >= confidence_threshold
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]
            
            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                all_predictions[label].append((image_id, score, box))
    
    # Calculate AP for each class
    ap_per_class = {}
    
    for class_id in num_gt_per_class.keys():
        # Get predictions for this class
        preds = all_predictions.get(class_id, [])
        
        if len(preds) == 0:
            ap_per_class[int(class_id)] = 0.0
            continue
        
        # Sort by confidence
        preds = sorted(preds, key=lambda x: x[1], reverse=True)
        
        # Match predictions to ground truths
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        
        # Group ground truths by image
        gt_by_image = defaultdict(list)
        for img_id, box in all_ground_truths[class_id]:
            gt_by_image[img_id].append(box)
        
        # Track which ground truths have been matched
        matched_gt = defaultdict(set)
        
        for i, (img_id, conf, pred_box) in enumerate(preds):
            gt_boxes = gt_by_image.get(img_id, [])
            
            max_iou = 0
            max_idx = -1
            
            for j, gt_box in enumerate(gt_boxes):
                if j in matched_gt[img_id]:
                    continue
                
                iou = calculate_iou(pred_box, gt_box)
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            
            if max_iou >= iou_threshold and max_idx >= 0:
                tp[i] = 1
                matched_gt[img_id].add(max_idx)
            else:
                fp[i] = 1
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / num_gt_per_class[class_id]
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Calculate AP
        ap = calculate_ap(recalls, precisions)
        ap_per_class[int(class_id)] = ap
    
    # Calculate mAP
    if len(ap_per_class) > 0:
        mAP = np.mean(list(ap_per_class.values()))
    else:
        mAP = 0.0
    
    results = {
        'mAP': mAP,
        **{f'AP_class_{k}': v for k, v in ap_per_class.items()}
    }
    
    return results


def print_metrics(metrics: Dict[str, float], class_names: Dict[int, str] = None):
    """
    Print evaluation metrics in a readable format
    
    Args:
        metrics: Dictionary of metrics
        class_names: Optional mapping of class IDs to names
    """
    print("\n" + "="*50)
    print("Evaluation Metrics")
    print("="*50)
    
    # Print mAP
    if 'mAP' in metrics:
        print(f"mAP@0.5: {metrics['mAP']:.4f}")
        print("-"*50)
    
    # Print per-class AP
    for key, value in sorted(metrics.items()):
        if key.startswith('AP_class_'):
            class_id = int(key.split('_')[-1])
            class_name = class_names.get(class_id, f"Class {class_id}") if class_names else f"Class {class_id}"
            print(f"  {class_name}: {value:.4f}")
    
    print("="*50 + "\n")
