"""
Difficulty-Aware Sampling for dice detection.

This module implements sampling strategies that consider instance difficulty,
extending the basic class-aware sampling to prioritize hard examples.

Difficulty is calculated based on:
- Size: Smaller dice are harder to detect
- Edge proximity: Dice near image edges may be cut off
- Density: High overlap regions are more challenging
- Occlusion: Estimated based on bbox overlap
"""

import random
import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
from typing import List, Dict, Tuple
from collections import defaultdict


def calculate_instance_difficulty(
    bbox: List[float],
    image_size: Tuple[int, int],
    all_bboxes: List[List[float]] = None,
    weights: Dict[str, float] = None
) -> float:
    """
    Calculate difficulty score for a single instance.
    
    Args:
        bbox: Bounding box [x1, y1, x2, y2]
        image_size: (width, height) of the image
        all_bboxes: List of all bboxes in the image (for occlusion calculation)
        weights: Dictionary of weights for different difficulty factors
            Keys: 'size', 'edge', 'density', 'occlusion'
    
    Returns:
        Difficulty score (higher = more difficult)
    """
    if weights is None:
        weights = {
            'size': 1.0,
            'edge': 0.5,
            'density': 0.3,
            'occlusion': 0.7
        }
    
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height
    img_w, img_h = image_size
    img_area = img_w * img_h
    
    difficulty = 0.0
    
    # Size difficulty: smaller objects are harder
    # Normalize by image area
    relative_size = area / img_area
    size_difficulty = 1.0 - min(relative_size * 10, 1.0)  # Cap at 1.0
    difficulty += weights['size'] * size_difficulty
    
    # Edge proximity: objects near edges might be cut off
    edge_margin = 20  # pixels
    distance_to_edge = min(x1, y1, img_w - x2, img_h - y2)
    edge_difficulty = max(0, (edge_margin - distance_to_edge) / edge_margin)
    difficulty += weights['edge'] * edge_difficulty
    
    # Density and occlusion (if other boxes provided)
    if all_bboxes is not None and len(all_bboxes) > 1:
        # Count overlapping boxes
        overlaps = 0
        total_overlap_iou = 0.0
        
        for other_bbox in all_bboxes:
            if other_bbox == bbox:
                continue
            
            # Calculate IoU
            ox1, oy1, ox2, oy2 = other_bbox
            
            # Intersection
            ix1 = max(x1, ox1)
            iy1 = max(y1, oy1)
            ix2 = min(x2, ox2)
            iy2 = min(y2, oy2)
            
            if ix1 < ix2 and iy1 < iy2:
                intersection = (ix2 - ix1) * (iy2 - iy1)
                other_area = (ox2 - ox1) * (oy2 - oy1)
                union = area + other_area - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > 0.1:  # Significant overlap
                    overlaps += 1
                    total_overlap_iou += iou
        
        # Density difficulty
        density_difficulty = min(overlaps / 3.0, 1.0)  # Normalize by 3 overlaps
        difficulty += weights['density'] * density_difficulty
        
        # Occlusion difficulty
        occlusion_difficulty = min(total_overlap_iou, 1.0)
        difficulty += weights['occlusion'] * occlusion_difficulty
    
    return difficulty


class DifficultyAwareSampler(Sampler):
    """
    Sampler that balances both class distribution and instance difficulty.
    
    This sampler:
    1. Builds a difficulty score for each instance in each image
    2. Samples images with probability proportional to:
       - Class balance (like ClassAwareSampler)
       - Average difficulty of instances in the image
    3. Prioritizes underrepresented classes AND hard examples
    
    Example usage:
        sampler = DifficultyAwareSampler(
            dataset,
            samples_per_epoch=len(dataset) * 2,
            difficulty_weight=0.5  # Balance between class and difficulty
        )
    """
    
    def __init__(
        self,
        dataset: Dataset,
        samples_per_epoch: int = None,
        difficulty_weight: float = 0.5,
        difficulty_weights: Dict[str, float] = None,
    ):
        """
        Args:
            dataset: Dataset with COCO-style annotations
            samples_per_epoch: Number of samples per epoch (default: len(dataset))
            difficulty_weight: Weight for difficulty vs class balance (0-1)
                0 = pure class balance, 1 = pure difficulty
            difficulty_weights: Weights for difficulty calculation components
        """
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch or len(dataset)
        self.difficulty_weight = difficulty_weight
        self.difficulty_calc_weights = difficulty_weights
        
        # Build difficulty and class indices
        self._build_indices()
    
    def _build_indices(self):
        """Build mapping of classes and difficulties."""
        # Class indices (like ClassAwareSampler)
        self.class_indices = defaultdict(list)
        
        # Image difficulties
        self.image_difficulties = {}
        
        # Class difficulties (average difficulty per class)
        class_difficulties = defaultdict(list)
        
        print("Calculating difficulty scores for all instances...")
        
        for idx in range(len(self.dataset)):
            try:
                _, target = self.dataset[idx]
                
                # Get image size
                # Assuming the dataset stores image info or we can infer from boxes
                boxes = target['boxes']
                labels = target['labels']
                
                if len(boxes) == 0:
                    continue
                
                # Estimate image size from max bbox coordinates
                if isinstance(boxes, torch.Tensor):
                    boxes_np = boxes.numpy()
                else:
                    boxes_np = np.array(boxes)
                
                img_w = int(np.max(boxes_np[:, 2]) + 10)
                img_h = int(np.max(boxes_np[:, 3]) + 10)
                image_size = (max(img_w, 640), max(img_h, 640))
                
                # Calculate difficulty for each instance
                difficulties = []
                for box_idx, (bbox, label) in enumerate(zip(boxes_np, labels)):
                    difficulty = calculate_instance_difficulty(
                        bbox.tolist(),
                        image_size,
                        boxes_np.tolist(),
                        self.difficulty_calc_weights
                    )
                    difficulties.append(difficulty)
                    
                    # Track class difficulties
                    label_val = int(label)
                    class_difficulties[label_val].append(difficulty)
                    
                    # Add to class indices
                    if idx not in self.class_indices[label_val]:
                        self.class_indices[label_val].append(idx)
                
                # Store average difficulty for this image
                self.image_difficulties[idx] = np.mean(difficulties) if difficulties else 0.0
                
            except Exception as e:
                print(f"Warning: Could not process image {idx}: {e}")
                self.image_difficulties[idx] = 0.0
        
        # Calculate class statistics
        self.classes = list(self.class_indices.keys())
        self.class_avg_difficulties = {
            cls: np.mean(diffs) if diffs else 0.0
            for cls, diffs in class_difficulties.items()
        }
        
        # Calculate sampling probabilities
        self._calculate_sampling_probs()
        
        print(f"Difficulty-aware sampler initialized:")
        print(f"  Classes: {len(self.classes)}")
        print(f"  Images with annotations: {len(self.image_difficulties)}")
        print(f"  Average difficulty by class:")
        for cls in sorted(self.classes):
            print(f"    Class {cls}: {self.class_avg_difficulties[cls]:.3f}")
    
    def _calculate_sampling_probs(self):
        """Calculate sampling probabilities for each image."""
        self.image_probs = {}
        
        # Base probability from class balance
        class_counts = {cls: len(indices) for cls, indices in self.class_indices.items()}
        total_instances = sum(class_counts.values())
        
        # Inverse frequency for class balance
        class_weights = {
            cls: total_instances / (len(self.classes) * count)
            for cls, count in class_counts.items()
        }
        
        # Calculate probability for each image
        for idx in range(len(self.dataset)):
            if idx not in self.image_difficulties:
                self.image_probs[idx] = 0.0
                continue
            
            # Get classes in this image
            _, target = self.dataset[idx]
            labels = target['labels']
            
            if len(labels) == 0:
                self.image_probs[idx] = 0.0
                continue
            
            # Average class weight for this image
            img_class_weight = np.mean([
                class_weights.get(int(label), 1.0) for label in labels
            ])
            
            # Difficulty weight
            img_difficulty = self.image_difficulties[idx]
            
            # Combined probability
            # Higher difficulty = higher probability
            # Higher class weight (rarer class) = higher probability
            class_prob = (1 - self.difficulty_weight) * img_class_weight
            diff_prob = self.difficulty_weight * (img_difficulty + 0.1)  # +0.1 to avoid zeros
            
            self.image_probs[idx] = class_prob + diff_prob
        
        # Normalize probabilities
        total_prob = sum(self.image_probs.values())
        if total_prob > 0:
            self.image_probs = {
                idx: prob / total_prob
                for idx, prob in self.image_probs.items()
            }
    
    def __iter__(self):
        """Generate indices with difficulty-aware sampling."""
        indices = []
        image_indices = list(self.image_probs.keys())
        probabilities = [self.image_probs[idx] for idx in image_indices]
        
        # Sample with replacement based on probabilities
        for _ in range(self.samples_per_epoch):
            idx = np.random.choice(image_indices, p=probabilities)
            indices.append(idx)
        
        return iter(indices)
    
    def __len__(self):
        return self.samples_per_epoch
