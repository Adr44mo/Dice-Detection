"""
Data augmentation techniques for dice detection
Includes class-aware sampling and mosaic augmentation
"""

import random
import numpy as np
import torch
from torch.utils.data import Sampler, Dataset
from PIL import Image, ImageDraw
import torchvision.transforms.functional as F
from typing import List, Tuple, Dict

# Import new augmentation modules
from .aug.annotation_manager import AnnotationManager
from .aug.difficulty_sampling import DifficultyAwareSampler, calculate_instance_difficulty
from .aug.copy_paste import CopyPasteAugmentation, MosaicWithCopyPaste


class ClassAwareSampler(Sampler):
    """
    Sampler that balances classes during training
    Samples images based on their class distribution
    """
    
    def __init__(
        self, 
        dataset: Dataset, 
        samples_per_epoch: int = None,
        balance_by: str = 'dice_value'  # 'dice_value' or 'num_dices'
    ):
        """
        Args:
            dataset: Dataset with get_class_distribution method
            samples_per_epoch: Number of samples per epoch (default: len(dataset))
            balance_by: How to balance ('dice_value' for dice faces, 'num_dices' for count)
        """
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch or len(dataset)
        self.balance_by = balance_by
        
        # Build indices for each class
        self.class_indices = self._build_class_indices()
        self.classes = list(self.class_indices.keys())
        
        if len(self.classes) == 0:
            raise ValueError("No classes found in dataset")
    
    def _build_class_indices(self) -> Dict[int, List[int]]:
        """Build mapping of class to image indices"""
        class_indices = {}
        
        for idx in range(len(self.dataset)):
            _, target = self.dataset[idx]
            labels = target['labels'].numpy()
            
            for label in labels:
                label = int(label)
                if label not in class_indices:
                    class_indices[label] = []
                if idx not in class_indices[label]:
                    class_indices[label].append(idx)
        
        return class_indices
    
    def __iter__(self):
        """Generate indices with class balancing"""
        indices = []
        
        for _ in range(self.samples_per_epoch):
            # Randomly select a class
            class_id = random.choice(self.classes)
            # Randomly select an image containing that class
            idx = random.choice(self.class_indices[class_id])
            indices.append(idx)
        
        return iter(indices)
    
    def __len__(self):
        return self.samples_per_epoch


class MosaicAugmentation:
    """
    Mosaic augmentation: combines 4 images into a 2x2 grid
    Increases diversity of object scales and contexts
    """
    
    def __init__(
        self,
        dataset: Dataset,
        output_size: Tuple[int, int] = (640, 640),
        prob: float = 0.5
    ):
        """
        Args:
            dataset: Source dataset
            output_size: Output image size (height, width)
            prob: Probability of applying mosaic augmentation
        """
        self.dataset = dataset
        self.output_size = output_size
        self.prob = prob
    
    def __call__(self, idx: int) -> Tuple[Image.Image, Dict]:
        """
        Apply mosaic augmentation
        
        Args:
            idx: Index of the main image
            
        Returns:
            Augmented image and combined targets
        """
        # Decide whether to apply mosaic
        if random.random() > self.prob:
            return self.dataset[idx]
        
        # Select 3 additional random images
        indices = [idx] + random.sample(range(len(self.dataset)), 3)
        
        # Load images and targets
        images = []
        targets_list = []
        for i in indices:
            img, target = self.dataset[i]
            if isinstance(img, torch.Tensor):
                img = F.to_pil_image(img)
            images.append(img)
            targets_list.append(target)
        
        # Create mosaic
        mosaic_img, mosaic_target = self._create_mosaic(images, targets_list)
        
        return mosaic_img, mosaic_target
    
    def _create_mosaic(
        self, 
        images: List[Image.Image], 
        targets: List[Dict]
    ) -> Tuple[Image.Image, Dict]:
        """
        Create a 2x2 mosaic from 4 images
        """
        output_h, output_w = self.output_size
        
        # Create new image
        mosaic = Image.new('RGB', (output_w, output_h))
        
        # Calculate split point (randomly choose center point)
        split_x = output_w // 2 + random.randint(-output_w // 4, output_w // 4)
        split_y = output_h // 2 + random.randint(-output_h // 4, output_h // 4)
        
        # Define quadrants: (x, y, w, h) for each position
        quadrants = [
            (0, 0, split_x, split_y),           # Top-left
            (split_x, 0, output_w - split_x, split_y),  # Top-right
            (0, split_y, split_x, output_h - split_y),  # Bottom-left
            (split_x, split_y, output_w - split_x, output_h - split_y)  # Bottom-right
        ]
        
        combined_boxes = []
        combined_labels = []
        
        for img, target, (qx, qy, qw, qh) in zip(images, targets, quadrants):
            # Resize image to fit quadrant
            img_resized = img.resize((qw, qh), Image.BILINEAR)
            
            # Paste into mosaic
            mosaic.paste(img_resized, (qx, qy))
            
            # Adjust bounding boxes
            orig_w, orig_h = img.size
            scale_x = qw / orig_w
            scale_y = qh / orig_h
            
            boxes = target['boxes'].clone() if isinstance(target['boxes'], torch.Tensor) else torch.tensor(target['boxes'])
            labels = target['labels'].clone() if isinstance(target['labels'], torch.Tensor) else torch.tensor(target['labels'])
            
            if len(boxes) > 0:
                # Scale boxes
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                
                # Translate boxes
                boxes[:, [0, 2]] += qx
                boxes[:, [1, 3]] += qy
                
                # Clip boxes to image boundaries
                boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, output_w)
                boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, output_h)
                
                # Remove invalid boxes (too small)
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]
                keep = (widths > 1) & (heights > 1)
                
                boxes = boxes[keep]
                labels = labels[keep]
                
                if len(boxes) > 0:
                    combined_boxes.append(boxes)
                    combined_labels.append(labels)
        
        # Combine all boxes and labels
        if len(combined_boxes) > 0:
            combined_boxes = torch.cat(combined_boxes, dim=0)
            combined_labels = torch.cat(combined_labels, dim=0)
        else:
            combined_boxes = torch.zeros((0, 4), dtype=torch.float32)
            combined_labels = torch.zeros((0,), dtype=torch.int64)
        
        # Create combined target
        combined_target = {
            'boxes': combined_boxes,
            'labels': combined_labels,
            'image_id': targets[0]['image_id'],  # Use first image's ID
            'area': (combined_boxes[:, 2] - combined_boxes[:, 0]) * (combined_boxes[:, 3] - combined_boxes[:, 1]),
            'iscrowd': torch.zeros((len(combined_boxes),), dtype=torch.int64)
        }
        
        return mosaic, combined_target


def apply_random_augmentations(image: Image.Image, target: Dict) -> Tuple[Image.Image, Dict]:
    """
    Apply random augmentations to image and target
    
    Args:
        image: PIL Image
        target: Target dictionary with boxes and labels
        
    Returns:
        Augmented image and target
    """
    # Random horizontal flip
    if random.random() < 0.5:
        image = F.hflip(image)
        boxes = target['boxes'].clone()
        width = image.width
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        target['boxes'] = boxes
    
    # Random color jitter
    if random.random() < 0.5:
        image = F.adjust_brightness(image, random.uniform(0.8, 1.2))
        image = F.adjust_contrast(image, random.uniform(0.8, 1.2))
        image = F.adjust_saturation(image, random.uniform(0.8, 1.2))
    
    return image, target
