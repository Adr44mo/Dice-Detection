"""
Instance-Level Copy-Paste Augmentation for dice detection.

This module implements copy-paste augmentation that extracts individual dice
instances from source images and pastes them onto target images with:
- Intelligent placement (avoiding overlaps)
- Optional transformations (rotation, scaling)
- Edge blending for natural appearance
- Proper annotation handling

Can be used standalone or combined with mosaic augmentation.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageEnhance
import torchvision.transforms.functional as F
from typing import List, Tuple, Dict, Optional


def extract_instance(
    image: Image.Image,
    bbox: List[float],
    padding: int = 5
) -> Tuple[Image.Image, List[float]]:
    """
    Extract a single instance from an image using its bounding box.
    
    Args:
        image: Source PIL Image
        bbox: Bounding box [x1, y1, x2, y2]
        padding: Extra padding around the bbox
    
    Returns:
        Tuple of (cropped_image, adjusted_bbox)
    """
    x1, y1, x2, y2 = bbox
    img_w, img_h = image.size
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img_w, x2 + padding)
    y2 = min(img_h, y2 + padding)
    
    # Crop
    cropped = image.crop((int(x1), int(y1), int(x2), int(y2)))
    
    # Adjusted bbox relative to cropped image
    adjusted_bbox = [padding, padding, x2 - x1 - padding, y2 - y1 - padding]
    
    return cropped, adjusted_bbox


def check_overlap(new_box: List[float], existing_boxes: List[List[float]], min_iou: float = 0.1) -> bool:
    """
    Check if a new box overlaps significantly with existing boxes.
    
    Args:
        new_box: [x1, y1, x2, y2]
        existing_boxes: List of existing boxes
        min_iou: Minimum IoU to consider as overlap
    
    Returns:
        True if significant overlap detected
    """
    x1, y1, x2, y2 = new_box
    area = (x2 - x1) * (y2 - y1)
    
    for box in existing_boxes:
        bx1, by1, bx2, by2 = box
        
        # Intersection
        ix1 = max(x1, bx1)
        iy1 = max(y1, by1)
        ix2 = min(x2, bx2)
        iy2 = min(y2, by2)
        
        if ix1 < ix2 and iy1 < iy2:
            intersection = (ix2 - ix1) * (iy2 - iy1)
            other_area = (bx2 - bx1) * (by2 - by1)
            union = area + other_area - intersection
            iou = intersection / union if union > 0 else 0
            
            if iou > min_iou:
                return True
    
    return False


class CopyPasteAugmentation:
    """
    Copy-paste augmentation that extracts dice instances and pastes them.
    
    This augmentation:
    1. Randomly selects instances from source images
    2. Extracts them with some context
    3. Optionally transforms them (rotate, scale, color adjust)
    4. Pastes them onto target image avoiding overlaps
    5. Updates annotations accordingly
    
    Example usage:
        copy_paste = CopyPasteAugmentation(
            dataset,
            prob=0.5,
            num_instances=(1, 3)
        )
        augmented_img, augmented_target = copy_paste(target_idx)
    """
    
    def __init__(
        self,
        dataset: Dataset,
        prob: float = 0.5,
        num_instances: Tuple[int, int] = (1, 4),
        scale_range: Tuple[float, float] = (0.8, 1.2),
        rotation_range: Tuple[int, int] = (-15, 15),
        blend: bool = True,
        max_attempts: int = 20
    ):
        """
        Args:
            dataset: Source dataset to sample instances from
            prob: Probability of applying copy-paste
            num_instances: (min, max) number of instances to paste
            scale_range: (min, max) scale factor for pasted instances
            rotation_range: (min_deg, max_deg) rotation range
            blend: Whether to apply edge blending
            max_attempts: Maximum attempts to place each instance
        """
        self.dataset = dataset
        self.prob = prob
        self.num_instances = num_instances
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.blend = blend
        self.max_attempts = max_attempts
    
    def __call__(self, idx: int) -> Tuple[Image.Image, Dict]:
        """
        Apply copy-paste augmentation to an image.
        
        Args:
            idx: Index of target image
        
        Returns:
            Augmented image and updated target dict
        """
        # Get target image and annotations
        image, target = self.dataset[idx]
        
        # Convert tensor to PIL if needed
        if isinstance(image, torch.Tensor):
            image = F.to_pil_image(image)
        
        # Decide whether to apply
        if random.random() > self.prob:
            return image, target
        
        # Start with existing boxes
        existing_boxes = target['boxes'].clone() if isinstance(target['boxes'], torch.Tensor) else torch.tensor(target['boxes'])
        existing_labels = target['labels'].clone() if isinstance(target['labels'], torch.Tensor) else torch.tensor(target['labels'])
        
        existing_boxes_list = existing_boxes.tolist() if len(existing_boxes) > 0 else []
        
        # Number of instances to paste
        num_to_paste = random.randint(*self.num_instances)
        
        new_boxes = []
        new_labels = []
        
        for _ in range(num_to_paste):
            # Sample a random source image
            source_idx = random.randint(0, len(self.dataset) - 1)
            source_img, source_target = self.dataset[source_idx]
            
            if isinstance(source_img, torch.Tensor):
                source_img = F.to_pil_image(source_img)
            
            source_boxes = source_target['boxes']
            source_labels = source_target['labels']
            
            if len(source_boxes) == 0:
                continue
            
            # Select a random instance from source
            instance_idx = random.randint(0, len(source_boxes) - 1)
            instance_bbox = source_boxes[instance_idx].tolist() if isinstance(source_boxes, torch.Tensor) else source_boxes[instance_idx]
            instance_label = source_labels[instance_idx].item() if isinstance(source_labels, torch.Tensor) else source_labels[instance_idx]
            
            # Extract instance
            try:
                instance_img, _ = extract_instance(source_img, instance_bbox, padding=5)
            except Exception:
                continue
            
            # Apply transformations
            # Scale
            if self.scale_range != (1.0, 1.0):
                scale = random.uniform(*self.scale_range)
                new_size = (int(instance_img.width * scale), int(instance_img.height * scale))
                instance_img = instance_img.resize(new_size, Image.BILINEAR)
            
            # Rotation
            if self.rotation_range != (0, 0):
                angle = random.randint(*self.rotation_range)
                instance_img = instance_img.rotate(angle, expand=True, fillcolor=(0, 0, 0))
            
            # Color jitter
            if random.random() < 0.5:
                instance_img = ImageEnhance.Brightness(instance_img).enhance(random.uniform(0.8, 1.2))
                instance_img = ImageEnhance.Contrast(instance_img).enhance(random.uniform(0.8, 1.2))
            
            # Try to place on target image
            inst_w, inst_h = instance_img.size
            img_w, img_h = image.size
            
            placed = False
            for attempt in range(self.max_attempts):
                # Random position
                x = random.randint(0, max(0, img_w - inst_w))
                y = random.randint(0, max(0, img_h - inst_h))
                
                new_box = [x, y, x + inst_w, y + inst_h]
                
                # Check overlap
                if not check_overlap(new_box, existing_boxes_list, min_iou=0.2):
                    # Paste instance
                    if self.blend:
                        # Apply Gaussian blur to edges for blending
                        mask = Image.new('L', instance_img.size, 255)
                        mask = mask.filter(ImageFilter.GaussianBlur(2))
                        image.paste(instance_img, (x, y), mask)
                    else:
                        image.paste(instance_img, (x, y))
                    
                    new_boxes.append(new_box)
                    new_labels.append(instance_label)
                    existing_boxes_list.append(new_box)
                    placed = True
                    break
            
            if not placed:
                # Could not place this instance, skip
                pass
        
        # Combine with existing annotations
        if len(new_boxes) > 0:
            new_boxes_tensor = torch.tensor(new_boxes, dtype=torch.float32)
            new_labels_tensor = torch.tensor(new_labels, dtype=torch.int64)
            
            combined_boxes = torch.cat([existing_boxes, new_boxes_tensor], dim=0)
            combined_labels = torch.cat([existing_labels, new_labels_tensor], dim=0)
            
            target['boxes'] = combined_boxes
            target['labels'] = combined_labels
            target['area'] = (combined_boxes[:, 2] - combined_boxes[:, 0]) * (combined_boxes[:, 3] - combined_boxes[:, 1])
            target['iscrowd'] = torch.zeros((len(combined_boxes),), dtype=torch.int64)
        
        return image, target


class MosaicWithCopyPaste:
    """
    Enhanced mosaic augmentation that incorporates copy-paste.
    
    This combines:
    1. Regular mosaic (4 images in 2x2 grid)
    2. Additional copy-paste instances on top
    
    This creates even more diverse training scenes.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        mosaic_aug,  # MosaicAugmentation instance
        copy_paste_aug,  # CopyPasteAugmentation instance
        prob: float = 0.5
    ):
        """
        Args:
            dataset: Source dataset
            mosaic_aug: MosaicAugmentation instance
            copy_paste_aug: CopyPasteAugmentation instance
            prob: Probability of applying copy-paste after mosaic
        """
        self.dataset = dataset
        self.mosaic_aug = mosaic_aug
        self.copy_paste_aug = copy_paste_aug
        self.prob = prob
    
    def __call__(self, idx: int) -> Tuple[Image.Image, Dict]:
        """
        Apply mosaic then copy-paste augmentation.
        
        Args:
            idx: Index of primary image
        
        Returns:
            Augmented image and target
        """
        # First apply mosaic
        image, target = self.mosaic_aug(idx)
        
        # Then optionally apply copy-paste
        if random.random() < self.prob:
            # Create a temporary "dataset item" for copy-paste
            # Note: copy-paste expects dataset[idx] format
            # We'll create a mock by modifying the dataset temporarily
            
            # Store original
            original_item = self.dataset[idx]
            
            # Temporarily override
            temp_target = target.copy()
            self.dataset._temp_override = (image, temp_target)
            
            # Apply copy-paste (it will get the mosaic'd image)
            image, target = self.copy_paste_aug(idx)
            
            # Clean up
            if hasattr(self.dataset, '_temp_override'):
                delattr(self.dataset, '_temp_override')
        
        return image, target
