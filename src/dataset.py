"""
Dataset handling for dice detection
Supports loading from Roboflow and local datasets
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from typing import Dict, List, Tuple, Optional


class DiceDetectionDataset(Dataset):
    """
    Dataset class for dice detection with bounding boxes
    
    Expected annotation format (COCO-style):
    {
        "images": [{"id": 1, "file_name": "image.jpg", "width": 640, "height": 480}, ...],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [x, y, w, h]}, ...],
        "categories": [{"id": 1, "name": "dice-1"}, ...]
    }
    """
    
    def __init__(
        self, 
        root_dir: str, 
        annotation_file: str = "annotations.json",
        transforms: Optional[callable] = None,
        split: str = "train"
    ):
        """
        Args:
            root_dir: Root directory containing images
            annotation_file: Path to COCO-format annotation JSON file
            transforms: Optional transforms to apply
            split: Dataset split ('train', 'val', 'test')
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.split = split
        
        # Load annotations
        annotation_path = os.path.join(root_dir, annotation_file)
        with open(annotation_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build image id to annotations mapping
        self.image_info = {img['id']: img for img in self.coco_data['images']}
        self.image_ids = list(self.image_info.keys())
        
        # Build category mapping
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.num_classes = len(self.categories) + 1  # +1 for background
        
        # Group annotations by image
        self.img_to_anns = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Returns:
            image: Tensor of shape (C, H, W)
            target: Dict containing boxes, labels, image_id, area, iscrowd
        """
        img_id = self.image_ids[idx]
        img_info = self.image_info[img_id]
        
        # Load image
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")
        
        # Get annotations for this image
        anns = self.img_to_anns.get(img_id, [])
        
        # Extract boxes and labels
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in anns:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(w * h)
            iscrowd.append(ann.get('iscrowd', 0))
        
        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            # Handle images with no annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd
        }
        
        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        else:
            # Default: convert to tensor
            image = T.ToTensor()(image)
        
        return image, target
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Return the distribution of classes in the dataset"""
        distribution = {name: 0 for name in self.categories.values()}
        
        for anns in self.img_to_anns.values():
            for ann in anns:
                cat_name = self.categories[ann['category_id']]
                distribution[cat_name] += 1
        
        return distribution


def collate_fn(batch):
    """Custom collate function for batching variable number of objects"""
    return tuple(zip(*batch))


def get_transform(train: bool = True):
    """
    Get standard transforms for the dataset
    
    Args:
        train: If True, returns training transforms, else validation transforms
    """
    transforms = []
    transforms.append(T.ToTensor())
    
    # Add data augmentation for training
    if train:
        # Additional augmentations can be added here
        pass
    
    return T.Compose(transforms)
