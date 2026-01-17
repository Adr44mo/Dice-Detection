"""
Annotation Manager for loading and managing different COCO annotation sets.

This module provides utilities to:
- Discover all annotation files in the Annotations directory
- Load specific annotation sets
- Cache loaded annotations for performance
- Provide metadata about available datasets
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class AnnotationManager:
    """
    Manager for loading and caching COCO annotation files.
    
    Example usage:
        manager = AnnotationManager('./Annotations')
        available = manager.get_available_annotations()
        train_data = manager.load_annotation_set('train_balanced.coco.json')
    """
    
    def __init__(self, annotations_dir: str):
        """
        Initialize the annotation manager.
        
        Args:
            annotations_dir: Path to directory containing annotation JSON files
        """
        self.annotations_dir = Path(annotations_dir)
        if not self.annotations_dir.exists():
            raise ValueError(f"Annotations directory not found: {annotations_dir}")
        
        self._cache = {}  # Cache loaded annotations
        self._discover_annotations()
    
    def _discover_annotations(self):
        """Discover all .json files in the annotations directory."""
        self.available_files = {}
        
        for json_file in self.annotations_dir.glob('*.json'):
            filename = json_file.name
            
            # Parse filename to extract metadata
            parts = filename.replace('.coco.json', '').split('_')
            
            # Determine split (train/val/test)
            split = parts[0] if parts else 'unknown'
            
            # Determine type (balanced, gan, etc.)
            if 'gan' in filename.lower():
                dataset_type = 'gan'
            elif 'balanced' in filename.lower():
                dataset_type = 'balanced'
            elif 'zipfian' in filename.lower():
                dataset_type = 'zipfian'
            elif 'image' in filename.lower():
                dataset_type = 'image_balanced'
            else:
                dataset_type = 'standard'
            
            self.available_files[filename] = {
                'path': str(json_file),
                'split': split,
                'type': dataset_type,
                'size': json_file.stat().st_size
            }
    
    def get_available_annotations(self, split: Optional[str] = None) -> Dict[str, Dict]:
        """
        Get dictionary of available annotation files.
        
        Args:
            split: Optional filter by split ('train', 'val', 'test')
        
        Returns:
            Dictionary mapping filename to metadata
        """
        if split is None:
            return self.available_files.copy()
        
        return {
            name: meta for name, meta in self.available_files.items()
            if meta['split'] == split
        }
    
    def load_annotation_set(self, filename: str, use_cache: bool = True) -> Dict:
        """
        Load a specific annotation file.
        
        Args:
            filename: Name of the annotation file (e.g., 'train_balanced.coco.json')
            use_cache: If True, return cached version if available
        
        Returns:
            Dictionary with 'images', 'annotations', and 'categories'
        
        Raises:
            ValueError: If file not found
        """
        if filename not in self.available_files:
            raise ValueError(
                f"Annotation file '{filename}' not found. "
                f"Available files: {list(self.available_files.keys())}"
            )
        
        # Check cache
        if use_cache and filename in self._cache:
            return self._cache[filename]
        
        # Load from disk
        filepath = self.available_files[filename]['path']
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Validate COCO format
        required_keys = {'images', 'annotations', 'categories'}
        if not required_keys.issubset(data.keys()):
            raise ValueError(
                f"Invalid COCO format in {filename}. "
                f"Missing keys: {required_keys - data.keys()}"
            )
        
        # Cache and return
        self._cache[filename] = data
        return data
    
    def get_dataset_stats(self, filename: str) -> Dict:
        """
        Get statistics about a dataset without fully loading it.
        
        Args:
            filename: Name of the annotation file
        
        Returns:
            Dictionary with dataset statistics
        """
        data = self.load_annotation_set(filename)
        
        num_images = len(data['images'])
        num_annotations = len(data['annotations'])
        num_categories = len(data['categories'])
        
        # Calculate class distribution
        class_counts = {}
        for ann in data['annotations']:
            cat_id = ann['category_id']
            class_counts[cat_id] = class_counts.get(cat_id, 0) + 1
        
        return {
            'filename': filename,
            'num_images': num_images,
            'num_annotations': num_annotations,
            'num_categories': num_categories,
            'annotations_per_image': num_annotations / num_images if num_images > 0 else 0,
            'class_distribution': class_counts
        }
    
    def print_summary(self):
        """Print a summary of all available annotation files."""
        print("=" * 80)
        print("Available Annotation Sets")
        print("=" * 80)
        
        # Group by split
        splits = {}
        for filename, meta in self.available_files.items():
            split = meta['split']
            if split not in splits:
                splits[split] = []
            splits[split].append((filename, meta))
        
        for split in sorted(splits.keys()):
            print(f"\n{split.upper()} Split:")
            print("-" * 80)
            
            for filename, meta in sorted(splits[split]):
                print(f"  📄 {filename}")
                print(f"     Type: {meta['type']}")
                print(f"     Size: {meta['size'] / 1024:.1f} KB")
                
                # Try to get stats
                try:
                    stats = self.get_dataset_stats(filename)
                    print(f"     Images: {stats['num_images']}, "
                          f"Annotations: {stats['num_annotations']}, "
                          f"Avg/Image: {stats['annotations_per_image']:.1f}")
                except Exception as e:
                    print(f"     (Could not load stats: {e})")
                
                print()
        
        print("=" * 80)
    
    def get_recommended_sets(self) -> Dict[str, str]:
        """
        Get recommended annotation sets for train/val/test.
        
        Returns:
            Dictionary with 'train', 'val', 'test' keys
        """
        recommendations = {}
        
        # Prefer balanced datasets
        for split in ['train', 'val', 'test']:
            split_files = self.get_available_annotations(split=split)
            
            # Priority: balanced > gan > image_balanced > standard
            priority_order = ['balanced', 'gan', 'image_balanced', 'standard']
            
            for dataset_type in priority_order:
                for filename, meta in split_files.items():
                    if meta['type'] == dataset_type:
                        recommendations[split] = filename
                        break
                if split in recommendations:
                    break
        
        return recommendations
