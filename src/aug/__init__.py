"""
Augmentation subpackage for advanced dice detection augmentation techniques.

This package contains:
- annotation_manager: Load and manage different annotation sets
- difficulty_sampling: Difficulty-aware sampling strategies
- copy_paste: Instance-level copy-paste augmentation
"""

from .annotation_manager import AnnotationManager
from .difficulty_sampling import DifficultyAwareSampler, calculate_instance_difficulty
from .copy_paste import CopyPasteAugmentation, MosaicWithCopyPaste

__all__ = [
    'AnnotationManager',
    'DifficultyAwareSampler',
    'calculate_instance_difficulty',
    'CopyPasteAugmentation',
    'MosaicWithCopyPaste'
]
