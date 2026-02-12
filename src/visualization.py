"""
Visualization utilities for dice detection
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

import seaborn as sns


def visualize_predictions(
    image: torch.Tensor,
    predictions: Dict,
    class_names: Dict[int, str] = None,
    confidence_threshold: float = 0.5,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Visualize object detection predictions
    
    Args:
        image: Image tensor (C, H, W)
        predictions: Dictionary with 'boxes', 'labels', 'scores'
        class_names: Mapping of class IDs to names
        confidence_threshold: Minimum confidence to display
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Convert tensor to numpy
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    
    # Normalize if needed
    if image.max() > 1.0:
        image = image / 255.0
    
    image = np.clip(image, 0, 1)
    
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    
    # Filter by confidence
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    keep = scores >= confidence_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    
    # Define colors for different classes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        color = colors[label % 10]
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        class_name = class_names.get(label, f"Class {label}") if class_names else f"Class {label}"
        label_text = f"{class_name}: {score:.2f}"
        
        ax.text(
            x1, y1 - 5,
            label_text,
            color='white',
            fontsize=10,
            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=1)
        )
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()


def visualize_ground_truth(
    image: torch.Tensor,
    target: Dict,
    class_names: Dict[int, str] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
):
    """
    Visualize ground truth annotations
    
    Args:
        image: Image tensor (C, H, W)
        target: Target dictionary with 'boxes' and 'labels'
        class_names: Mapping of class IDs to names
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Convert tensor to numpy
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    
    if image.max() > 1.0:
        image = image / 255.0
    
    image = np.clip(image, 0, 1)
    
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    
    boxes = target['boxes'].cpu().numpy()
    labels = target['labels'].cpu().numpy()
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        color = colors[label % 10]
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        class_name = class_names.get(label, f"Class {label}") if class_names else f"Class {label}"
        
        ax.text(
            x1, y1 - 5,
            class_name,
            color='white',
            fontsize=10,
            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=1)
        )
    
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
):
    """
    Plot training history
    
    Args:
        history: Dictionary with metric names and their values over epochs
        figsize: Figure size
        save_path: Optional path to save figure
    """
    num_metrics = len(history)
    fig, axes = plt.subplots(1, num_metrics, figsize=figsize)
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, (metric_name, values) in zip(axes, history.items()):
        ax.plot(values, marker='o', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()


def plot_class_distribution(
    distribution: Dict[str, int],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
):
    """
    Plot class distribution as a bar chart
    
    Args:
        distribution: Dictionary mapping class names to counts
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    classes = list(distribution.keys())
    counts = list(distribution.values())
    
    colors = sns.color_palette("husl", len(classes))
    bars = ax.bar(classes, counts, color=colors)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom'
        )
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Class Distribution in Dataset', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()


def plot_ap_comparison(
    baseline_ap: Dict[int, float],
    improved_ap: Dict[int, float],
    class_names: Dict[int, str] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
):
    """
    Compare AP scores between baseline and improved models
    
    Args:
        baseline_ap: Baseline AP per class
        improved_ap: Improved AP per class
        class_names: Mapping of class IDs to names
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    classes = sorted(baseline_ap.keys())
    class_labels = [class_names.get(c, f"Class {c}") if class_names else f"Class {c}" for c in classes]
    
    baseline_values = [baseline_ap[c] for c in classes]
    improved_values = [improved_ap.get(c, 0) for c in classes]
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
    bars2 = ax.bar(x + width/2, improved_values, width, label='With Augmentation', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Average Precision (AP)', fontsize=12)
    ax.set_title('Per-Class AP Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show()


def display_sample_batch(
    dataset: torch.utils.data.Dataset,
    num_samples: int = 4,
    figsize: Tuple[int, int] = (15, 10),
    class_names: Dict[int, str] = None
):
    """
    Display a grid of sample images from dataset
    
    Args:
        dataset: Dataset to sample from
        num_samples: Number of samples to display
        figsize: Figure size
        class_names: Mapping of class IDs to names
    """
    rows = int(np.ceil(num_samples / 2))
    fig, axes = plt.subplots(rows, 2, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(num_samples):
        idx = np.random.randint(0, len(dataset))
        image, target = dataset[idx]
        
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
        
        if image.max() > 1.0:
            image = image / 255.0
        
        image = np.clip(image, 0, 1)
        
        axes[i].imshow(image)
        
        # Draw boxes
        boxes = target['boxes'].cpu().numpy()
        labels = target['labels'].cpu().numpy()
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            color = colors[label % 10]
            
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            axes[i].add_patch(rect)
        
        axes[i].axis('off')
        axes[i].set_title(f'Sample {idx} - {len(boxes)} objects')
    
    # Hide extra subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
