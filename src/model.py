"""
Model definitions for dice detection using Faster R-CNN.
"""

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from typing import Optional
from src.Loss_function import RoIHeadsWithFocalLoss


def get_fasterrcnn_model(
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    min_size: int = 800,
    max_size: int = 1333,
    use_focal_loss: bool = False,
    alpha: float = 0.25,
    gamma: float = 2.0
) -> FasterRCNN:
    """
    Get Faster R-CNN model with ResNet50-FPN backbone.
    
    Args:
        num_classes: Number of classes (including background)
        pretrained: Use pretrained weights for backbone
        trainable_backbone_layers: Number of trainable backbone layers (0-5)
        min_size: Minimum image size for training
        max_size: Maximum image size for training
        use_focal_loss: Enable focal loss for class imbalance
        alpha: Focal loss alpha parameter
        gamma: Focal loss gamma parameter
        
    Returns:
        Faster R-CNN model
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT" if pretrained else None,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=min_size,
        max_size=max_size
    )
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if use_focal_loss:
        rh = model.roi_heads
        model.roi_heads = RoIHeadsWithFocalLoss(
            box_roi_pool=rh.box_roi_pool,
            box_head=rh.box_head,
            box_predictor=rh.box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=128,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
            alpha=alpha,
            gamma=gamma
        )

    return model



def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: str = "cuda"
) -> torch.nn.Module:
    """
    Load model from checkpoint
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Model with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model


def save_model_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    additional_info: Optional[dict] = None
):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
        additional_info: Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")
