"""
Training utilities for dice detection
"""

import torch
import time
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    epoch: int,
    print_freq: int = 10
) -> Dict[str, float]:
    """
    Train the model for one epoch
    
    Args:
        model: Model to train
        optimizer: Optimizer
        data_loader: Training data loader
        device: Device to train on
        epoch: Current epoch number
        print_freq: Print frequency
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    loss_meter = AverageMeter()
    loss_classifier_meter = AverageMeter()
    loss_box_reg_meter = AverageMeter()
    loss_objectness_meter = AverageMeter()
    loss_rpn_box_reg_meter = AverageMeter()
    
    start_time = time.time()
    
    for i, (images, targets) in enumerate(tqdm(data_loader, desc=f"Epoch {epoch}")):
        # Move to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Check for invalid loss
        if not torch.isfinite(losses):
            print(f"Loss is {losses}, stopping training")
            print(f"Loss dict: {loss_dict}")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Update metrics
        loss_meter.update(losses.item())
        loss_classifier_meter.update(loss_dict.get('loss_classifier', torch.tensor(0.0)).item())
        loss_box_reg_meter.update(loss_dict.get('loss_box_reg', torch.tensor(0.0)).item())
        loss_objectness_meter.update(loss_dict.get('loss_objectness', torch.tensor(0.0)).item())
        loss_rpn_box_reg_meter.update(loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0)).item())
        
        # Print progress
        if (i + 1) % print_freq == 0:
            print(f"  Batch [{i+1}/{len(data_loader)}] "
                  f"Loss: {loss_meter.avg:.4f} "
                  f"(Cls: {loss_classifier_meter.avg:.4f}, "
                  f"Box: {loss_box_reg_meter.avg:.4f}, "
                  f"Obj: {loss_objectness_meter.avg:.4f}, "
                  f"RPN: {loss_rpn_box_reg_meter.avg:.4f})")
    
    elapsed_time = time.time() - start_time
    
    return {
        'loss': loss_meter.avg,
        'loss_classifier': loss_classifier_meter.avg,
        'loss_box_reg': loss_box_reg_meter.avg,
        'loss_objectness': loss_objectness_meter.avg,
        'loss_rpn_box_reg': loss_rpn_box_reg_meter.avg,
        'time': elapsed_time
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str
) -> Dict[str, float]:
    """
    Evaluate the model on validation set
    
    Args:
        model: Model to evaluate
        data_loader: Validation data loader
        device: Device to evaluate on
        
    Returns:
        Dictionary with validation metrics
    """
    model.train()  # Keep in train mode to get losses
    
    loss_meter = AverageMeter()
    
    for images, targets in tqdm(data_loader, desc="Evaluating"):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        loss_meter.update(losses.item())
    
    return {'val_loss': loss_meter.avg}


def get_optimizer(
    model: torch.nn.Module,
    lr: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 0.0005
) -> torch.optim.Optimizer:
    """
    Get SGD optimizer with standard parameters
    
    Args:
        model: Model to optimize
        lr: Learning rate
        momentum: Momentum factor
        weight_decay: Weight decay
        
    Returns:
        Optimizer
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, 
        lr=lr, 
        momentum=momentum, 
        weight_decay=weight_decay
    )
    return optimizer


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    step_size: int = 3,
    gamma: float = 0.1
) -> torch.optim.lr_scheduler.StepLR:
    """
    Get learning rate scheduler
    
    Args:
        optimizer: Optimizer
        step_size: Period of learning rate decay
        gamma: Multiplicative factor of learning rate decay
        
    Returns:
        Learning rate scheduler
    """
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
    return lr_scheduler
