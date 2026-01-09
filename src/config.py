"""
Configuration file for dice detection project
Centralized hyperparameters and settings
"""

# Training Configuration
BATCH_SIZE = 4
NUM_WORKERS = 2
NUM_EPOCHS = 10
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

# Learning Rate Scheduler
LR_STEP_SIZE = 3
LR_GAMMA = 0.1

# Model Configuration
PRETRAINED = True
TRAINABLE_BACKBONE_LAYERS = 3
MIN_IMAGE_SIZE = 800
MAX_IMAGE_SIZE = 1333

# Augmentation Configuration
USE_CLASS_AWARE_SAMPLING = True
USE_MOSAIC_AUGMENTATION = True
MOSAIC_PROB = 0.5
MOSAIC_OUTPUT_SIZE = (640, 640)

# Data Augmentation Probabilities
HORIZONTAL_FLIP_PROB = 0.5
COLOR_JITTER_PROB = 0.5
BRIGHTNESS_RANGE = (0.8, 1.2)
CONTRAST_RANGE = (0.8, 1.2)
SATURATION_RANGE = (0.8, 1.2)

# Evaluation Configuration
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.05

# Paths
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_DIR_AUGMENTED = "checkpoints_augmented"
RESULTS_DIR = "results"
VISUALIZATION_DIR = "visualizations"

# Dataset
ANNOTATION_FILE = "_annotations.coco.json"

# Roboflow Dataset
ROBOFLOW_WORKSPACE = "workspace-spezm"
ROBOFLOW_PROJECT = "dice-0sexk"
ROBOFLOW_VERSION = 1

# Training Settings
PRINT_FREQ = 10  # Print training stats every N batches
SAVE_FREQ = 1    # Save checkpoint every N epochs

# Device
DEVICE = "cuda"  # Will be overridden to "cpu" if CUDA not available

# Random Seeds (for reproducibility)
RANDOM_SEED = 42


def get_config():
    """Return configuration as dictionary"""
    import torch
    
    config = {
        # Training
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'momentum': MOMENTUM,
        
        # Scheduler
        'lr_step_size': LR_STEP_SIZE,
        'lr_gamma': LR_GAMMA,
        
        # Model
        'pretrained': PRETRAINED,
        'trainable_backbone_layers': TRAINABLE_BACKBONE_LAYERS,
        'min_image_size': MIN_IMAGE_SIZE,
        'max_image_size': MAX_IMAGE_SIZE,
        
        # Augmentation
        'use_class_aware_sampling': USE_CLASS_AWARE_SAMPLING,
        'use_mosaic_augmentation': USE_MOSAIC_AUGMENTATION,
        'mosaic_prob': MOSAIC_PROB,
        'mosaic_output_size': MOSAIC_OUTPUT_SIZE,
        
        # Evaluation
        'iou_threshold': IOU_THRESHOLD,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        
        # Device
        'device': DEVICE if torch.cuda.is_available() else 'cpu',
        
        # Seed
        'random_seed': RANDOM_SEED
    }
    
    return config


def print_config():
    """Print current configuration"""
    config = get_config()
    
    print("="*60)
    print("Configuration")
    print("="*60)
    
    for section in ['Training', 'Model', 'Augmentation', 'Evaluation']:
        print(f"\n{section}:")
        # Print relevant configs for each section
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print_config()
