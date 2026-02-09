# Dice Detection

A comprehensive implementation of dice detection using Faster R-CNN with data augmentation techniques and class imbalance handling strategies.

## Overview

This project implements an object detection system for detecting and classifying dice in images. The system uses Faster R-CNN with either ResNet50 or MobileNetV3 backbone and includes several features:

- Multiple data augmentation strategies (mosaic, copy-paste, random transforms)
- Class-aware and difficulty-aware sampling for handling imbalanced datasets
- Focal loss implementation for improved class imbalance handling
- GAN-based synthetic data generation
- Comprehensive evaluation metrics
- Flexible training and inference pipelines

## Project Structure

```
Dice-Detection/
├── src/                          # Core source code
│   ├── augmentation.py          # Augmentation strategies
│   ├── config.py                # Configuration settings
│   ├── dataset.py               # Dataset loading
│   ├── gan.py                   # GAN implementation
│   ├── Loss_function.py         # Focal loss implementation
│   ├── metrics.py               # Evaluation metrics
│   ├── model.py                 # Model definitions
│   ├── training.py              # Training utilities
│   ├── visualization.py         # Visualization tools
│   └── aug/                     # Advanced augmentation modules
│       ├── annotation_manager.py
│       ├── copy_paste.py
│       └── difficulty_sampling.py
├── notebook/                     # Jupyter notebooks (main entry point)
│   ├── augmentation_comparison.ipynb  # Training pipeline
│   └── GAN.ipynb                      # GAN data generation
├── Annotations/                  # COCO format annotation files
├── doc/                          # Documentation and guides
└── results/                      # Training results and metrics
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (for GPU training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Dice-Detection.git
cd Dice-Detection
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install roboflow matplotlib seaborn tqdm numpy pillow
```

3. Download the dataset using Roboflow or prepare your own COCO-format dataset.

## Usage

This project uses Jupyter notebooks for training and experimentation. The notebooks provide an interactive environment for configuring augmentation strategies, training models, and evaluating results.

### Main Notebooks

1. **`notebook/augmentation_comparison.ipynb`**: Complete training pipeline
   - Load and prepare datasets with custom annotations
   - Configure augmentation strategies (mosaic, copy-paste, random transforms)
   - Enable class-aware or difficulty-aware sampling
   - Train Faster R-CNN models with optional focal loss
   - Evaluate performance with mAP metrics
   - Visualize predictions and training history

2. **`notebook/GAN.ipynb`**: GAN-based data generation
   - Train conditional DCGAN on dice crops
   - Generate synthetic dice images
   - Create augmented training sets

### Workflow

1. Open the augmentation comparison notebook
2. Configure your training settings (augmentation, sampling, loss function)
3. Load your annotation files from the `Annotations/` directory
4. Run training cells to train the model
5. Evaluate and visualize results

## Features

### Data Augmentation

1. **Mosaic Augmentation**: Combines 4 images into a 2x2 grid, increasing diversity of object scales and contexts.

2. **Copy-Paste Augmentation**: Extracts dice instances from source images and intelligently pastes them onto target images with edge blending.

3. **Random Augmentations**: Standard augmentations including horizontal flipping, color jitter, brightness/contrast adjustments.

### Sampling Strategies

1. **Class-Aware Sampling**: Balances training by oversampling underrepresented classes.

2. **Difficulty-Aware Sampling**: Prioritizes harder examples based on:
   - Object size (smaller objects are harder)
   - Edge proximity (objects near edges may be cut off)
   - Density and occlusion (overlapping objects)

### Loss Functions

1. **Standard Cross-Entropy**: Default classification loss in Faster R-CNN.

2. **Focal Loss**: Addresses class imbalance by down-weighting easy examples and focusing on hard negatives.

### Model Architectures

1. **Faster R-CNN with ResNet50-FPN**: Standard architecture with strong performance.

2. **Faster R-CNN with MobileNetV3**: Lighter and faster variant suitable for resource-constrained environments.

## Results

Model performance is evaluated using:
- **mAP (mean Average Precision)**: Primary metric at IoU threshold 0.5
- **Per-class AP**: Individual class performance
- **Training history**: Loss curves over epochs

Results are saved to the `results/` directory in JSON format.


