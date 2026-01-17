"""
GAN (Generative Adversarial Network) module for synthetic dice image generation.

This module contains the conditional DCGAN architecture and utilities for:
- Training GANs on dice crops
- Generating synthetic dice images
- Creating full COCO-formatted scenes with bounding box annotations
"""

import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn


class Generator(nn.Module):
    """Conditional DCGAN Generator with label embedding."""
    
    def __init__(self, latent_dim, num_classes, embed_dim, ngf, nc):
        """
        Initialize the Generator.
        
        Args:
            latent_dim: Dimension of latent noise vector
            num_classes: Number of dice classes (typically 6 for dice 1-6)
            embed_dim: Dimension of label embedding
            ngf: Number of generator filters in first conv layer
            nc: Number of channels in output image (3 for RGB)
        """
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        input_dim = latent_dim + embed_dim
        
        self.main = nn.Sequential(
            # Input is Z (latent + embedding), going into a convolution
            nn.ConvTranspose2d(input_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State size: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # State size: (nc) x 64 x 64
        )
    
    def forward(self, noise, labels):
        """
        Forward pass through generator.
        
        Args:
            noise: Random noise tensor [batch_size, latent_dim]
            labels: Class labels tensor [batch_size]
            
        Returns:
            Generated images tensor [batch_size, nc, 64, 64]
        """
        label_embed = self.label_embedding(labels)
        x = torch.cat([noise, label_embed], dim=1)
        x = x.view(x.size(0), -1, 1, 1)
        return self.main(x)


class Discriminator(nn.Module):
    """Conditional DCGAN Discriminator with label embedding."""
    
    def __init__(self, num_classes, ndf, nc, img_size):
        """
        Initialize the Discriminator.
        
        Args:
            num_classes: Number of dice classes
            ndf: Number of discriminator filters in first conv layer
            nc: Number of channels in input image (3 for RGB)
            img_size: Size of input images (e.g., 64)
        """
        super(Discriminator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, img_size * img_size)
        self.img_size = img_size
        
        self.main = nn.Sequential(
            # Input is (nc+1) x 64 x 64
            nn.Conv2d(nc + 1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        """
        Forward pass through discriminator.
        
        Args:
            img: Input images tensor [batch_size, nc, img_size, img_size]
            labels: Class labels tensor [batch_size]
            
        Returns:
            Probability that images are real [batch_size]
        """
        label_embed = self.label_embedding(labels)
        label_channel = label_embed.view(labels.size(0), 1, self.img_size, self.img_size)
        x = torch.cat([img, label_channel], dim=1)
        return self.main(x).view(-1, 1).squeeze(1)


def weights_init(m):
    """
    Custom weights initialization for Conv and BatchNorm layers.
    
    Args:
        m: Module to initialize
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def generate_dice_image(generator, class_idx, size, device, latent_dim=100):
    """
    Generate a single dice image of given class and resize.
    
    Args:
        generator: Trained Generator model
        class_idx: Class index (0-5 for dice 1-6)
        size: Target size in pixels (will be square)
        device: Torch device (cuda or cpu)
        latent_dim: Dimension of latent noise vector
        
    Returns:
        PIL Image of generated dice
    """
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(1, latent_dim, device=device)
        label = torch.tensor([class_idx], device=device)
        fake_img = generator(noise, label)[0].cpu().numpy().transpose(1, 2, 0)
        # Convert from [-1, 1] to [0, 255]
        fake_img = ((fake_img + 1) / 2 * 255).astype(np.uint8)
        fake_img = np.clip(fake_img, 0, 255)
        pil_img = Image.fromarray(fake_img)
        return pil_img.resize((size, size), Image.LANCZOS)


def check_overlap(new_box, existing_boxes, min_distance=10):
    """
    Check if new bounding box overlaps with existing boxes.
    
    Args:
        new_box: New box as [x_min, y_min, x_max, y_max]
        existing_boxes: List of existing boxes in same format
        min_distance: Minimum distance between boxes
        
    Returns:
        True if overlap detected, False otherwise
    """
    for box in existing_boxes:
        if (new_box[0] < box[2] + min_distance and new_box[2] > box[0] - min_distance and
            new_box[1] < box[3] + min_distance and new_box[3] > box[1] - min_distance):
            return True
    return False


def extract_backgrounds(image_dir, output_dir, num_backgrounds=50):
    """
    Extract background images from dataset for scene generation.
    
    Args:
        image_dir: Directory containing source images
        output_dir: Directory to save background images
        num_backgrounds: Number of backgrounds to extract
        
    Returns:
        Number of backgrounds extracted
    """
    os.makedirs(output_dir, exist_ok=True)
    
    bg_count = 0
    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(image_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img.save(os.path.join(output_dir, f'bg_{bg_count:04d}.jpg'))
            bg_count += 1
            if bg_count >= num_backgrounds:
                break
    
    return bg_count


def create_synthetic_coco_dataset(
    generator,
    background_dir,
    output_dir,
    config,
    device,
    class_counts=None,
    latent_dim=100
):
    """
    Generate a complete COCO-format dataset with synthetic dice scenes.
    
    Args:
        generator: Trained Generator model
        background_dir: Directory with background images
        output_dir: Directory to save synthetic dataset
        config: Configuration dict with keys:
            - scene_size: Tuple (width, height) for output scenes
            - dice_size_range: Tuple (min_size, max_size) for dice
            - dice_per_image: Tuple (min_dice, max_dice) per scene
            - num_images: Number of scenes to generate
        device: Torch device
        class_counts: Dict of current class counts to balance (optional)
        latent_dim: Dimension of latent noise vector
        
    Returns:
        Dictionary with 'images', 'annotations', and 'categories' generated
    """
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    
    # Load backgrounds
    background_files = [f for f in os.listdir(background_dir) if f.endswith(('.jpg', '.png'))]
    if not background_files:
        raise ValueError(f"No background images found in {background_dir}")
    
    # COCO format structures
    coco_images = []
    coco_annotations = []
    coco_categories = [{'id': i, 'name': str(i)} for i in range(1, 7)]
    
    # Calculate target counts if class_counts provided
    if class_counts:
        target_count = max(class_counts.values())
        images_to_generate = {k: max(0, target_count - v) for k, v in class_counts.items()}
    else:
        images_to_generate = {str(i): 0 for i in range(1, 7)}
    
    image_id = 1
    annotation_id = 1
    generated_per_class = {str(i): 0 for i in range(1, 7)}
    
    scene_size = config.get('scene_size', (640, 640))
    dice_size_range = config.get('dice_size_range', (60, 120))
    dice_per_image = config.get('dice_per_image', (1, 4))
    num_images = config.get('num_images', 100)
    
    print(f"Generating {num_images} synthetic scenes...")
    
    for scene_idx in tqdm(range(num_images)):
        # Load random background and resize
        bg_file = random.choice(background_files)
        background = Image.open(os.path.join(background_dir, bg_file)).convert('RGB')
        background = background.resize(scene_size, Image.LANCZOS)
        scene = background.copy()
        
        # Determine number of dice
        num_dice = random.randint(*dice_per_image)
        
        # Prioritize underrepresented classes
        if class_counts:
            needed_classes = [k for k, v in images_to_generate.items() if generated_per_class[k] < v]
            if not needed_classes:
                needed_classes = [str(i) for i in range(1, 7)]
        else:
            needed_classes = [str(i) for i in range(1, 7)]
        
        placed_boxes = []
        scene_annotations = []
        
        for _ in range(num_dice):
            # Select class
            class_name = random.choice(needed_classes)
            class_idx = int(class_name) - 1
            
            # Random dice size
            dice_size = random.randint(*dice_size_range)
            
            # Try to place dice without overlap
            max_attempts = 20
            for attempt in range(max_attempts):
                x = random.randint(0, scene_size[0] - dice_size)
                y = random.randint(0, scene_size[1] - dice_size)
                new_box = [x, y, x + dice_size, y + dice_size]
                
                if not check_overlap(new_box, placed_boxes):
                    # Generate and paste dice
                    dice_img = generate_dice_image(generator, class_idx, dice_size, device, latent_dim)
                    scene.paste(dice_img, (x, y))
                    
                    placed_boxes.append(new_box)
                    scene_annotations.append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': int(class_name),
                        'bbox': [x, y, dice_size, dice_size],  # COCO format: x, y, w, h
                        'area': dice_size * dice_size,
                        'iscrowd': 0
                    })
                    annotation_id += 1
                    generated_per_class[class_name] += 1
                    break
        
        if scene_annotations:
            # Save image
            img_filename = f"synthetic_{image_id:05d}.jpg"
            scene.save(os.path.join(output_dir, 'train', img_filename))
            
            coco_images.append({
                'id': image_id,
                'file_name': img_filename,
                'width': scene_size[0],
                'height': scene_size[1]
            })
            coco_annotations.extend(scene_annotations)
            image_id += 1
    
    # Save COCO annotations
    coco_data = {
        'images': coco_images,
        'annotations': coco_annotations,
        'categories': coco_categories
    }
    
    with open(os.path.join(output_dir, 'train', '_annotations.coco.json'), 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\n✅ Generated {len(coco_images)} images with {len(coco_annotations)} annotations")
    print(f"\nPer-class generation counts:")
    for k, v in generated_per_class.items():
        print(f"  Class {k}: {v} dice")
    
    return coco_data
