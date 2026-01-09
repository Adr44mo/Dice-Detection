"""
Utility functions for setting up the project on Colab
"""

import os
import sys


def setup_colab_environment(repo_url=None):
    """
    Setup the environment for Google Colab
    
    Args:
        repo_url: Optional GitHub repository URL to clone
    """
    print("="*60)
    print("Setting up Dice Detection Environment")
    print("="*60)
    
    # Install dependencies
    print("\n1. Installing dependencies...")
    os.system("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    os.system("pip install -q roboflow pillow matplotlib seaborn tqdm numpy")
    
    # Clone repository if URL provided
    if repo_url:
        print("\n2. Cloning repository...")
        if os.path.exists("dice-detection"):
            print("   Repository already exists, skipping clone")
        else:
            os.system(f"git clone {repo_url} dice-detection")
            os.chdir("dice-detection")
    else:
        print("\n2. Repository URL not provided (TODO)")
        print("   Make sure to upload src/ folder manually or provide repo URL")
    
    # Add src to path
    if os.path.exists("src"):
        sys.path.insert(0, "./src")
        print("\n3. Added src/ to Python path")
    
    # Check GPU
    import torch
    print("\n4. Checking environment...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "="*60)
    print("Setup complete!")
    print("="*60)


def download_roboflow_dataset(api_key, workspace="workspace-spezm", project="dice-0sexk", version=1):
    """
    Download dataset from Roboflow
    
    Args:
        api_key: Your Roboflow API key
        workspace: Roboflow workspace name
        project: Project name
        version: Dataset version
        
    Returns:
        Dataset object with location information
    """
    from roboflow import Roboflow
    
    print("="*60)
    print("Downloading Dataset from Roboflow")
    print("="*60)
    
    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download("coco")
    
    print(f"\nDataset downloaded to: {dataset.location}")
    print("="*60)
    
    return dataset


def create_directories():
    """Create necessary directories for checkpoints and results"""
    directories = [
        "checkpoints",
        "checkpoints_augmented",
        "results",
        "visualizations"
    ]
    
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
        print(f"Created directory: {dir_name}")


def print_dataset_info(dataset_path):
    """
    Print information about the dataset
    
    Args:
        dataset_path: Path to dataset root
    """
    import json
    from pathlib import Path
    
    print("="*60)
    print("Dataset Information")
    print("="*60)
    
    # Check train and val directories
    train_path = Path(dataset_path) / "train"
    val_path = Path(dataset_path) / "valid"
    
    if train_path.exists():
        ann_file = train_path / "_annotations.coco.json"
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
            
            print(f"\nTraining Set:")
            print(f"  Images: {len(coco_data['images'])}")
            print(f"  Annotations: {len(coco_data['annotations'])}")
            print(f"  Categories: {len(coco_data['categories'])}")
            print(f"  Classes: {[cat['name'] for cat in coco_data['categories']]}")
    
    if val_path.exists():
        ann_file = val_path / "_annotations.coco.json"
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
            
            print(f"\nValidation Set:")
            print(f"  Images: {len(coco_data['images'])}")
            print(f"  Annotations: {len(coco_data['annotations'])}")
    
    print("="*60)


if __name__ == "__main__":
    # Example usage
    print("Dice Detection Setup Utilities")
    print("\nTo use in your notebook:")
    print("  from src.utils import setup_colab_environment, download_roboflow_dataset")
    print("  setup_colab_environment(repo_url='YOUR_GITHUB_URL')")
    print("  dataset = download_roboflow_dataset(api_key='YOUR_API_KEY')")
