import torch
import torchvision
from torchvision import transforms
from pathlib import Path
from torch.utils.data import DataLoader


def get_image_transforms():
    """Get the image transformation pipeline for training and inference.
    
    Returns:
        transforms.Compose: Composed transformations
    """
    return transforms.Compose([
        # 1. Ensure grayscale (1 channel)
        transforms.Grayscale(num_output_channels=1), 
        
        # 2. Scale to [0.0, 1.0]; Change shape to [C, H, W]
        transforms.ToTensor(), 
        
        # 3. Normalization
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def load_datasets(config, project_root):
    """Load training and validation datasets.
    
    Args:
        config (dict): Configuration dictionary
        project_root (Path): Project root directory
        
    Returns:
        tuple: (dataset_train, dataset_val, dataloader_train, dataloader_val)
    """
    processed_data_dir = config['paths']['processed_data_dir']
    dataset_name = config['paths']['dataset_name']
    batch_size = config['training']['batch_size']
    
    dataset_path = project_root / processed_data_dir / dataset_name
    dataset_train_path = dataset_path / "train"
    dataset_val_path = dataset_path / "val"
    
    image_transforms = get_image_transforms()
    
    dataset_train = torchvision.datasets.ImageFolder(
        root=dataset_train_path, 
        transform=image_transforms
    )
    dataset_val = torchvision.datasets.ImageFolder(
        root=dataset_val_path, 
        transform=image_transforms
    )
    
    dataloader_train = DataLoader(
        dataset_train, 
        shuffle=True, 
        batch_size=batch_size
    )
    dataloader_val = DataLoader(
        dataset_val, 
        shuffle=True, 
        batch_size=batch_size
    )
    
    return dataset_train, dataset_val, dataloader_train, dataloader_val