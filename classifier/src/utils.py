import yaml
import torch
from pathlib import Path


def load_config(config_path):
    """Load configuration from YAML file.
    
    Args:
        config_path (Path or str): Path to config.yaml
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_device():
    """Get the available device (CUDA or CPU).
    
    Returns:
        str: Device name ('cuda' or 'cpu')
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    return device


def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, class_names):
    """Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch (int): Current epoch
        val_loss (float): Validation loss
        checkpoint_path (Path or str): Path to save checkpoint
        class_names (list[str]): List of class_names in correct order
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'class_names': class_names
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_train_checkpoint(model, optimizer, checkpoint_path, device):
    """Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        checkpoint_path (Path or str): Path to checkpoint file
        device (str): Device to load model to
        
    Returns:
        int: Epoch number from checkpoint
    """
    try:
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        
        print(f"Loaded checkpoint from epoch {epoch}")
        return epoch
        
    except FileNotFoundError:
        print("No checkpoint found.")
        return 0
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0
    

def load_model_checkpoint(model, checkpoint_path, device):
    """Load only model weights for inference.
    
    Args:
        model: PyTorch model
        checkpoint_path (Path or str): Path to checkpoint file
        device (str): Device to load model to
        
    Returns:
        dict: Checkpoint dictionary (in case you need metadata like class_names)
    """
    try:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully")
        return checkpoint
        
    except FileNotFoundError:
        print(f"Checkpoint not found: {checkpoint_path}")
        raise
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise