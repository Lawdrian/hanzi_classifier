# train.py
import torch
from torch import nn
from torch.optim.adamw import AdamW
import numpy as np
import os
from pathlib import Path
import sys

CLASSIFIER_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CLASSIFIER_DIR))
from src.model import HanziConv
from src.dataset import load_datasets
from src.utils import load_config, get_device, save_checkpoint, load_train_checkpoint


class ModelTrainer:
    """Trainer class for the Hanzi classifier."""
    
    def __init__(self, config, project_root, device):
        """
        Args:
            config (dict): Configuration dictionary
            project_root (Path): Project root directory
            device (str): Device to train on ('cuda' or 'cpu')
        """
        self.config = config
        self.project_root = project_root
        self.device = device
        
        # Model parameters
        in_size = config['training']['in_size']
        in_channels = config['training']['in_channels']
        kernel_size = config['training']['kernel_size']
        n_classes = len(config['classes']['names'])
        
        # Training parameters
        self.learning_rate = config['training']['learning_rate']
        self.max_epochs = config['training']['max_epochs']
        self.eval_interval = config['training']['eval_interval']
        
        # Initialize model, loss, and optimizer
        print(f"Storing model in: {device}")
        self.model = HanziConv(
            in_size=in_size,
            in_channels=in_channels,
            n_classes=n_classes,
            kernel_size=kernel_size
        ).to(device)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        
        # Load datasets
        self.dataset_train, self.dataset_val, self.dataloader_train, self.dataloader_val = \
            load_datasets(config, project_root)
    
    def train(self, resume_from_checkpoint=None):
        """Train the model.
        
        Args:
            resume_from_checkpoint (str, optional): Path to checkpoint to resume from
        """
        start_epoch = 1
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            start_epoch = load_train_checkpoint(
                self.model, 
                self.optimizer, 
                resume_from_checkpoint, 
                self.device
            ) + 1
        
        best_val_loss = np.inf
        
        for epoch in range(start_epoch, self.max_epochs + 1):
            # Training phase
            self.model.train()
            for input, y_true in self.dataloader_train:
                input = input.to(self.device) 
                y_true = y_true.to(self.device)
                
                # Perform training step
                y_pred = self.model(input)
                loss = self.loss_fn(y_pred, y_true)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Evaluation phase
            if epoch % self.eval_interval == 0:
                train_loss = self._evaluate(self.dataloader_train)
                val_loss = self._evaluate(self.dataloader_val)
                
                print(f"epoch {epoch}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

                # Save checkpoint if validation loss improved
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_dir_path = self.project_root / self.config['paths']['checkpoint_dir'] / self.config['training']['model_name']
                    os.makedirs(checkpoint_dir_path, exist_ok=True)
                    checkpoint_file_path = checkpoint_dir_path / "best_model.pth"
                    save_checkpoint(self.model, self.optimizer, epoch, val_loss, checkpoint_file_path, class_names=self.dataset_train.classes)
        
        print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")
    
    def _evaluate(self, dataloader):
        """Evaluate model on a dataset.
        
        Args:
            dataloader: PyTorch DataLoader
            
        Returns:
            float: Average loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for input, y_true in dataloader:
                input = input.to(self.device) 
                y_true = y_true.to(self.device)
                y_pred = self.model(input)
                total_loss += self.loss_fn(y_pred, y_true)
        
        return total_loss / len(dataloader)


def main():
    """Main training function."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    print(project_root)
    config_path = project_root / "config.yaml"
    # Load configuration
    config = load_config(config_path)
    print(config)
    
    # Get device
    device = get_device()
    
    # Initialize trainer
    trainer = ModelTrainer(config, project_root, device)
    
    # Train model
    trainer.train()


if __name__ == "__main__":
    main()