import torch
from torch import nn


class HanziConv(nn.Module):
    """Simple Convolutional Neural Network for Hanzi character classification.
    
    Architecture:
    - 4 convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool)
    - 2 fully connected layers for classification
    
    Formula to calculate output size: O = floor((W - K + 2P) / S) + 1
    where W=input size, K=kernel size, P=padding, S=stride
    """
    
    def __init__(self, in_size, n_classes, in_channels=1, kernel_size=3):
        """
        Args:
            in_size (int): Input image size (assumes square images)
            n_classes (int): Number of output classes
            in_channels (int): Number of input channels (1 for grayscale)
            kernel_size (int): Size of convolutional kernels
        """
        super().__init__()
        
        self.cnn_layers = nn.Sequential(
            # 1. Layer: 32x32x1 -> 32x32x8 -> 16x16x8
            nn.Conv2d(in_channels=in_channels, out_channels=8, padding=1, kernel_size=kernel_size),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 2. Layer: 16x16x8 -> 16x16x16 -> 8x8x16
            nn.Conv2d(in_channels=8, out_channels=16, padding=1, kernel_size=kernel_size),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 3. Layer: 8x8x16 -> 8x8x32 -> 4x4x32
            nn.Conv2d(in_channels=16, out_channels=32, padding=1, kernel_size=kernel_size),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # 4. Layer: 4x4x32 -> 4x4x64 -> 2x2x64
            nn.Conv2d(in_channels=32, out_channels=64, padding=1, kernel_size=kernel_size),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Calculate linear layer input size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, in_size, in_size)
            flattened_features = self.cnn_layers(dummy_input).numel()
            print(f"Flattened features: {flattened_features}")

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_features, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=n_classes)
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        x = self.cnn_layers(x)
        x = self.classifier(x)
        return x