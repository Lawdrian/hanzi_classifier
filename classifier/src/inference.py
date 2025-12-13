import torch
from PIL import Image
from pathlib import Path
import numpy as np
import sys
import base64
import io

CLASSIFIER_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CLASSIFIER_DIR))
from src.model import HanziConv
from src.dataset import get_image_transforms
from src.utils import load_config, get_device


class HanziClassifier:
    """Inference class for Hanzi character classification."""
    
    def __init__(self, checkpoint_path, config_path=None, device=None):
        """
        Args:
            checkpoint_path (str or Path): Path to model checkpoint
            config_path (str or Path, optional): Path to config.yaml
            device (str, optional): Device to run inference on
        """
        self.checkpoint_path = Path(checkpoint_path)
        
        # Load config
        if config_path is None:
            config_path = self.checkpoint_path.parent.parent.parent / "config.yaml"
        self.config = load_config(config_path)
        
        # Setup device
        self.device = device if device else get_device()
        
        # Get model parameters
        in_size = self.config['training']['in_size']
        in_channels = self.config['training']['in_channels']
        kernel_size = self.config['training']['kernel_size']
        n_classes = len(self.config['classes']['names'])
        
        # Load model
        print(f"Loading model from {checkpoint_path}")
        self.model = HanziConv(
            in_size=in_size,
            in_channels=in_channels,
            n_classes=n_classes,
            kernel_size=kernel_size
        )
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get class names from checkpoint if available (Config is fallback)
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        else:
            self.class_names = self.config['classes']['names']
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get transforms with correct resize
        self.transforms = get_image_transforms(resize=in_size)
        
        print(f"Model loaded successfully. Classes: {self.class_names}")
    
    def predict(self, image_input, return_confidence=False):
        """Predict class for a single image.
        
        Args:
            image_input (str or Path or bytes): Image file path, base64 string, or raw bytes
            return_confidence (bool): Whether to return confidence scores
            
        Returns:
            str or tuple: Predicted class name, optionally with confidence dict
        """
        # Load image
        if isinstance(image_input, (str, Path)):
            image_input = str(image_input)
            # Check if it's a file path
            if image_input.startswith(('\\', '/', 'C:', '.')):  # File path
                image = Image.open(image_input)
            else:  # Assume base64
                image_data = base64.b64decode(image_input)
                image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input))
        else:
            raise ValueError("image_input must be file path, base64 string, or bytes")

        # Preprocess image
        image_tensor = self.transforms(image)
        assert isinstance(image_tensor, torch.Tensor)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_idx = output.argmax(dim=1).item()
            confidence = probabilities[0, pred_idx].item()
        
        pred_class = self.class_names[pred_idx]
        
        if return_confidence:
            confidence_dict = {
                self.class_names[i]: probabilities[0, i].item()
                for i in range(len(self.class_names))
            }
            return pred_class, confidence, confidence_dict
        
        return pred_class
    
    def predict_batch(self, image_paths):
        """Predict classes for multiple images.
        
        Args:
            image_paths (list): List of image paths
            
        Returns:
            list: List of predicted class names
        """
        predictions = []
        for image_path in image_paths:
            pred_class = self.predict(image_path)
            predictions.append(pred_class)
        return predictions


def main():
    """Example usage of the classifier."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.yaml"
    config = load_config(config_path)
    
    # Path to checkpoint
    checkpoint_dir = config['paths']['checkpoint_dir']
    model_name = config['training']['model_name']
    checkpoint_path = project_root / checkpoint_dir / model_name / "best_model.pth"
    
    # Initialize classifier
    classifier = HanziClassifier(checkpoint_path, config_path)
    
    # Example: Predict on a validation image
    val_dataset_path = project_root / config['paths']['processed_data_dir'] / config['paths']['dataset_name'] / "val"
    
    # Get first image from first class
    first_class = config['classes']['names'][1]
    sample_images = list((val_dataset_path / first_class).glob("*.jpg"))
    
    if sample_images:
        sample_image = sample_images[0]
        pred_class, confidence, all_confidences = classifier.predict(
            sample_image, 
            return_confidence=True
        )
        
        print(f"\nPrediction for {sample_image.name}:")
        print(f"Predicted class: {pred_class} (confidence: {confidence:.2%})")
        print(f"\nAll class confidences:")
        for class_name, conf in all_confidences.items():
            print(f"  {class_name}: {conf:.2%}")
    else:
        print("No validation images found. Train the model first.")


if __name__ == "__main__":
    main()