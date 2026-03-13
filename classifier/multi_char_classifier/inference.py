import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from pathlib import Path
import numpy as np  
import sys
import base64
import io
import cv2

CLASSIFIER_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CLASSIFIER_DIR))
from multi_char_classifier.model import CRNN
from multi_char_classifier.utils import load_config, get_device


class HanziClassifier:
    """Inference class for Hanzi character classification."""
    
    def __init__(self, checkpoint_path):
        """
        Args:
            checkpoint_path (str or Path): Path to model checkpoint
            config_path (str or Path, optional): Path to config.yaml
            device (str, optional): Device to run inference on
        """
        
        # Setup device
        self.device = get_device()
        
        # Load checkpoint
        self.checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        

        # Get model parameters
        model_config = checkpoint.get("model_config")
        if not model_config:
            raise ValueError("Model config is missing in checkpoint!")
        
        self.class_names = checkpoint['class_names']
        
        # Initialize the model
        self.model = CRNN(
            img_channel=model_config['model']['img_channel'],
            num_class=len(self.class_names),
            rnn_hidden=model_config['model']['rnn_hidden']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully. Classes: {self.class_names}")
    

    def predict(self, image_input, return_confidence=False):
        """Predict text for a single image.
        
        Args:
            image_input (str or Path or bytes): Image file path, base64 string, or raw bytes
            return_confidence (bool): Whether to return confidence scores
            
        Returns:
            str or tuple: Predicted sequence, optionally with confidence value
        """
        image_tensor = self._preprocess_image(image_input)
        
        # Run inference
        with torch.no_grad():
            output = self.model(image_tensor)
            pred_text, confidence = ctc_decode(output, self.class_names)
        
        if return_confidence:
            return pred_text, confidence
        else:
            return pred_text
        

    def predict_batch(self, image_inputs, return_confidence=False):
        """Predict classes for multiple images in a single forward pass.
        
        Args:
            image_inputs list(str or Path or bytes): Image file paths, base64 strings, or raw bytes
            return_confidence (bool): Whether to return confidence scores
            
        Returns:
            str or tuple: Predicted sequences, optionally with confidences list
        """
        if not image_inputs:
            return []

        # 1. Preprocess all images into individual tensors
        # Each tensor is shape: [1, 1, 32, W]
        tensors = [self._preprocess_image(img) for img in image_inputs]
        
        # 2. Find the maximum width in the batch
        max_width = max([t.shape[3] for t in tensors])
        
        # 3. Pad all tensors to the max_width (right side padding)
        padded_tensors = []
        for t in tensors:
            pad_width = max_width - t.shape[3]
            # Take left/right border columns and estimate background intensity
            border = torch.cat([t[:, :, :, :1], t[:, :, :, -1:]], dim=3)
            pad_value = float(border.median().item())

            padded_t = F.pad(t, (0, pad_width, 0, 0), value=pad_value)
            padded_tensors.append(padded_t)

            
        # 4. Stack into a single batch tensor: [Batch, 1, 32, max_width]
        batch_tensor = torch.cat(padded_tensors, dim=0)
        
        # 5. Single Forward Pass
        with torch.no_grad():
            # Output shape: [Batch, Time, Classes]
            batch_output = self.model(batch_tensor)
            
        # 6. Decode each sequence in the batch
        predictions = []
        for i in range(batch_output.size(0)):
            pred_text, confidence = ctc_decode(batch_output[i], self.class_names)
            if return_confidence:
                predictions.append((pred_text, confidence))
            else:
                predictions.append(pred_text)
            
        return predictions

            
    def _preprocess_image(self, image_input):
        """
        Load an image (file_path, or bytes) and scale it to 32xW
        """

        # 1. Load image into numpy array (Grayscale)
        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Failed to load image from {image_input}")
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        else:
            raise ValueError("image_input must be file path or bytes")

        # 2. Resize image to 32xW
        orig_height, orig_width = image.shape
        aspect_ratio = orig_width / orig_height
        new_width = int(32 * aspect_ratio)
        
        # Match dataset.py exactly: cv2.resize expects (width, height)
        image = cv2.resize(image, (new_width, 32))


        # 3. Convert to PyTorch Tensor: [1, 1, 32, W]
        # Match dataset.py exactly: from_numpy -> float -> / 255.0
        image_tensor = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        return image_tensor
   

def ctc_decode(log_probs: torch.Tensor, class_names) -> str:
    """
    Decodes the log_probs into the char sequence.

    Args:
        log_probs (Tensor): Tensor containing the log_probs in shape (batch, time, classes) or (time, classes)
        
    Returns:
        tuple: Predicted char sequence and confidence dict
    """
    # Remove batch dim
    if log_probs.dim() == 3:
        log_probs = log_probs.squeeze(0)
    
    # 1. Transform to probabilities [0:1]
    probs = log_probs.exp()

    # 2. Greedy decode highest probability class
    sequence = torch.argmax(probs, dim=-1).tolist()
    sequence_conf = torch.max(probs, dim=-1).values.tolist()
    pruned_sequence = []
    pruned_sequence_conf = []

    #print("".join([self.class_names[idx] for idx in sequence]))
    # 3. Only add chars separated by a blank (-)
    for i, idx in enumerate(sequence):
        char = class_names[idx]
        if char == "-":
            continue
        elif i > 0 and idx == sequence[i-1]:
            continue
        else:
            pruned_sequence.append(char)
            pruned_sequence_conf.append(sequence_conf[i])
        
    if not pruned_sequence_conf:
        final_conf = 0.0
    else:
        final_conf = sum(pruned_sequence_conf) / len(pruned_sequence_conf)
        
    return "".join(pruned_sequence), final_conf

def main():
    import pandas as pd

    """Example usage of the classifier."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    config_path = project_root / "multi_char_classifier" / "config.yaml"
    config = load_config(config_path)
    dataset_dir = project_root / config['paths']['dataset10']
    
    # Path to checkpoint
    checkpoint_dir = config['paths']['save_dir']
    model_name = config['model']['model_name']
    checkpoint_path = project_root / checkpoint_dir / model_name / "best_model_10phrases_val.pth"
    
    # Initialize classifier
    classifier = HanziClassifier(checkpoint_path)
    
    # Example: Predict on a image
    train_labels_file = dataset_dir / config['paths']['val_labels_csv']
    df = pd.read_csv(train_labels_file)
    num_correct = 0
    num_false = 0
    for i in range(len(df)):
        row = df.iloc[i]
        image_file = row['filename']
        text = row['text']

        train_image_filepath = dataset_dir / config['paths']['val_img_dir'] / image_file
        
        pred_class, confidence = classifier.predict(
            train_image_filepath, 
            return_confidence=True
        )
            
        if text == pred_class:
            num_correct += 1
        else:
            num_false += 1
        print(f"True text: {text}, Predicted text: {pred_class} (confidence: {confidence:.2%})")
    
    print("Correct count:", num_correct, "false count:", num_false)
if __name__ == "__main__":
    main()