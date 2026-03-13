import cv2
import torch
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
import torch.nn.functional as F

class OCRDataset(Dataset):
    def __init__(self, csv_path: Path, img_dir: Path, vocab_dict: dict, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.vocab_dict = vocab_dict
        self.transform = transform

        # Validate that all chars in a phrase are contained in the dictionary
        valid_rows = []
        dropped_count = 0
        for _, row in self.df.iterrows():
            text = str(row['text'])
            if all(char in self.vocab_dict for char in text):
                valid_rows.append(row)
            else:
                dropped_count += 1
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        
        print(f"Dataset Loaded: Kept {len(self.df)} images. Dropped {dropped_count} images with unknown characters.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1. Get filename and text (using column names is safer than unpacking)
        row = self.df.iloc[idx]
        filename = row['filename']
        text = row['text']

        # 2. Open image in Grayscale
        img_path = str(self.img_dir / filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # 3. Force Height=32 and scale Width proportionally
        orig_height, orig_width = image.shape
        aspect_ratio = orig_width / orig_height
        new_width = int(32 * aspect_ratio)
        
        # cv2.resize expects (width, height)
        image = cv2.resize(image, (new_width, 32))
        
        # PyTorch expects a channel dimension: (1, 32, new_width)
        image = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        
        # Apply standard transforms (like normalization) if provided
        if self.transform:
            image = self.transform(image)

        # 4. Convert text to integers
        # Using .get() with a default of 0 (assuming 0 is your blank/unknown token)
        text_ids = [self.vocab_dict.get(char, 0) for char in text]
        text_tensor = torch.LongTensor(text_ids)

        return image, text_tensor
    


def pad_collate(batch):
    """
    Expects batch to be a list of tuples: [(image1, text1), (image2, text2), ...]
    """
    images, targets = zip(*batch)

    # 1. Pad the images
    max_width = max([img.shape[2] for img in images])

    padded_images = []
    for img in images:
        pad_width = max_width - img.shape[2]

        # Take leftmost pixel column and rightmost pixel column and user median value
        border = torch.cat([img[:, :, :1], img[:, :, -1:]], dim=2)
        pad_value = float(border.median().item())

        padded_img = F.pad(img, (0, pad_width, 0, 0), value=pad_value)
        padded_images.append(padded_img)

    images_tensor = torch.stack(padded_images)

    # Track original image widths before padding (needed for CTC input_lengths)
    original_widths = torch.IntTensor([img.shape[2] for img in images])

    # 2. Prepare targets for CTC loss
    # CTC loss mathematically requires a 1D tensor of all targets smashed together,
    # AND a separate tensor telling it how long each target was before smashing.
    target_lengths = torch.IntTensor([len(t) for t in targets])
    targets_tensor = torch.cat(targets) 
    
    return images_tensor, targets_tensor, target_lengths, original_widths
