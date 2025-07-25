# data.py

"""
Defines PyTorch Dataset classes for various ophthalmology datasets.

This module provides a base class, RIADDDataset, with common functionality
for loading and transforming retinal images. Several other dataset-specific
classes inherit from it, overriding image loading or file extensions as needed.
"""

import functools
import os

import cv2
import numpy as np
import pandas as pd
import skimage.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

# --- Pre-defined Normalization Statistics ---
NORM_STATS = {
    "imagenet": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "clip": {
        "mean": [0.48145466, 0.4578275, 0.40821073],
        "std": [0.26862954, 0.26130258, 0.27577711],
    },
}

# --- Pre-defined Transformation Components ---
class SquarePad:
    """Pads a PIL image to make it square."""
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant")

def get_transforms(
    is_testing: bool, input_size: int, normalization: str = "imagenet"
):
    """Returns the appropriate transformation pipeline."""
    if normalization not in NORM_STATS:
        raise ValueError(f"Invalid normalization: {normalization}")

    stats = NORM_STATS[normalization]

    if is_testing:
        return transforms.Compose([
            SquarePad(),
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=stats["mean"], std=stats["std"]),
        ])
    else:
        return transforms.Compose([
            SquarePad(),
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=(0, 10)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=stats["mean"], std=stats["std"]),
            transforms.RandomErasing(),
        ])


class RIADDDataset(Dataset):
    """
    Base dataset for the Retinal Images for Autonomous Detection (RIADD) challenge.
    Also serves as a parent class for other datasets.
    """
    def __init__(
        self,
        csv_path: str,
        img_path: str,
        testing: bool = False,
        input_size: int = 224,
        normalization: str = "imagenet",
        ext: str = ".png",
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path, header=0)
        self.img_path = img_path
        self.ext = ext
        self.preprocess = get_transforms(testing, input_size, normalization)
        self.N_CLASSES = 29  # <-- ADD THIS LINE

    @functools.lru_cache(maxsize=None)
    def _load_image_tensor(self, path: str) -> Image.Image:
        """
        Loads an image and applies dataset-specific cropping.
        This logic appears tailored to the unique dimensions of the RIADD dataset images.
        """
        input_image = sio.imread(path)

        # Apply specific cropping based on image width
        if input_image.shape[1] == 4288:
            pil_image = transforms.ToPILImage()(input_image)
            pil_image = transforms.functional.affine(pil_image, angle=0.0, scale=1, shear=0, translate=[175, 0])
            pil_image = transforms.CenterCrop((3423))(pil_image)
        elif input_image.shape[1] == 2144:
            pil_image = transforms.ToPILImage()(input_image)
            pil_image = transforms.CenterCrop(1424)(pil_image)
        else:
            pil_image = transforms.ToPILImage()(input_image)
            pil_image = transforms.CenterCrop(1536)(pil_image)
        return pil_image

    def __getitem__(self, index: int):
        img_id = str(self.df.iloc[index, 0])
        
        # Construct the full image path, ensuring correct extension
        base_name = os.path.splitext(img_id)[0]
        path = os.path.join(self.img_path, base_name + self.ext)

        input_image = self._load_image_tensor(path)
        input_tensor = self.preprocess(input_image)

        # Extract labels and process them
        label = self.df.iloc[index, 1:].to_list()
        label_tensor = torch.tensor(label)

        # This logic collapses multiple trailing classes into a single 'other' class
        # if the original label has more than 29 columns.
        if len(label_tensor) > 29:
            other_finding_present = label_tensor[28:].sum() > 0
            new_last_label = torch.tensor([1]) if other_finding_present else torch.tensor([0])
            label_tensor = torch.cat((label_tensor[:28], new_last_label), dim=0)

        return input_tensor, label_tensor

    def __len__(self) -> int:
        return len(self.df)


class REFUGEDataset(RIADDDataset):
    """Dataset for REFUGE challenge images."""
    def __init__(self, **kwargs):
        super().__init__(ext=".jpg", **kwargs)

    @functools.lru_cache(maxsize=None)
    def _load_image_tensor(self, path: str) -> Image.Image:
        """Loads a standard JPG image without special cropping."""
        return Image.open(path).convert("RGB")


class EyePACSDataset(RIADDDataset):
    """Dataset for EyePACS, which requires cropping to the circular fundus area."""
    def __init__(self, **kwargs):
        super().__init__(ext=".jpeg", **kwargs)

    def _load_image_tensor(self, path: str) -> Image.Image:
        """
        Loads an image and crops it to the largest contour found,
        effectively isolating the circular fundus from the black background.
        """
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to grayscale and threshold to create a binary mask
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Find all contours in the mask
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return Image.fromarray(img_rgb) # Return original if no contours found

        # Assume the largest contour by area is the fundus
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        # Crop the original image using the bounding box of the largest contour
        crop = img_rgb[y : y + h, x : x + w]

        return Image.fromarray(crop)


class IDRiDDataset(RIADDDataset):
    """Dataset for IDRiD challenge images."""
    def __init__(self, **kwargs):
        super().__init__(ext=".jpg", **kwargs)


class MessidorDataset(RIADDDataset):
    """Dataset for Messidor challenge images."""
    def __init__(self, **kwargs):
        super().__init__(ext=".tif", **kwargs)


class ORIGADataset(EyePACSDataset):
    """Dataset for ORIGA images, uses EyePACS-style cropping."""
    def __init__(self, **kwargs):
        # ORIGA images might not have extensions in the CSV, so ext=''
        super().__init__(ext="", **kwargs)


class HRFDataset(ORIGADataset):
    """Dataset for High-Resolution Fundus (HRF) images."""
    ...


class DeepDRID(IDRiDDataset):
    """Dataset for DeepDRiД challenge."""
    ...
