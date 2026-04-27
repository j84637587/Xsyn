import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class Caption(Dataset):
    """Loads preprocessed GLIGEN caption data from a .pt file.

    Each item in the .pt file is a dict with:
        file_path  : str   – image filename relative to args.image_path
        captions   : str   – whole-image text prompt
        annos      : list  – each element is a dict with:
                       bbox    : [x1, y1, x2, y2] in original pixel coords
                       caption : str label for the object
    """

    def __init__(self, args):
        self.data = torch.load(args.gligen_caption_pt, map_location='cpu')
        if isinstance(self.data, dict):
            # some .pt files store a dict with a top-level key
            for key in ('data', 'samples', 'items'):
                if key in self.data:
                    self.data = self.data[key]
                    break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    @staticmethod
    def collate_fn(batch):
        return [res for res in batch]


def transform_image(pil_image, image_size):
    """Resize PIL image to image_size×image_size and normalise to [-1, 1].

    Returns a float tensor of shape (1, 3, image_size, image_size).
    """
    pil_image = pil_image.convert('RGB').resize((image_size, image_size), Image.BICUBIC)
    arr = np.array(pil_image).astype(np.float32) / 127.5 - 1.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)  # (3, H, W) — caller does .unsqueeze(0) for batch dim


def recalculate_box_and_verify_if_valid(x1, y1, w, h, image_size, original_image_size,
                                        min_box_size=0.002):
    """Scale a bounding box from the original image space to image_size×image_size.

    Args:
        x1, y1, w, h          : box in original pixel coords (x1,y1 = top-left)
        image_size             : target square size (e.g. 512)
        original_image_size    : (width, height) of the source PIL image
        min_box_size           : minimum valid fractional area (box_area / image_size²)

    Returns:
        (valid, (new_x1, new_y1, new_x2, new_y2))
        Coordinates are in [0, image_size] pixel space.
    """
    orig_w, orig_h = original_image_size
    x2 = x1 + w
    y2 = y1 + h

    scale_x = image_size / orig_w
    scale_y = image_size / orig_h

    new_x1 = max(0.0, min(float(image_size), x1 * scale_x))
    new_y1 = max(0.0, min(float(image_size), y1 * scale_y))
    new_x2 = max(0.0, min(float(image_size), x2 * scale_x))
    new_y2 = max(0.0, min(float(image_size), y2 * scale_y))

    new_w = new_x2 - new_x1
    new_h = new_y2 - new_y1

    area_frac = (new_w * new_h) / (image_size * image_size)
    valid = area_frac >= min_box_size

    return valid, (new_x1, new_y1, new_x2, new_y2)


def make_a_sentence(categories):
    """Build a simple text prompt from a list of category names."""
    if not categories:
        return 'an x-ray security image'
    joined = ' and '.join(categories)
    return f'an x-ray security image with {joined}'


def specific_category_fti(category, fti_filenames, specific_category_filenames):
    """Populate specific_category_filenames[category] with matching filenames.

    Matches by looking for the category name (case-insensitive) in each filename.
    """
    matched = [f for f in fti_filenames if category.lower() in f.lower()]
    specific_category_filenames[category] = matched
