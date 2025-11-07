import os
import numpy as np
from PIL import Image

def find_max_dimensions(image_dir, label_dir):
    """
    Iterates through all images in the image and label directories to find
    the maximum width and maximum height.

    Args:
        image_dir (str): Path to the directory of original signature images.
        label_dir (str): Path to the directory of label masks.

    Returns:
        tuple: (max_height, max_width)
    """
    max_h, max_w = 0, 0
    
    # Check signature images
    print("Finding max dimensions in image directory...")
    for filename in os.listdir(image_dir):
        try:
            with Image.open(os.path.join(image_dir, filename)) as img:
                w, h = img.size
                if w > max_w: max_w = w
                if h > max_h: max_h = h
        except Exception as e:
            print(f"Warning: Could not read {filename}. Error: {e}")

    # Check label masks
    print("Finding max dimensions in label directory...")
    for filename in os.listdir(label_dir):
        try:
            with Image.open(os.path.join(label_dir, filename)) as img:
                w, h = img.size
                if w > max_w: max_w = w
                if h > max_h: max_h = h
        except Exception as e:
            print(f"Warning: Could not read {filename}. Error: {e}")

    print(f"Max dimensions found: (Height: {max_h}, Width: {max_w})")
    return (max_h, max_w)

find_max_dimensions('Data/Padded_Labels', 'Data/Labels')

