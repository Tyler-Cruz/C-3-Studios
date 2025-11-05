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

def pad_image(img_arr, target_size):
    """
    Pads a NumPy image array (H, W, C) to a target size (target_h, target_w)
    by adding black padding to center it.

    Args:
        img_arr (np.array): The image as a NumPy array.
        target_size (tuple): (target_h, target_w)

    Returns:
        np.array: The padded image array.
    """
    target_h, target_w = target_size
    h, w, c = img_arr.shape

    # Calculate total padding needed
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    # Calculate padding for each side (to center the image)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Define the padding structure for np.pad
    # We pad height (axis 0), width (axis 1), but not channels (axis 2)
    padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

    # Apply padding with a constant value of 0 (black)
    padded_arr = np.pad(img_arr, padding, mode='constant', constant_values=0)
    
    return padded_arr