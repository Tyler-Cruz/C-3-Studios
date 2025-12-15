import cv2
import numpy as np
import os

# --- CONFIGURATION ---
# Input: Folder containing the original images with red marks
INPUT_DIR = "Data/Padded_Lift" 
# Output: Folder where the new Black/White masks will be saved
OUTPUT_DIR = "Data/Padded_Lifts_Masks"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_mask_from_specific_red(filename):
    # Load image
    img = cv2.imread(filename)
    if img is None:
        return None

    # Convert to HSV (Hue, Saturation, Value)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # --- Define Lower/Upper Bounds ---
    # We add a wide buffer because JPEG compression creates "noise" around the edges.
    
    # Lower Red Range (0-15)
    # Hue: 0-15
    # Saturation: 140-255
    # Value: 100-255
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([15, 255, 255])
    
    # Upper Red Range (165-180) - Red wraps around the circle
    lower_red2 = np.array([165, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    final_mask = mask1 + mask2
    
    # --- CLEANUP STEPS ---
    # 1. Morphological Open: Removes tiny single-pixel noise (sparkles)
    kernel_clean = np.ones((2,2), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_clean)

    # 2. Dilate: Makes the remaining red dots slightly fatter/solid
    # This ensures the AI has a clear "blob" to learn from.
    kernel_fatten = np.ones((3,3), np.uint8)
    final_mask = cv2.dilate(final_mask, kernel_fatten, iterations=2)
    
    return final_mask

print(f"Processing images from {INPUT_DIR}...")

count = 0
for file in os.listdir(INPUT_DIR):
    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
        path = os.path.join(INPUT_DIR, file)
        
        mask = create_mask_from_specific_red(path)
        
        if mask is not None:
            # Save the new mask
            save_path = os.path.join(OUTPUT_DIR, file)
            cv2.imwrite(save_path, mask)
            count += 1
            if count % 5 == 0:
                print(f"Generated {count} masks...")

print(f"Done! {count} masks saved to {OUTPUT_DIR}")