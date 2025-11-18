# mandatory flags
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "0"

# imports
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import cv2
from scipy.ndimage import label

#dataset loading (only set to two channels until lift data is in)
def load_signature_dataset(image_dir, cross_dir, curve_dir, target_size=(128, 128), test_size=0.1, val_size=0.1):
    """
    Loads images and combines two separate mask files (Cross, Curve) 
    into a single 2-channel label.
    Channel 0 (Red)   = Crosses
    Channel 1 (Green) = Curves
    """
    image_files = sorted(os.listdir(image_dir))
    X, y = [], []

    print(f"Loading data from:\n Images: {image_dir}\n Crosses: {cross_dir}\n Curves: {curve_dir}")

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        
        # Paths for the 2 different masks
        cross_mask_path = os.path.join(cross_dir, filename)
        curve_mask_path = os.path.join(curve_dir, filename)

        # Skip if any mask is missing
        if not (os.path.exists(cross_mask_path) and os.path.exists(curve_mask_path)):
            print(f"Missing one or more labels for {filename}, skipping.")
            continue

        # 1. Load Input Image (Grayscale)
        img = load_img(image_path, color_mode='grayscale', target_size=target_size)
        img_arr = img_to_array(img) / 255.0
        X.append(img_arr)

        # 2. Load Each Label Mask (Grayscale)
        # We load them individually and then stack them.
        cross_mask = img_to_array(load_img(cross_mask_path, color_mode='grayscale', target_size=target_size))
        curve_mask = img_to_array(load_img(curve_mask_path, color_mode='grayscale', target_size=target_size))

        # Normalize to 0-1
        cross_mask /= 255.0
        curve_mask /= 255.0

        # Threshold to binary (0 or 1)
        cross_mask = (cross_mask > 0.5).astype(np.float32)
        curve_mask = (curve_mask > 0.5).astype(np.float32)

        # 3. Stack into a single (128, 128, 2) tensor
        # axis=-1 stacks along the depth (channels)
        combined_mask = np.concatenate([cross_mask, curve_mask], axis=-1)
        y.append(combined_mask)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"Successfully loaded {len(X)} samples.")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_size + test_size), random_state=42)
    relative_val_size = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - relative_val_size), random_state=42)

    print(f"Dataset split: {len(X_train)} train | {len(X_val)} val | {len(X_test)} test")
    return X_train, X_val, X_test, y_train, y_val, y_test


#loss functions
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


#structures the cnn 
def create_signature_cnn(input_shape=(128, 128, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)

    c3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)

    # Bottleneck
    b = layers.Conv2D(256, (3,3), activation='relu', padding='same')(c3)
    b = layers.BatchNormalization()(b)

    # Decoder
    u1 = layers.UpSampling2D((2,2))(b)
    u1 = layers.Concatenate()([u1, c2])
    c4 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(u1)
    c4 = layers.BatchNormalization()(c4)

    u2 = layers.UpSampling2D((2,2))(c4)
    u2 = layers.Concatenate()([u2, c1])
    c5 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(u2)
    c5 = layers.BatchNormalization()(c5)

    # OUTPUT: 2 Channels (Cross, Curve)
    outputs = layers.Conv2D(2, (1,1), activation='sigmoid')(c5)

    model = models.Model(inputs, outputs, name="SignatureFeatureExtractor")
    return model


# --- MAIN EXECUTION ---

# Define Directories
image_dir = 'Data/Padded_Raw'
cross_dir = 'Data/Padded_Labels' # Corresponds to Pen Crosses
curve_dir = 'Data/Padded_Curve'  # Corresponds to Pen Curves

# Load Data
try:
    X_train, X_val, X_test, y_train, y_val, y_test = load_signature_dataset(image_dir, cross_dir, curve_dir)
except Exception as e:
    print(f"Error loading data: {e}")
    print(f"Please ensure '{image_dir}', '{cross_dir}', and '{curve_dir}' directories exist and contain the mask images.")
    exit()

# Compile Model
model = create_signature_cnn(input_shape=(128, 128, 1))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=bce_dice_loss,
    metrics=['accuracy']
)
model.summary()

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25, 
    batch_size=8,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# --- PREDICTION & COUNTING ---
preds = model.predict(X_test)

# Helper function to count blobs (connected components)
def count_features(mask, threshold=0.5, min_size=5):
    mask = (mask * 255).astype(np.uint8)
    _, binary = cv2.threshold(mask, int(threshold * 255), 255, cv2.THRESH_BINARY)

    # Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Connected component analysis
    labeled_array, num_features = label(binary > 0)
    
    # Filter small noise
    sizes = np.bincount(labeled_array.ravel())
    if len(sizes) > 1:
        sizes[0] = 0  # ignore background
        num_features = np.sum(sizes > min_size)
    else:
        num_features = 0
        
    return num_features, binary

# Storage for metrics
metrics = {
    "Cross": {"true": [], "pred": []},
    "Curve": {"true": [], "pred": []}
}

# Channel mapping: 0 -> Cross, 1 -> Curve
channel_names = {0: "Cross", 1: "Curve"} 

# Iterate through test set
for i in range(len(X_test)):
    # Loop through the 2 channels
    for channel_idx in range(2): # Now only 2 channels
        c_name = channel_names[channel_idx]
        
        true_mask = y_test[i][:, :, channel_idx]
        pred_mask = preds[i][:, :, channel_idx]

        # Count
        true_count, _ = count_features(true_mask, threshold=0.5)
        pred_count, _ = count_features(pred_mask, threshold=0.5)

        metrics[c_name]["true"].append(true_count)
        metrics[c_name]["pred"].append(pred_count)

    # Visualization for the first few images
    if i < 3:
        plt.figure(figsize=(9, 4)) # Adjusted figsize for 3 plots
        
        # Original Image
        plt.subplot(1, 3, 1) # Adjusted subplot for 3 plots
        plt.imshow(X_test[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # Show predicted masks for each type
        for idx in range(2): # Now only 2 channels for visualization
            plt.subplot(1, 3, idx + 2) # Adjusted subplot position
            # Use Red for Cross, Green for Curve (arbitrary choice for now)
            cmap = ['Reds', 'Greens'][idx] 
            plt.imshow(preds[i][:, :, idx], cmap=cmap, alpha=0.8)
            plt.title(f"{channel_names[idx]} (Pred)")
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()

# --- Summary ---
print("\n--- Final Analysis Report ---")
for key in ["Cross", "Curve"]: # Only these two keys now
    t = np.array(metrics[key]["true"])
    p = np.array(metrics[key]["pred"])
    
    mae = np.mean(np.abs(t - p))
    corr = np.corrcoef(t, p)[0, 1] if np.std(t) > 0 and np.std(p) > 0 else 0
    
    print(f"\n{key.upper()}:")
    print(f"  MAE: {mae:.2f}")
    print(f"  Correlation: {corr:.2f}")
    print(f"  Avg True: {np.mean(t):.2f} | Avg Pred: {np.mean(p):.2f}")

# Save the unified model
model.save("signature_cross_curve_model.h5") # Changed model name
print("\nModel saved as 'signature_cross_curve_model.h5'")

#hopefully will work to save the model
# model.save("signature_feature_cnn.h5")
# To reload:
# model = tf.keras.models.load_model("signature_feature_cnn.h5", custom_objects={'bce_dice_loss': bce_dice_loss})