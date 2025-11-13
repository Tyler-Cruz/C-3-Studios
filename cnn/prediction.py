#mandatory compiler flags
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT"] = "1"
os.environ["TF_DETERMINISTIC_OPS"] = "0"

#imports
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd

#processes labels before training
def create_binary_curve_mask(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size)

    #invert and Otsu threshold
    _, binary = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #cleans up image for better accuracy
    binary = cv2.medianBlur((binary * 255).astype(np.uint8), 3)
    binary = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)
    binary = binary / 255.0

    #model compatability
    binary_3ch = np.repeat(binary[..., np.newaxis], 3, axis=-1)
    return binary_3ch

#loading the dataset
def load_signature_dataset(image_dir, label_dir, target_size=(128, 128), test_size=0.1, val_size=0.1):
    image_files = sorted(os.listdir(image_dir))
    X, y, names = [], [], []

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename)

        if not os.path.exists(label_path):
            print(f"Missing label for {filename}, skipping.")
            continue

        #makes the image greyscale while importing it
        img = load_img(image_path, color_mode='grayscale', target_size=target_size)
        img_arr = img_to_array(img) / 255.0
        X.append(img_arr)

        #makes the binary mask a curve mask
        mask_arr = create_binary_curve_mask(label_path, target_size)
        y.append(mask_arr)
        names.append(filename)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"Loaded {len(X)} samples from {image_dir}")

    #splits the data into train, val, and test while retaining where it got them from
    X_train, X_temp, y_train, y_temp, names_train, names_temp = train_test_split(
        X, y, names, test_size=(val_size + test_size), random_state=42
    )
    relative_val_size = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test, names_val, names_test = train_test_split(
        X_temp, y_temp, names_temp, test_size=(1 - relative_val_size), random_state=42
    )

    print(f"Dataset split: {len(X_train)} train | {len(X_val)} val | {len(X_test)} test")
    return X_train, X_val, X_test, y_train, y_val, y_test, names_train, names_val, names_test


#creating the loss functions
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


#creating the cnn proper
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

    outputs = layers.Conv2D(3, (1,1), activation='sigmoid')(c5)

    model = models.Model(inputs, outputs, name="SignatureFeatureExtractor")
    return model


#loading the dataset
image_dir = 'Data/Padded_Raw'
label_dir = 'Data/Curve'

#training dataset
X_train, X_val, X_test, y_train, y_val, y_test, names_train, names_val, names_test = load_signature_dataset(image_dir, label_dir)

#compiling and displaying
model = create_signature_cnn(input_shape=(128, 128, 1))
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=bce_dice_loss,
    metrics=['accuracy']
)
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=8,
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")


#gets the number of groups in the binary mask
def count_curves(binary_mask, threshold=0.5):
    mask = (binary_mask > threshold).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(mask)
    return num_labels - 1  # exclude background

#displays curve count for each siganture
results = []
preds = model.predict(X_test)

for i in range(len(X_test)):
    true_mask = np.max(y_test[i], axis=-1)
    pred_mask = np.max(preds[i], axis=-1)

    true_count = count_curves(true_mask)
    pred_count = count_curves(pred_mask)

    results.append({
        "filename": names_test[i],
        "true_curves": true_count,
        "predicted_curves": pred_count
    })

#save to CSV
df = pd.DataFrame(results)
df.to_csv("curve_counts.csv", index=False)
print("Saved curve counts to curve_counts.csv")

#display first few results
print(df.head())
