#mandatory flags
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

#adding imports 
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import pandas as pd

#preprocess teh data
def create_binary_mask(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, target_size)

    _, binary = cv2.threshold(img, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary = cv2.medianBlur((binary * 255).astype(np.uint8), 3)
    binary = cv2.dilate(binary, np.ones((2, 2), np.uint8), iterations=1)
    binary = binary / 255.0

    return binary[..., np.newaxis]

#loading sig dataset
def load_signature_dataset(image_dir, curve_dir, lift_dir, cross_dir,
                           target_size=(256,256), test_size=0.1, val_size=0.1):

    image_files = sorted(os.listdir(image_dir))
    X, Y, names = [], [], []

    for filename in image_files:
        raw_path = os.path.join(image_dir, filename)
        curve_path = os.path.join(curve_dir, filename)
        lift_path  = os.path.join(lift_dir, filename)
        cross_path = os.path.join(cross_dir, filename)

        if not (os.path.exists(curve_path) and os.path.exists(lift_path) and os.path.exists(cross_path)):
            print(f"Missing label for {filename}, skipping.")
            continue

        img = load_img(raw_path, color_mode='grayscale', target_size=target_size)
        img_arr = img_to_array(img) / 255.0
        X.append(img_arr)

        curve_mask = create_binary_mask(curve_path, target_size)
        lift_mask  = create_binary_mask(lift_path, target_size)
        cross_mask = create_binary_mask(cross_path, target_size)

        three_channel = np.concatenate([curve_mask, lift_mask, cross_mask], axis=-1)
        Y.append(three_channel)
        names.append(filename)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    print(f"Loaded {len(X)} samples.")

    X_train, X_temp, Y_train, Y_temp, names_train, names_temp = \
        train_test_split(X, Y, names, test_size=(test_size + val_size), random_state=42)

    rel_val = val_size / (test_size + val_size)
    X_val, X_test, Y_val, Y_test, names_val, names_test = \
        train_test_split(X_temp, Y_temp, names_temp, test_size=1 - rel_val, random_state=42)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, names_train, names_val, names_test


#Loss function
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / \
           (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

#more loss function
def bce_dice(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

#residual block (more advanced version of cnn to improve accuracy)
def residual_block(x, filters):
    shortcut = x

    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Project shortcut if needed
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same")(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


#creates the model
def create_signature_cnn(input_shape=(256, 256, 1)):
    inputs = layers.Input(shape=input_shape)

    # ---- Encoder ----
    c1 = residual_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = residual_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = residual_block(p2, 256)
    p3 = layers.MaxPooling2D()(c3)

    # ---- Bottleneck ----
    b = residual_block(p3, 256)

    # ---- Decoder ----
    u1 = layers.UpSampling2D()(b)
    u1 = layers.Concatenate()([u1, c3])
    c4 = residual_block(u1, 256)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.Concatenate()([u2, c2])
    c5 = residual_block(u2, 64)

    u3 = layers.UpSampling2D()(c5)
    u3 = layers.Concatenate()([u3, c1])
    c6 = residual_block(u3, 32)

    # ---- Output ----
    outputs = layers.Conv2D(3, 1, activation="sigmoid")(c6)

    return models.Model(inputs, outputs)



#load and train the CNN
image_dir = "Data/Padded_Raw"
curve_dir = "Data/Padded_Curve"
lift_dir  = "Data/Padded_Lift"
cross_dir = "Data/Padded_Cross"

X_train, X_val, X_test, Y_train, Y_val, Y_test, names_train, names_val, names_test = \
    load_signature_dataset(image_dir, curve_dir, lift_dir, cross_dir)

model = create_signature_cnn()
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss=bce_dice,
              metrics=["accuracy"])

model.fit(X_train, Y_train,
          validation_data=(X_val, Y_val),
          epochs=25,
          batch_size=8,
          verbose=1)

#saving model as a keras file
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/model.keras", save_format="keras_v3")
print("Model saved → saved_model/model.keras")