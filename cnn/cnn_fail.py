import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from sklearn.model_selection import train_test_split

# Dataset loading section
def load_signature_dataset(image_dir, label_dir, target_size=(128, 128), test_size=0.1, val_size=0.1):
    """
    Loads signature images and masks, normalizes them, and splits into train/val/test.
    Each label mask should have 3 channels: (pen_lift, pen_cross, pen_curve).
    """

    image_files = sorted(os.listdir(image_dir))
    X, y = [], []

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename)

        if not os.path.exists(label_path):
            print(f"⚠️ Warning: Missing label for {filename}, skipping.")
            continue

        # Load grayscale image
        img = load_img(image_path, color_mode='grayscale', target_size=target_size)
        img_arr = img_to_array(img) / 255.0
        X.append(img_arr)

        # Load corresponding RGB label
        mask = load_img(label_path, color_mode='rgb', target_size=target_size)
        mask_arr = img_to_array(mask) / 255.0
        y.append(mask_arr)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"✅ Loaded {len(X)} samples from {image_dir}")

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_size + test_size), random_state=42)
    relative_val_size = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - relative_val_size), random_state=42)

    print(f"📊 Dataset split: {len(X_train)} train | {len(X_val)} val | {len(X_test)} test")
    return X_train, X_val, X_test, y_train, y_val, y_test

# loss function for improved accuracy
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

# CNN creation
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

# Loading dataset
image_dir = "Data/images"   
label_dir = "Data/labels"   

X_train, X_val, X_test, y_train, y_val, y_test = load_signature_dataset(image_dir, label_dir)

# Training sect
model = create_signature_cnn(input_shape=(128, 128, 1))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=bce_dice_loss,
              metrics=['accuracy'])
model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=25,
    batch_size=8,
    verbose=1
)

#Evaluation
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"🧪 Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

# Printing graph
preds = model.predict(X_test[:5])

for i in range(5):
    #input
    plt.figure(figsize=(14,4))
    plt.subplot(1,5,1)
    plt.title("Original")
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.axis('off')

    #pen lifts
    plt.subplot(1,5,2)
    plt.title("Pen Lift")
    plt.imshow(preds[i][:,:,0], cmap='hot')
    plt.axis('off')

    #pen cross
    plt.subplot(1,5,3)
    plt.title("Pen Cross")
    plt.imshow(preds[i][:,:,1], cmap='hot')
    plt.axis('off')

    #pen curves
    plt.subplot(1,5,4)
    plt.title("Pen Curve")
    plt.imshow(preds[i][:,:,2], cmap='hot')
    plt.axis('off')

    #AOTA
    plt.subplot(1,5,5)
    plt.title("Combined")
    plt.imshow(np.max(preds[i], axis=-1), cmap='hot')
    plt.axis('off')

    #Printing
    plt.show()

#saving model
# model.save("signature_feature_cnn.h5")
# load model:
# model = tf.keras.models.load_model("signature_feature_cnn.h5", custom_objects={'bce_dice_loss': bce_dice_loss})

