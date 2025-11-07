#mandatory flags
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO and WARNING logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # Enable oneDNN optimizations for CPU
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Prevent TensorFlow from grabbing all GPU memory
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"  # Enable XLA compilation for performance
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"  # Optimize GPU threading
os.environ["TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT"] = "1"  # Faster batchnorm
os.environ["TF_DETERMINISTIC_OPS"] = "0"  # Allow non-deterministic fast ops

#imports
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from sklearn.model_selection import train_test_split

#dataset loading
def load_signature_dataset(image_dir, label_dir, target_size=(128, 128), test_size=0.1, val_size=0.1):

    #will probably have all 3 in one, but for now only takes in pen curves
    image_files = sorted(os.listdir(image_dir))
    X, y = [], []

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        label_path = os.path.join(label_dir, filename)

        if not os.path.exists(label_path):
            print(f"⚠️ Warning: Missing label for {filename}, skipping.")
            continue

        img = load_img(image_path, color_mode='grayscale', target_size=target_size)
        img_arr = img_to_array(img) / 255.0
        X.append(img_arr)

        mask = load_img(label_path, color_mode='rgb', target_size=target_size)
        mask_arr = img_to_array(mask) / 255.0
        y.append(mask_arr)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"✅ Loaded {len(X)} samples from {image_dir}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_size + test_size), random_state=42)
    relative_val_size = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=(1 - relative_val_size), random_state=42)

    print(f"📊 Dataset split: {len(X_train)} train | {len(X_val)} val | {len(X_test)} test")
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

    outputs = layers.Conv2D(3, (1,1), activation='sigmoid')(c5)

    model = models.Model(inputs, outputs, name="SignatureFeatureExtractor")
    return model

#load data
image_dir = 'Data/Padded_Raw'
label_dir = 'Data/Curve'
X_train, X_val, X_test, y_train, y_val, y_test = load_signature_dataset(image_dir, label_dir)

#compile and train
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

#evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
print(f"🧪 Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

#displaying results
preds = model.predict(X_test[:5])

for i in range(5):
    plt.figure(figsize=(14, 4))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    plt.imshow(X_test[i].squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Predicted Pen Curves")
    plt.imshow(preds[i][:, :, 2], cmap='hot')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Combined Features")
    plt.imshow(np.max(preds[i], axis=-1), cmap='hot')
    plt.axis('off')

    plt.show()

#hopefully will work to save the model
# model.save("signature_feature_cnn.h5")
# To reload:
# model = tf.keras.models.load_model("signature_feature_cnn.h5", custom_objects={'bce_dice_loss': bce_dice_loss})
