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

    image_files = sorted(os.listdir(image_dir))
    X, y = [], []

    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        #may need updating later once other datasets are implemented
        label_path = os.path.join(label_dir, filename)

        if not os.path.exists(label_path):
            print(f"Missing label for {filename}, skipping.")
            continue

        #loads signature to be greyscale
        img = load_img(image_path, color_mode='grayscale', target_size=target_size)
        img_arr = img_to_array(img) / 255.0
        X.append(img_arr)

        #loads the marked signautres so each type is its own color
        #will probably have to be updated again once more datasets get added
        mask = load_img(label_path, color_mode='rgb', target_size=target_size)
        mask_arr = img_to_array(mask) / 255.0

        #makes each color a binary
        mask_arr = (mask_arr > 0.5).astype(np.float32)  # threshold to binary
        y.append(mask_arr)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"Loaded {len(X)} samples from {image_dir}")

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
print(f"Test accuracy: {test_acc:.4f}, Test loss: {test_loss:.4f}")

#displaying results
preds = model.predict(X_test[:5])

# visualize the loaded labels
for i in range(2):
    plt.figure(figsize=(10,4))
    plt.subplot(1,4,1)
    plt.imshow(X_train[i].squeeze(), cmap='gray')
    plt.title("Signature")
    plt.axis('off')

    #Lift
    plt.subplot(1,4,2)
    plt.imshow(y_train[i][:,:,0], cmap='Reds')
    plt.title("Pen Lift")
    plt.axis('off')

    #Cross
    plt.subplot(1,4,3)
    plt.imshow(y_train[i][:,:,1], cmap='Greens')
    plt.title("Pen Cross")
    plt.axis('off')

    #Curve
    plt.subplot(1,4,4)
    plt.imshow(y_train[i][:,:,2], cmap='Blues')
    plt.title("Pen Curve")
    plt.axis('off')
    plt.show()


#hopefully will work to save the model
# model.save("signature_feature_cnn.h5")
# To reload:
# model = tf.keras.models.load_model("signature_feature_cnn.h5", custom_objects={'bce_dice_loss': bce_dice_loss})
