# =============================================
# FINAL, KNOWN-GOOD TRAINING SCRIPT
# Pen Curves / Lifts / Intersections Segmentation
# =============================================

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# -------------------------------------------------
# DATA LOADING (MASK-SAFE)
# -------------------------------------------------

def load_mask(path, target_size=(128,128)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
    return (img > 127).astype(np.float32)[..., np.newaxis]


def load_dataset(image_dir, curve_dir, lift_dir, cross_dir,
                 target_size=(128,128)):
    X, Y = [], []
    files = sorted(os.listdir(image_dir))

    for f in files:
        ip = os.path.join(image_dir, f)
        cp = os.path.join(curve_dir, f)
        lp = os.path.join(lift_dir, f)
        xp = os.path.join(cross_dir, f)

        if not all(os.path.exists(p) for p in [ip, cp, lp, xp]):
            continue

        img = load_img(ip, color_mode='grayscale', target_size=target_size)
        img = img_to_array(img) / 255.0

        curve = load_mask(cp, target_size)
        lift  = load_mask(lp, target_size)
        cross = load_mask(xp, target_size)

        X.append(img)
        Y.append(np.concatenate([curve, lift, cross], axis=-1))

    X = np.asarray(X, np.float32)
    Y = np.asarray(Y, np.float32)

    X_train, X_tmp, Y_train, Y_tmp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_tmp, Y_tmp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

# -------------------------------------------------
# LOSSES (ANTI-COLLAPSE)
# -------------------------------------------------

def dice_loss(y_true, y_pred, smooth=1e-6):
    yt = K.flatten(y_true)
    yp = K.flatten(y_pred)
    inter = K.sum(yt * yp)
    return 1 - (2 * inter + smooth) / (K.sum(yt) + K.sum(yp) + smooth)


def weighted_bce(y_true, y_pred, pos_weight):
    """
    y_true, y_pred: (B, H, W)
    Use backend BCE to avoid unintended axis reduction
    """
    y_pred = tf.clip_by_value(y_pred, 1e-4, 1.0 - 1e-4)

    # Elementwise BCE: (B, H, W)
    bce = K.binary_crossentropy(y_true, y_pred)

    # Pixel-wise weighting: (B, H, W)
    weight = y_true * pos_weight + (1.0 - y_true)

    return tf.reduce_mean(weight * bce)


def multi_channel_loss(y_true, y_pred):
    # Reduced, stabilizing positive weights
    curve = dice_loss(y_true[...,0], y_pred[...,0]) + weighted_bce(y_true[...,0], y_pred[...,0], 8.0)
    lift  = dice_loss(y_true[...,1], y_pred[...,1]) + weighted_bce(y_true[...,1], y_pred[...,1], 4.0)
    cross = dice_loss(y_true[...,2], y_pred[...,2]) + weighted_bce(y_true[...,2], y_pred[...,2], 6.0)
    return curve + lift + cross

# -------------------------------------------------
# METRICS
# -------------------------------------------------

def curve_recall(y_true, y_pred):
    yt = K.flatten(y_true[...,0])
    yp = K.flatten(K.round(y_pred[...,0]))
    tp = K.sum(yt * yp)
    fn = K.sum(yt * (1 - yp))
    return tp / (tp + fn + K.epsilon())

# -------------------------------------------------
# MODEL (BASELINE U-NET)
# -------------------------------------------------

def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
    return x


def build_unet(input_shape=(128,128,1)):
    inp = layers.Input(shape=input_shape)

    c1 = conv_block(inp, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 256)

    u1 = layers.UpSampling2D()(c4)
    u1 = layers.Concatenate()([u1, c3])
    c5 = conv_block(u1, 128)

    u2 = layers.UpSampling2D()(c5)
    u2 = layers.Concatenate()([u2, c2])
    c6 = conv_block(u2, 64)

    u3 = layers.UpSampling2D()(c6)
    u3 = layers.Concatenate()([u3, c1])
    c7 = conv_block(u3, 32)

    out = layers.Conv2D(
        3, 1, activation='sigmoid',
        bias_initializer=tf.keras.initializers.Constant(-2.0)
    )(c7)

    return models.Model(inp, out)

# -------------------------------------------------
# TRAINING
# -------------------------------------------------

image_dir = "Data/Padded_Raw"
curve_dir = "Data/Padded_Curve"
lift_dir  = "Data/Padded_Lift"
cross_dir = "Data/Padded_Cross"

X_train, X_val, X_test, Y_train, Y_val, Y_test = load_dataset(
    image_dir, curve_dir, lift_dir, cross_dir
)

model = build_unet()
model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-5),
    loss=multi_channel_loss,
    metrics=[curve_recall]
)

model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=60,
    batch_size=8
)

# -------------------------------------------------
# DEBUG VISUALIZATION
# -------------------------------------------------

idx = np.random.randint(len(X_val))
img = X_val[idx]
true = Y_val[idx]
pred = model.predict(img[np.newaxis])[0]

fig, axs = plt.subplots(3, 4, figsize=(12, 9))

axs[0,0].imshow(img.squeeze(), cmap='gray'); axs[0,0].set_title('Input')
axs[0,1].imshow(true[...,0], cmap='gray'); axs[0,1].set_title('GT Curve')
axs[0,2].imshow(pred[...,0], cmap='gray', vmin=0, vmax=1); axs[0,2].set_title('Pred Curve (raw)')
axs[0,3].imshow(pred[...,0] > 0.3, cmap='gray'); axs[0,3].set_title('Pred Curve (thr)')

axs[1,0].imshow(true[...,1], cmap='gray'); axs[1,0].set_title('GT Lift')
axs[1,1].imshow(pred[...,1], cmap='gray', vmin=0, vmax=1); axs[1,1].set_title('Pred Lift (raw)')
axs[1,2].imshow(pred[...,1] > 0.3, cmap='gray'); axs[1,2].set_title('Pred Lift (thr)')
axs[1,3].axis('off')

axs[2,0].imshow(true[...,2], cmap='gray'); axs[2,0].set_title('GT Cross')
axs[2,1].imshow(pred[...,2], cmap='gray', vmin=0, vmax=1); axs[2,1].set_title('Pred Cross (raw)')
axs[2,2].imshow(pred[...,2] > 0.3, cmap='gray'); axs[2,2].set_title('Pred Cross (thr)')
axs[2,3].axis('off')

for ax in axs.flatten(): ax.axis('off')
plt.tight_layout(); plt.show()

# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------

os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/pen_unet.keras")
print("Model saved successfully")
