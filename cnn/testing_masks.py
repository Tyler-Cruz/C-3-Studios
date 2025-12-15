#proper imports and ignoring warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#makes mask to train the data on
def load_mask(path, target_size=(128,128)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
    return (img > 127).astype(np.float32)[..., np.newaxis]

#loads the dataset
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


#helps train the data correctly by removing uneeded parts of it
def dice_loss(y_true, y_pred, smooth=1e-6):
    yt = K.flatten(y_true)
    yp = K.flatten(y_pred)
    inter = K.sum(yt * yp)
    return 1 - (2 * inter + smooth) / (K.sum(yt) + K.sum(yp) + smooth)

#more help with loss training
def weighted_bce(y_true, y_pred, pos_weight):
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    weight = y_true * pos_weight + (1.0 - y_true)
    return tf.reduce_mean(weight * bce)

#mode loss functions
def dice_bce(y_true, y_pred, w=1.0):
    d = dice_loss(y_true, y_pred)
    b = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return d + w * tf.reduce_mean(b)

#weights parts of the signatures so it is more accurate
def multi_channel_loss(y_true, y_pred):
    curve = dice_bce(y_true[...,0], y_pred[...,0], w=2.0)
    lift  = dice_bce(y_true[...,1], y_pred[...,1], w=1.5)
    cross = dice_bce(y_true[...,2], y_pred[...,2], w=2.0)
    return curve + lift + cross

#helps know what a curve is so it does not go away
def curve_recall(y_true, y_pred):
    # Soft recall (no rounding) to avoid saturation
    yt = K.flatten(y_true[...,0])
    yp = K.flatten(y_pred[...,0])
    tp = K.sum(yt * yp)
    fn = K.sum(yt * (1 - yp))
    return tp / (tp + fn + K.epsilon())

#starts making the model
def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
    return x

#builds the unet
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

            # Output heads (sigmoid for stability)
    curve_out = layers.Conv2D(1, 1, activation="sigmoid", name="curve")(c7)
    lift_out  = layers.Conv2D(1, 1, activation="sigmoid", name="lift")(c7)
    cross_out = layers.Conv2D(1, 1, activation="sigmoid", name="cross")(c7)

    out = layers.Concatenate()([curve_out, lift_out, cross_out])

    return models.Model(inp, out)

#starts training and loads the dataset
image_dir = "Data/Padded_Raw"
curve_dir = "Data/Padded_Curve"
lift_dir  = "Data/Padded_Lift"
cross_dir = "Data/Padded_Cross"

X_train, X_val, X_test, Y_train, Y_val, Y_test = load_dataset(
    image_dir, curve_dir, lift_dir, cross_dir
)

model = build_unet()
model.compile(
    optimizer=Adam(
        learning_rate=1e-4,
        clipnorm=1.0
    ),
    loss=multi_channel_loss,
    metrics=[curve_recall]
)

model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=5,
    batch_size=8
)


#model is trained by this point, helps debug and make sure everything is good in the model
idx = np.random.randint(len(X_val))
img = X_val[idx]
true = Y_val[idx]
pred = model.predict(img[np.newaxis])[0]

#convert to numpy
curve_p = pred[..., 0]
lift_p  = pred[..., 1]
cross_p = pred[..., 2]

print(
    "Curve pred min/max/mean:",
    curve_p.min(),
    curve_p.max(),
    curve_p.mean()
)

print(
    "Lift  pred min/max/mean:",
    lift_p.min(),
    lift_p.max(),
    lift_p.mean()
)

print(
    "Cross pred min/max/mean:",
    cross_p.min(),
    cross_p.max(),
    cross_p.mean()
)


fig, axs = plt.subplots(3, 4, figsize=(12, 9))

axs[0,0].imshow(img.squeeze(), cmap='gray'); axs[0,0].set_title('Input')
axs[0,1].imshow(true[...,0], cmap='gray'); axs[0,1].set_title('GT Curve')
axs[0,2].imshow(tf.sigmoid(pred[...,0]), cmap='gray'); axs[0,2].set_title('Pred Curve (raw)')
axs[0,3].imshow(tf.sigmoid(pred[...,0]) > 0.3, cmap='gray'); axs[0,3].set_title('Pred Curve (thr)')

axs[1,0].imshow(true[...,1], cmap='gray'); axs[1,0].set_title('GT Lift')
axs[1,1].imshow(tf.sigmoid(pred[...,1]), cmap='gray'); axs[1,1].set_title('Pred Lift (raw)')
axs[1,2].imshow(tf.sigmoid(pred[...,1]) > 0.3, cmap='gray'); axs[1,2].set_title('Pred Lift (thr)')
axs[1,3].axis('off')

axs[2,0].imshow(true[...,2], cmap='gray'); axs[2,0].set_title('GT Cross')
axs[2,1].imshow(tf.sigmoid(pred[...,2]), cmap='gray'); axs[2,1].set_title('Pred Cross (raw)')
axs[2,2].imshow(tf.sigmoid(pred[...,2]) > 0.3, cmap='gray'); axs[2,2].set_title('Pred Cross (thr)')
axs[2,3].axis('off')

for ax in axs.flatten(): ax.axis('off')
plt.tight_layout(); plt.show()

#saves the models
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/pen_unet.keras")
print("Model saved successfully")
