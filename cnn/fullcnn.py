#proper imports and warning ignores
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

#loads dataset
def load_mask(path, target_size=(128,128)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return np.zeros(target_size + (1,), dtype=np.float32)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
    return (img < 127).astype(np.float32)[..., np.newaxis]

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

    if len(X) == 0:
        raise ValueError("No images found. Check directory paths.")

    X_train, X_tmp, Y_train, Y_tmp = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_tmp, Y_tmp, test_size=0.5, random_state=42)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

#loss function (helps it know what is needed)
def simple_loss(y_true, y_pred):
    #convert logits to probabilities for dice 
    y_pred_prob = tf.sigmoid(y_pred)
    
    #Binary Cross Entropy (With Logits for numerical stability)
    #BCE is separate
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    bce = tf.reduce_mean(bce)

    #dice Loss (used for thin lines such as signatures)
    numerator = 2.0 * tf.reduce_sum(y_true * y_pred_prob, axis=[1, 2, 3])
    denominator = tf.reduce_sum(y_true + y_pred_prob, axis=[1, 2, 3])
    dice = 1.0 - (numerator + 1e-6) / (denominator + 1e-6)
    
    #combined Loss
    return bce + tf.reduce_mean(dice)

#creates the model (CNN and UNet implementation)
def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
    return x

def build_unet(input_shape=(128,128,1)):
    inp = layers.Input(shape=input_shape)

    #creates augmentation layers to know what parts of the image matters
    c1 = conv_block(inp, 16)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 32)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 64)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 128)

    u1 = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(c4)
    u1 = layers.Concatenate()([u1, c3])
    c5 = conv_block(u1, 64)

    u2 = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(c5)
    u2 = layers.Concatenate()([u2, c2])
    c6 = conv_block(u2, 32)

    u3 = layers.Conv2DTranspose(16, 3, strides=2, padding="same")(c6)
    u3 = layers.Concatenate()([u3, c1])
    c7 = conv_block(u3, 16)

    #hutput heads (logits)
    curve_out = layers.Conv2D(1, 1, activation=None, name="curve")(c7)
    lift_out  = layers.Conv2D(1, 1, activation=None, name="lift")(c7)
    cross_out = layers.Conv2D(1, 1, activation=None, name="cross")(c7)

    out = layers.Concatenate()([curve_out, lift_out, cross_out])

    return models.Model(inp, out)

#actually training the data

#grabs and loads the data
image_dir = "Data/Padded_Raw"
curve_dir = "Data/Padded_Curve"
lift_dir  = "Data/Padded_Lift"
cross_dir = "Data/Padded_Cross"

X_train, X_val, X_test, Y_train, Y_val, Y_test = load_dataset(
    image_dir, curve_dir, lift_dir, cross_dir
)

model = build_unet()
model.compile(
    optimizer=Adam(learning_rate=1e-3), #converges the data to train better
    loss=simple_loss
)

print("Starting training...")
model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=15, #how many iterations
    batch_size=8
)



#model is fully trained at this point, it is just to debug and make sure everything is right
def count_components(mask_thr):
    #counts mask and gets the count
    mask_u8 = (mask_thr * 255).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(mask_u8)
    return num_labels - 1 # Subtract background

#pick random image
idx = np.random.randint(len(X_val))
img = X_val[idx]
true = Y_val[idx]

#get Logits
pred_logits = model.predict(img[np.newaxis])[0]

#sigmoid -> probabilities
pred_probs = tf.sigmoid(pred_logits).numpy()

curve_p = pred_probs[..., 0]
lift_p  = pred_probs[..., 1]
cross_p = pred_probs[..., 2]

#threshold at 0.5 (can change but this works for now)
curve_thr = curve_p > 0.5
lift_thr  = lift_p  > 0.5
cross_thr = cross_p > 0.5

#count components (complexity)
n_curves = count_components(curve_thr)
n_lifts  = count_components(lift_thr)
n_cross  = count_components(cross_thr)

print(f"\n--- Analysis for Image {idx} ---")
print(f"Predicted Curves: {n_curves}")
print(f"Predicted Lifts:  {n_lifts}")
print(f"Predicted Crosses:{n_cross}")

#debug stats to see if the mdoel is accurate
print("\nStats (should be close to 0 or 1, not 0.3):")
print(f"Curve Max Prob: {curve_p.max():.4f}")
print(f"Lift Max Prob:  {lift_p.max():.4f}")

fig, axs = plt.subplots(3, 4, figsize=(12, 9))

axs[0,0].imshow(img.squeeze(), cmap='gray'); axs[0,0].set_title('Input')
axs[0,1].imshow(true[...,0], cmap='gray'); axs[0,1].set_title('GT Curve')
axs[0,2].imshow(curve_p, cmap='gray', vmin=0, vmax=1); axs[0,2].set_title('Pred Curve (Prob)')
axs[0,3].imshow(curve_thr, cmap='gray'); axs[0,3].set_title(f'Pred Curve (Thresh)\nCount: {n_curves}')

axs[1,0].imshow(true[...,1], cmap='gray'); axs[1,0].set_title('GT Lift')
axs[1,1].imshow(lift_p, cmap='gray', vmin=0, vmax=1); axs[1,1].set_title('Pred Lift (Prob)')
axs[1,2].imshow(lift_thr, cmap='gray'); axs[1,2].set_title(f'Pred Lift (Thresh)\nCount: {n_lifts}')
axs[1,3].axis('off')

axs[2,0].imshow(true[...,2], cmap='gray'); axs[2,0].set_title('GT Cross')
axs[2,1].imshow(cross_p, cmap='gray', vmin=0, vmax=1); axs[2,1].set_title('Pred Cross (Prob)')
axs[2,2].imshow(cross_thr, cmap='gray'); axs[2,2].set_title(f'Pred Cross (Thresh)\nCount: {n_cross}')
axs[2,3].axis('off')

for ax in axs.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()

#save the model
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/model.keras")
print("Model saved successfully")