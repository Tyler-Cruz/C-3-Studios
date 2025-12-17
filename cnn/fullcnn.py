#imports and warning ignores
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence # <-- Import Sequence
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau

#loads the dataset and mask sizes
def load_mask(path, target_size=(128,128)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return np.zeros(target_size + (1,), dtype=np.float32)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
    return (img < 127).astype(np.float32)[..., np.newaxis] 


def load_dataset_multi_output(image_dir, curve_dir, lift_dir, cross_dir,
                 target_size=(128,128)):
    #loads and splits x and the three ys separately
    X, Y_curve, Y_lift, Y_cross = [], [], [], []
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

        Y_curve.append(load_mask(cp, target_size))
        Y_lift.append(load_mask(lp, target_size))
        Y_cross.append(load_mask(xp, target_size))
        X.append(img)

    X = np.asarray(X, np.float32)
    Y_curve = np.asarray(Y_curve, np.float32)
    Y_lift = np.asarray(Y_lift, np.float32)
    Y_cross = np.asarray(Y_cross, np.float32)

    if len(X) == 0:
        raise ValueError("No images found. Check your directory paths.")

    #splits logic for x and all three y
    (X_train, X_tmp, 
     Y_curve_train, Y_curve_tmp, 
     Y_lift_train, Y_lift_tmp, 
     Y_cross_train, Y_cross_tmp) = train_test_split(
        X, Y_curve, Y_lift, Y_cross, test_size=0.2, random_state=42
    )
    
    (X_val, X_test, 
     Y_curve_val, Y_curve_test, 
     Y_lift_val, Y_lift_test, 
     Y_cross_val, Y_cross_test) = train_test_split(
        X_tmp, Y_curve_tmp, Y_lift_tmp, Y_cross_tmp, test_size=0.5, random_state=42
    )

    Y_train_list = [Y_curve_train, Y_lift_train, Y_cross_train]
    Y_val_list = [Y_curve_val, Y_lift_val, Y_cross_val]
    Y_test_list = [Y_curve_test, Y_lift_test, Y_cross_test]

    return X_train, X_val, X_test, Y_train_list, Y_val_list, Y_test_list


#creates data to help train the dataset to make up for lack of natural data
class SegmentationDataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size, augment=True):
        self.x = x_set
        # y_set is a list: [Y_curve, Y_lift, Y_cross]
        self.y_curve, self.y_lift, self.y_cross = y_set[0], y_set[1], y_set[2] 
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(self.x))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.x) / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

    def __getitem__(self, index):
        #gets batch index
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        #loads the batches
        X_batch = self.x[batch_indices]
        Yc_batch = self.y_curve[batch_indices]
        Yl_batch = self.y_lift[batch_indices]
        Yx_batch = self.y_cross[batch_indices]

        #augments and returns
        if self.augment:
            X_aug, Yc_aug, Yl_aug, Yx_aug = self._data_augmentation(X_batch, Yc_batch, Yl_batch, Yx_batch)
            #returns tuple
            return X_aug, (Yc_aug, Yl_aug, Yx_aug) 
        
        #returns tuple
        return X_batch, (Yc_batch, Yl_batch, Yx_batch)

    def _data_augmentation(self, X_batch, Yc_batch, Yl_batch, Yx_batch):
        X_aug, Yc_aug, Yl_aug, Yx_aug = [], [], [], []
        
        for i in range(X_batch.shape[0]):
            img = X_batch[i]
            #concatenate masks for transformation
            mask_combined = np.concatenate([Yc_batch[i], Yl_batch[i], Yx_batch[i]], axis=-1)
            
            #rotates for more data
            angle = np.random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((64, 64), angle, 1) #gets the center
            
            #transformation to image (cubic interpolation)
            img_rot = cv2.warpAffine(img, M, (128, 128), flags=cv2.INTER_CUBIC)[..., np.newaxis]
            
            #transformation to masks (nearest interpolation)
            mask_rot = cv2.warpAffine(mask_combined, M, (128, 128), flags=cv2.INTER_NEAREST)
            
            #random horizontal flip
            if np.random.rand() < 0.5:
                img_rot = np.fliplr(img_rot)
                mask_rot = np.fliplr(mask_rot)
            
            X_aug.append(img_rot)
            Yc_aug.append(mask_rot[..., 0:1])
            Yl_aug.append(mask_rot[..., 1:2])
            Yx_aug.append(mask_rot[..., 2:3])
            
        return (
            np.asarray(X_aug, np.float32), 
            np.asarray(Yc_aug, np.float32), 
            np.asarray(Yl_aug, np.float32), 
            np.asarray(Yx_aug, np.float32)
        )

#loss function
def simple_loss(y_true, y_pred):
    y_pred_prob = tf.sigmoid(y_pred)
    bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    bce = tf.reduce_mean(bce)

    numerator = 2.0 * tf.reduce_sum(y_true * y_pred_prob, axis=[1, 2, 3])
    denominator = tf.reduce_sum(y_true + y_pred_prob, axis=[1, 2, 3])
    dice = 1.0 - (numerator + 1e-6) / (denominator + 1e-6)
    
    return bce + tf.reduce_mean(dice)

# -------------------------------------------------
# MODEL (MULTI-OUTPUT RETAINED)
# -------------------------------------------------
#residual block for model and mapping
def residual_block(input_tensor, filters):
    # 1x1 convolution for identity mapping shortcut (if needed)
    shortcut = input_tensor
    if K.int_shape(shortcut)[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)

    #path of training: Conv -> BN -> ReLU -> Conv -> BN -> Add Shortcut -> ReLU
    
    #block 1
    x = layers.Conv2D(filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    #block 2 with skip
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    #shortcut and finalize with ReLU
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

#model creation
def build_resnet_unet(input_shape=(128,128,1)):
    inp = layers.Input(shape=input_shape)

    #encoder path
    
    #level 1
    c1 = residual_block(inp, 16)
    p1 = layers.MaxPooling2D()(c1)
    p1 = layers.SpatialDropout2D(0.1)(p1) 

    #level 2
    c2 = residual_block(p1, 32)
    p2 = layers.MaxPooling2D()(c2)
    p2 = layers.SpatialDropout2D(0.2)(p2) 

    #level 3
    c3 = residual_block(p2, 64)
    p3 = layers.MaxPooling2D()(c3)
    p3 = layers.SpatialDropout2D(0.2)(p3) 

    #bottleneck
    c4 = residual_block(p3, 128) 

    #decoder (Conv2DTranspose for upsampling)

    u1 = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(c4)
    u1 = layers.Concatenate()([u1, c3])
    c5 = residual_block(u1, 64)

    u2 = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(c5)
    u2 = layers.Concatenate()([u2, c2])
    c6 = residual_block(u2, 32)

    u3 = layers.Conv2DTranspose(16, 3, strides=2, padding="same")(c6)
    u3 = layers.Concatenate()([u3, c1])
    c7 = residual_block(u3, 16)

    #output layers
    curve_out = layers.Conv2D(1, 1, activation=None, name="curve_output")(c7)
    lift_out  = layers.Conv2D(1, 1, activation=None, name="lift_output")(c7)
    cross_out = layers.Conv2D(1, 1, activation=None, name="cross_output")(c7)

    return models.Model(inp, [curve_out, lift_out, cross_out])

#training the dataset
image_dir = "Data/Padded_Raw"
curve_dir = "Data/Padded_Curve"
lift_dir  = "Data/Padded_Lift"
cross_dir = "Data/Padded_Cross"
BATCH_SIZE = 8

#load data
X_train, X_val, X_test, Y_train_list, Y_val_list, Y_test_list = load_dataset_multi_output(
    image_dir, curve_dir, lift_dir, cross_dir
)

#start generators
train_gen = SegmentationDataGenerator(X_train, Y_train_list, BATCH_SIZE, augment=True)
val_gen = SegmentationDataGenerator(X_val, Y_val_list, BATCH_SIZE, augment=False) 


model = build_resnet_unet()

#define scheduler callback
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,       
    patience=15,      
    min_lr=1e-6,      
    verbose=1
)

#loss dictionary
model.compile(
    optimizer=Adam(learning_rate=3e-4, clipnorm=1.0),
    loss={
        "curve_output": simple_loss,
        "lift_output": simple_loss,
        "cross_output": simple_loss
    },
    loss_weights={
        "curve_output": 1.0,
        "lift_output": 1.0,
        "cross_output": 1.0
    }
)

print("Starting FINAL training (ResU-Net + LR Scheduler)...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=100, 
    callbacks=[lr_scheduler]
)

#visualizing the trained data and helper functions to debug
def count_components_post_processed(mask_thr, kernel_size=1): 
    #checks to see if kernel needs to be smoothed
    if kernel_size > 1:
        mask_u8 = (mask_thr * 255).astype(np.uint8)
        kernel = np.ones((kernel_size, kernel_size), np.uint8) 
        dilated_mask = cv2.dilate(mask_u8, kernel, iterations=1)
        cleaned_mask = cv2.erode(dilated_mask, kernel, iterations=1)
    else:
        #minimal merging for raw masks
        cleaned_mask = (mask_thr * 255).astype(np.uint8)

    num_labels, _ = cv2.connectedComponents(cleaned_mask)
    
    return num_labels - 1

#gets random image for validation
idx = np.random.randint(len(X_val))
img = X_val[idx]

#prediction returns list of three arrays
pred_logits_list = model.predict(img[np.newaxis])

curve_logits = pred_logits_list[0][0]
lift_logits = pred_logits_list[1][0]
cross_logits = pred_logits_list[2][0]

curve_p = tf.sigmoid(curve_logits).numpy()
lift_p = tf.sigmoid(lift_logits).numpy()
cross_p = tf.sigmoid(cross_logits).numpy()

curve_thr = curve_p > 0.78 
lift_thr = lift_p > 0.2 #good
cross_thr = cross_p > 0.83 #good

n_curves = count_components_post_processed(curve_thr)
n_lifts = count_components_post_processed(lift_thr)
n_cross = count_components_post_processed(cross_thr)

#list of [Y_curve, Y_lift, Y_cross]
Y_curve_val_arr, Y_lift_val_arr, Y_cross_val_arr = Y_val_list
true_curve = Y_curve_val_arr[idx]
true_lift = Y_lift_val_arr[idx]
true_cross = Y_cross_val_arr[idx]

print(f"\n--- Analysis for Image {idx} ---")
print(f"Predicted Curves (P-P): {n_curves}")
print(f"Predicted Lifts (P-P):  {n_lifts}")
print(f"Predicted Crosses (P-P):{n_cross}")

print("\nStats:")
print(f"Curve Max Prob: {curve_p.max():.4f}")
print(f"Lift Max Prob:  {lift_p.max():.4f}")

fig, axs = plt.subplots(3, 4, figsize=(12, 9))

axs[0,0].imshow(img.squeeze(), cmap='gray'); axs[0,0].set_title('Input')
axs[0,1].imshow(true_curve.squeeze(), cmap='gray'); axs[0,1].set_title('GT Curve')
axs[0,2].imshow(curve_p.squeeze(), cmap='gray', vmin=0, vmax=1); axs[0,2].set_title('Pred Curve (Prob)')
axs[0,3].imshow(curve_thr.squeeze(), cmap='gray'); axs[0,3].set_title(f'Pred Curve (Thresh)\nCount: {n_curves}')

axs[1,0].imshow(true_lift.squeeze(), cmap='gray'); axs[1,0].set_title('GT Lift')
axs[1,1].imshow(lift_p.squeeze(), cmap='gray', vmin=0, vmax=1); axs[1,1].set_title('Pred Lift (Prob)')
axs[1,2].imshow(lift_thr.squeeze(), cmap='gray'); axs[1,2].set_title(f'Pred Lift (Thresh)\nCount: {n_lifts}')
axs[1,3].axis('off')

axs[2,0].imshow(true_cross.squeeze(), cmap='gray'); axs[2,0].set_title('GT Cross')
axs[2,1].imshow(cross_p.squeeze(), cmap='gray', vmin=0, vmax=1); axs[2,1].set_title('Pred Cross (Prob)')
axs[2,2].imshow(cross_thr.squeeze(), cmap='gray'); axs[2,2].set_title(f'Pred Cross (Thresh)\nCount: {n_cross}')
axs[2,3].axis('off')

for ax in axs.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()

#saves model to be used later
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/model.keras")
print("Model saved successfully")