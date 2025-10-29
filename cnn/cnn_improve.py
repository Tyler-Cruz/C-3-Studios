import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# dataset loading (does not actually load dataset yet, keeps breaking when try)
def load_signature_dataset():
    """
    Placeholder dataset loader.
    Replace this with actual paths to (image_dir, label_dir).
    y should have 3 channels for (pen_lift, pen_cross, pen_curve)
    """
    X = np.random.rand(100, 128, 128, 1).astype(np.float32)
    y = np.random.randint(0, 2, (100, 128, 128, 3)).astype(np.float32)
    return X, y


X, y = load_signature_dataset()
X = X / 255.0
print("Data shape:", X.shape, y.shape)

# entropy
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

    outputs = layers.Conv2D(3, (1,1), activation='sigmoid')(c5)  # 3 classes: lift, cross, curve

    model = models.Model(inputs, outputs, name="SignatureFeatureExtractor")
    return model


model = create_signature_cnn()
model.summary()

#training section
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=bce_dice_loss,
    metrics=['accuracy']
)

history = model.fit(
    X, y,
    epochs=10,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

#Prediction and Printing
preds = model.predict(X[:5])

for i in range(5):
    #shows original image
    plt.figure(figsize=(14,4))
    plt.subplot(1,5,1)
    plt.title("Original")
    plt.imshow(X[i].squeeze(), cmap='gray')
    plt.axis('off')

    #pen lifts
    plt.subplot(1,5,2)
    plt.title("Pen Lift")
    plt.imshow(preds[i][:,:,0], cmap='hot')
    plt.axis('off')

    #pen crosses
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

    #prints
    plt.show()


#should eventually save the dataset once completed
# model.save("signature_feature_cnn.h5")
# how to load:
# model = tf.keras.models.load_model("signature_feature_cnn.h5", custom_objects={'bce_dice_loss': bce_dice_loss})
