#imports and warning ignores
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
import pandas as pd  # <-- Added pandas for CSV handling
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau

def load_dataset_multi_output(image_dir, csv_path, target_size=(128,128)):
    # Loads images and pulls true values from CSV
    X, Y_curve, Y_lift, Y_cross = [], [], [], []
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise ValueError(f"CSV file not found at {csv_path}")

    # Ensure column names match user requirements (strip whitespace just in case)
    df.columns = [c.strip() for c in df.columns]
    required_cols = ['Image name', 'Curve Value', 'Cross Value', 'Lift value']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Create a dictionary for fast lookup: image_name -> row data
    data_map = df.set_index('Image name').to_dict('index')
    
    files = sorted(os.listdir(image_dir))

    # Load targets (list of measured values)
    for f in files:
        img_path = os.path.join(image_dir, f)

        # Check if file exists and if it is in our CSV data
        if not os.path.exists(img_path):
            continue
        
        # logic to handle if CSV names include extension or not
        # We try exact match first
        csv_key = f
        if csv_key not in data_map:
             # try removing extension if exact match failed
             csv_key = os.path.splitext(f)[0]
        
        if csv_key not in data_map:
            continue # Skip images not found in CSV

        # Load Image
        img = load_img(img_path, color_mode='grayscale', target_size=target_size)
        img = img_to_array(img) / 255.0

        # Load Values from Map
        row = data_map[csv_key]
        Y_curve.append(row['Curve Value'])
        Y_lift.append(row['Lift value'])
        Y_cross.append(row['Cross Value'])
        X.append(img)

    X = np.asarray(X, np.float32)
    # Reshape Ys to (N, 1) for regression
    Y_curve = np.asarray(Y_curve, np.float32).reshape(-1, 1)
    Y_lift = np.asarray(Y_lift, np.float32).reshape(-1, 1)
    Y_cross = np.asarray(Y_cross, np.float32).reshape(-1, 1)

    if len(X) == 0:
        raise ValueError("No matched images found. Check directory paths and CSV filenames.")

    # Splits logic for x and all three y
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


# Updated Generator for Regression (Scalar targets don't need spatial augmentation)
class RegressionDataGenerator(Sequence):
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
        # Gets batch index
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Loads the batches
        X_batch = self.x[batch_indices]
        Yc_batch = self.y_curve[batch_indices]
        Yl_batch = self.y_lift[batch_indices]
        Yx_batch = self.y_cross[batch_indices]

        # Augments and returns
        if self.augment:
            X_aug = self._data_augmentation(X_batch)
            # Targets do not change with rotation/flip in this context
            return X_aug, (Yc_batch, Yl_batch, Yx_batch) 
        
        return X_batch, (Yc_batch, Yl_batch, Yx_batch)

    def _data_augmentation(self, X_batch):
        X_aug = []
        for i in range(X_batch.shape[0]):
            img = X_batch[i]
            
            # Rotation
            angle = np.random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((64, 64), angle, 1) 
            img_rot = cv2.warpAffine(img, M, (128, 128), flags=cv2.INTER_CUBIC)[..., np.newaxis]
            
            # Random horizontal flip
            if np.random.rand() < 0.5:
                img_rot = np.fliplr(img_rot)
            
            X_aug.append(img_rot)
            
        return np.asarray(X_aug, np.float32)

# -------------------------------------------------
# MODEL (MULTI-OUTPUT REGRESSION)
# -------------------------------------------------
def residual_block(input_tensor, filters):
    shortcut = input_tensor
    if K.int_shape(shortcut)[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same')(shortcut)

    x = layers.Conv2D(filters, (3, 3), padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    
    return x

def build_resnet_regression(input_shape=(128,128,1)):

    inp = layers.Input(shape=input_shape)

    # Encoder path
    c1 = residual_block(inp, 16)
    p1 = layers.MaxPooling2D()(c1)
    p1 = layers.SpatialDropout2D(0.1)(p1) 

    c2 = residual_block(p1, 32)
    p2 = layers.MaxPooling2D()(c2)
    p2 = layers.SpatialDropout2D(0.2)(p2) 

    c3 = residual_block(p2, 64)
    p3 = layers.MaxPooling2D()(c3)
    p3 = layers.SpatialDropout2D(0.2)(p3) 

    # Bottleneck
    c4 = residual_block(p3, 128) 
    
    # Global Pooling (Flattening the spatial dimensions)
    x = layers.GlobalAveragePooling2D()(c4)
    x = layers.Dropout(0.4)(x)

    # Output heads (Dense layers for scalar prediction)
    # Curve Branch
    dense_c = layers.Dense(64, activation='relu')(x)
    curve_out = layers.Dense(1, activation='linear', name="curve_output")(dense_c)
    
    # Lift Branch
    dense_l = layers.Dense(64, activation='relu')(x)
    lift_out = layers.Dense(1, activation='linear', name="lift_output")(dense_l)

    # Cross Branch
    dense_x = layers.Dense(64, activation='relu')(x)
    cross_out = layers.Dense(1, activation='linear', name="cross_output")(dense_x)

    return models.Model(inp, [curve_out, lift_out, cross_out])

# Training setup
image_dir = "Data/Padded_Raw"
csv_path = "Data/data_values.csv" 
BATCH_SIZE = 8

# Load data

try:
    X_train, X_val, X_test, Y_train_list, Y_val_list, Y_test_list = load_dataset_multi_output(
        image_dir, csv_path
    )
    
    # Start generators
    train_gen = RegressionDataGenerator(X_train, Y_train_list, BATCH_SIZE, augment=True)
    val_gen = RegressionDataGenerator(X_val, Y_val_list, BATCH_SIZE, augment=False) 

    model = build_resnet_regression()

    # Define scheduler callback
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5,       
        patience=10,      
        min_lr=1e-6,      
        verbose=1
    )

# Compile with MSE loss for regression
    model.compile(
        optimizer=Adam(learning_rate=3e-4, clipnorm=1.0),
        loss={
            "curve_output": "mse",
            "lift_output": "mse",
            "cross_output": "mse"
        },
        loss_weights={
            "curve_output": 1.0,
            "lift_output": 1.0,
            "cross_output": 1.0
        },
        # Map the 'mae' metric to each specific output name
        metrics={
            "curve_output": "mae",
            "lift_output": "mae",
            "cross_output": "mae"
        }

    )

    print("Starting FINAL training (Regression ResNet + LR Scheduler)...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=100, 
        callbacks=[lr_scheduler]
    )

    # Visualization of Regression Results
    idx = np.random.randint(len(X_val))
    img = X_test[idx]

    # Prediction
    pred_list = model.predict(img[np.newaxis])
    pred_curve = pred_list[0][0][0]
    pred_lift = pred_list[1][0][0]
    pred_cross = pred_list[2][0][0]

    # True Values
    true_curve = Y_test_list[0][idx][0]
    true_lift = Y_test_list[1][idx][0]
    true_cross = Y_test_list[2][idx][0]

    print(f"\n--- Analysis for Image {idx} ---")
    print(f"{'Feature':<10} | {'True':<10} | {'Pred':<10} | {'Diff':<10}")
    print("-" * 46)
    print(f"{'Curve':<10} | {true_curve:<10.2f} | {pred_curve:<10.2f} | {abs(true_curve-pred_curve):<10.2f}")
    print(f"{'Lift':<10} | {true_lift:<10.2f} | {pred_lift:<10.2f} | {abs(true_lift-pred_lift):<10.2f}")
    print(f"{'Cross':<10} | {true_cross:<10.2f} | {pred_cross:<10.2f} | {abs(true_cross-pred_cross):<10.2f}")

    plt.figure(figsize=(6, 6))
    plt.imshow(img.squeeze(), cmap='gray')
    plt.title(f"Input Image\nTrue: C={true_curve:.1f}, L={true_lift:.1f}, X={true_cross:.1f}\nPred: C={pred_curve:.1f}, L={pred_lift:.1f}, X={pred_cross:.1f}")
    plt.axis('off')
    plt.show()

    # Save model
    os.makedirs("saved_model", exist_ok=True)
    model.save("saved_model/regression_model.keras")
    print("Model saved successfully")

except ValueError as e:
    print(f"Error loading data: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

#loop through all sets (test, val, training)
#get id for each image
    #get true and pred for each image
#for monday

#k fold validation =5
#see how it goes if ambitious

#be able to graph it later (dr sheets)