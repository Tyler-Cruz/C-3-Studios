from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img

app = Flask(__name__)

# ---- Load your trained model ----
MODEL_PATH = "saved_model/model.keras"   # adjust if needed
model = tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess_image(file, target_size=(128,128)):
    # read image bytes -> numpy
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return np.expand_dims(img, axis=0)    # shape: (1,128,128,1)


def count_components(mask, threshold=0.5):
    mask_bin = (mask > threshold).astype(np.uint8)
    n, _ = cv2.connectedComponents(mask_bin)
    return n - 1


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # preprocess
    x = preprocess_image(file)

    # predict
    pred = model.predict(x)[0]   # shape: (128,128,3)

    curve = count_components(pred[..., 0])
    lift  = count_components(pred[..., 1])
    cross = count_components(pred[..., 2])

    # you can define complexity however you want
    complexity_score = curve + lift + cross

    return jsonify({
        "score": float(complexity_score),
        "lifts": int(lift),
        "reversals": int(curve),
        "intersections": int(cross)
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)
