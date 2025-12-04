import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# --- LOAD MODEL ---
MODEL_PATH = "saved_model/model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load model. {e}")
    model = None

def preprocess_image(file, target_size=(128,128)):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return np.expand_dims(img, axis=0)

def count_components(mask, name="debug", threshold=0.5, min_pixel_size=5):
    """
    Counts blobs but also saves the mask as an image for debugging.
    """
    # 1. Threshold
    mask_bin = (mask > threshold).astype(np.uint8) * 255
    
    # 2. SAVE DEBUG IMAGE (Crucial Step!)
    # This saves the AI's view to your folder so you can inspect it.
    cv2.imwrite(f"debug_output_{name}.png", mask_bin)
    
    # 3. Noise Removal (Morphological Opening)
    # This removes tiny specks that might be causing your "1" score.
    kernel = np.ones((3,3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 4. Count Components with Size Filter
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned)
    
    true_count = 0
    # Start loop at 1 to ignore background (label 0)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_pixel_size:
            true_count += 1
            
    return true_count

@app.route("/analyze", methods=["POST"])
def analyze():
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    try:
        file = request.files["file"]
        input_tensor = preprocess_image(file)

        # Predict
        pred = model.predict(input_tensor, verbose=0)[0]

        # Extract & Count (With Debug Names)
        # Verify your channel order from fullcnn.py: [Curve, Lift, Cross]
        curve_count = count_components(pred[..., 0], name="curve")
        lift_count  = count_components(pred[..., 1], name="lift")
        cross_count = count_components(pred[..., 2], name="cross")

        complexity_score = curve_count + lift_count + cross_count

        print(f"Analyzed: Lifts={lift_count}, Curves={curve_count}, Crosses={cross_count}")

        return jsonify({
            "score": float(complexity_score),
            "lifts": int(lift_count),
            "reversals": int(curve_count),
            "intersections": int(cross_count)
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)