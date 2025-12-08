from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)  # Enables the frontend to connect without triggering potential security blocks

# --- LOAD MODEL ---
MODEL_PATH = "saved_model/model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load model. {e}")
    model = None

def preprocess_image(file, target_size=(128,128)):
    # 1. Read Image
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # 2. Binarize (Cleaning)
    # Forces grey background to White (255) and Ink to Black (0)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Smart Resize (Padding)
    h, w = img.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    # Create white canvas
    canvas = np.full(target_size, 255, dtype=np.uint8)
    top = (target_size[0] - new_h) // 2
    left = (target_size[1] - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = img_resized

    # 4. Normalize (0.0 to 1.0)
    # Important: Model expects Ink=0.0, Paper=1.0
    canvas_norm = canvas.astype("float32") / 255.0
    
    # --- DEBUG: SAVE WHAT THE AI SEES ---
    # We multiply by 255 to make it visible again for the PNG
    #cv2.imwrite("debug_input_to_ai.png", (canvas_norm * 255).astype(np.uint8))
    
    return np.expand_dims(np.expand_dims(canvas_norm, axis=-1), axis=0)

def count_components(mask, name="debug", threshold=0.5, min_pixel_size=10):
    """
    Counts connected components with noise reduction.
    """
    # 1. Threshold: Convert probability map to binary (0 or 1)
    mask_bin = (mask > threshold).astype(np.uint8) * 255

    # Debug: Save what the AI sees after thresholding
    #cv2.imwrite(f"debug_result_{name}.png", mask_bin)
    
    # 2. Noise Removal (Morphological Opening): Removes tiny specks
    kernel = np.ones((3,3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 3. Connected Components with Stats
    # We use connectedComponentsWithStats to get the size (area) of each blob
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned)
    
    # 4. Filter by Size
    # Label 0 is always the background, so we start range at 1
    true_count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_pixel_size:
            true_count += 1
            
    return true_count

@app.route("/analyze", methods=["POST"])
def analyze():
    if model is None:
        return jsonify({"error": "Model not loaded on server."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    try:
        file = request.files["file"]
        
        # Preprocess
        input_tensor = preprocess_image(file)

        # Predict (Returns: [Batch, Height, Width, Channels])
        pred = model.predict(input_tensor, verbose=0)[0]

        # Extract Channels:
        # Channel 0: Curves
        # Channel 1: Lifts
        # Channel 2: Crosses
        curve_mask = pred[..., 0]
        lift_mask  = pred[..., 1]
        cross_mask = pred[..., 2]

        # Count with noise filtering
        curve_count = count_components(curve_mask, name="curve")
        lift_count  = count_components(lift_mask, name="lift")
        cross_count = count_components(cross_mask, name="cross")

        # Calculate Score
        complexity_score = curve_count + lift_count + cross_count

        return jsonify({
            "score": float(complexity_score),
            "lifts": int(lift_count),
            "reversals": int(curve_count),
            "intersections": int(cross_count)
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Server starting on http://127.0.0.1:5000")
    app.run(port=5000, debug=True)