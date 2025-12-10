from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

#loads model
MODEL_PATH = "saved_model/model.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
except Exception as e:
    print(f"ERROR: Could not load model. {e}")
    model = None


#processing image imported by user
def preprocess_image(file, target_size=(128,128)):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = img.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    canvas = np.full(target_size, 255, dtype=np.uint8)
    top = (target_size[0] - new_h) // 2
    left = (target_size[1] - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = img_resized

    canvas_norm = canvas.astype("float32") / 255.0
    return np.expand_dims(np.expand_dims(canvas_norm, axis=-1), axis=0)


#logic for analyzing the images
def count_components(mask, name="debug", threshold=0.5, min_pixel_size=10):
    mask_bin = (mask > threshold).astype(np.uint8) * 255

    kernel = np.ones((3,3), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_cleaned)

    true_count = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_pixel_size:
            true_count += 1

    return true_count


#frontend
@app.route("/index.html")
def index():
    css_ = url_for("static", filename="style.css")
    return render_template("index.html", css_path=css_)


#api logic
@app.route("/analyze", methods=["POST"])
def analyze():
    if model is None:
        return jsonify({"error": "Model not loaded on server."}), 500

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    try:
        file = request.files["file"]
        input_tensor = preprocess_image(file)
        pred = model.predict(input_tensor, verbose=0)[0]

        curve_mask = pred[..., 0]
        lift_mask  = pred[..., 1]
        cross_mask = pred[..., 2]

        curve_count = count_components(curve_mask, name="curve")
        lift_count  = count_components(lift_mask, name="lift")
        cross_count = count_components(cross_mask, name="cross")

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

#actually running the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=49164, debug=True)
