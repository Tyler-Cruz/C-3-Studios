from flask import Flask, render_template, url_for, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

#gets cnn model
MODEL_PATH = "saved_model/model.keras"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

#main page
@app.route('/')
@app.route('/index.html')
def index():
    css_ = url_for('static', filename='style.css')
    return render_template("index.html", css_path=css_)

#analyze imported image
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No filename provided"}), 400

        # Read image file into OpenCV format
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return jsonify({"error": "Invalid image format"}), 400

        #resize to model size (assuming your model uses 300x300)
        img_resized = cv2.resize(img, (300, 300))
        img_arr = img_resized.astype("float32") / 255.0
        img_arr = np.expand_dims(img_arr, axis=(0, -1))  # (1, 300, 300, 1)

        #prediction
        preds = model.predict(img_arr)
        
        #convert predictions to JSON-friendly output
        preds_list = preds[0].tolist()

        response = {
            "prediction_raw": preds_list,
            "curve_score": float(preds_list[0]),
            "cross_score": float(preds_list[1]),
            "lift_score": float(preds_list[2])
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

#run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=49164, debug=True)
