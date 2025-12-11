from flask import Flask, render_template, url_for, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

#load the model
MODEL_PATH = "saved_model/model.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")

#preprocess images
def preprocess_image(file, target_size=(128,128)):
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    #clean image
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #resize image
    h, w = img.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h))

    canvas = np.full(target_size, 255, dtype=np.uint8)
    top = (128 - new_h) // 2
    left = (128 - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = img_resized

    #normalize image
    canvas_norm = canvas.astype("float32") / 255.0

    #shape of image
    return np.expand_dims(np.expand_dims(canvas_norm, axis=-1), axis=0)

#index
@app.route('/')
@app.route('/index.html')
def index():
    css_= url_for('static', filename='style.css')
    return render_template("index.html", css_path=css_)

#analyze
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        #preprocess model
        input_tensor = preprocess_image(file)

        #prediction
        pred = model.predict(input_tensor, verbose=0)[0]

        curve = float(pred[..., 0])
        lift = float(pred[..., 1])
        cross = float(pred[..., 2])

        response = {
            "curve_score": curve,
            "lift_score": lift,
            "cross_score": cross
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

#running app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=49164, debug=True)
