import os
#makes server run with only CPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#hides warnings from tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

from flask import Flask, render_template, url_for, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import cv2
import numpy as np

def count_components(mask):
    mask_uint8 = (mask > 0.5).astype("uint8")
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask_uint8)
    return int(num_labels - 1)

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

#load the model
MODEL_PATH = "saved_model/model.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully.")
print(model.output_shape)

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

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files["file"]

        # preprocess
        input_tensor = preprocess_image(file)
        # Get the prediction list (e.g., [curve_output, lift_output, cross_output])
        predictions = model.predict(input_tensor, verbose=0)
        
        # Assume the order is: 0=curve, 1=lift, 2=cross. 
        #get single mask (128, 128) from each output (1, 128, 128, 1)
        curve_mask = predictions[0][0, ..., 0]  
        lift_mask  = predictions[1][0, ..., 0]  
        cross_mask = predictions[2][0, ..., 0]  

        # count connected components
        curve_count = count_components(curve_mask)
        lift_count  = count_components(lift_mask)
        cross_count = count_components(cross_mask)

        # final complexity score
        complexity_score = curve_count + lift_count + cross_count

        return jsonify({
            "score": int(complexity_score),
            "lifts": int(lift_count),
            "reversals": int(curve_count),
            "intersections": int(cross_count)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



#running app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=49164, debug=True)
