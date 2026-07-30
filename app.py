import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from werkzeug.utils import secure_filename

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max size

MODEL_PATH = "cifake_model.onnx"
model = None

def load_ml_model():
    global model
    if ONNX_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            model = ort.InferenceSession(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print("Model file not found or onnxruntime not installed.")

# Try to load model at startup
load_ml_model()

def prepare_image(image, target_size=(32, 32)):
    # Convert to RGB to ensure 3 channels
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize the image
    image = image.resize(target_size)
    # Convert to array and scale to [0, 1]
    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0
    # Expand dimensions to create a batch of 1
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        if model is None:
            # Fallback if model isn't loaded
            import random
            is_fake = random.choice([True, False])
            confidence = round(random.uniform(50.0, 99.9), 2)
            result = "AI Generated" if is_fake else "Real Image"
            return jsonify({
                'prediction': result,
                'confidence': confidence,
                'warning': 'Model not loaded. Using random simulation.'
            })

        # Preprocess and predict
        processed_image = prepare_image(image, target_size=(32, 32))
        
        # Make prediction using ONNX
        input_name = model.get_inputs()[0].name
        output = model.run(None, {input_name: processed_image})
        prediction = output[0][0][0]
        
        # Binary classification logic
        if prediction > 0.5:
            result = "Real Image"
            confidence = prediction * 100
        else:
            result = "AI Generated"
            confidence = (1 - prediction) * 100

        return jsonify({
            'prediction': result,
            'confidence': round(float(confidence), 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
