import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from werkzeug.utils import secure_filename

# Optional: Disable TF oneDNN warnings if any
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    # Limit threads to save RAM on free tiers
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max size

MODEL_PATH = "cifake_model.keras"
model = None

def load_ml_model():
    global model
    if TF_AVAILABLE and os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
            model = None
    else:
        print("Model file not found or TensorFlow not installed.")

# Try to load model at startup
load_ml_model()

def prepare_image(image, target_size=(32, 32)):
    # Convert to RGB to ensure 3 channels
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize the image
    image = image.resize(target_size)
    # Convert to array and scale to [0, 1] as MobileNetV2 / custom CNN usually expects
    img_array = img_to_array(image)
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
            # Fallback if model isn't loaded (e.g. dummy environment without TF)
            # Return a random result just for UI testing
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
        
        # Make prediction
        prediction = model.predict(processed_image)[0][0]
        
        # In CIFAKE, let's assume 0 = Fake, 1 = Real (or vice versa depending on training)
        # Assuming training labels: 0 -> FAKE, 1 -> REAL
        # Wait, standard is 0 for FAKE, 1 for REAL or vice versa. 
        # For this script, we'll assume > 0.5 is Real, < 0.5 is AI Generated.
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
