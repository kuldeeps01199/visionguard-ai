import tensorflow as tf
import tf2onnx
import onnx

# Load Keras model
model = tf.keras.models.load_model('cifake_model.keras')

# Define input signature based on what the model expects
spec = (tf.TensorSpec((None, 32, 32, 3), tf.float32, name="input"),)

# Convert and save
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path="cifake_model.onnx")
print("Successfully converted to cifake_model.onnx")
