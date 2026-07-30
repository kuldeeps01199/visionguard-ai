import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input

def create_dummy_model():
    print("Generating a lightweight dummy model for UI testing...")
    
    # Create a very simple sequential model that expects 32x32x3 input
    model = Sequential([
        Input(shape=(32, 32, 3)),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid') # Binary classification output
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Save the model
    model_path = 'cifake_model.keras'
    model.save(model_path)
    
    print(f"Dummy model saved to {model_path}.")
    print("You can now run 'python app.py' to test the web interface!")

if __name__ == '__main__':
    create_dummy_model()
