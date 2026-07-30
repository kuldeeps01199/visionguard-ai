import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tf2onnx

# ==========================================================
# HIGH-ACCURACY TRAINING SCRIPT FOR CIFAKE (32x32)
# ==========================================================

IMG_SIZE = (32, 32)
BATCH_SIZE = 64
EPOCHS = 15
DATASET_DIR = "dataset"

def build_model():
    # A Custom CNN is much better for 32x32 images than a frozen MobileNetV2
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid') # Binary classification
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    train_dir = os.path.join(DATASET_DIR, "train")
    test_dir = os.path.join(DATASET_DIR, "test")
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"Error: Dataset not found in {DATASET_DIR}. Please download CIFAKE and place it here.")
        return

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=10
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading Dataset...")
    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', classes=['FAKE', 'REAL']
    )

    validation_generator = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', classes=['FAKE', 'REAL']
    )

    model = build_model()
    model.summary()

    checkpoint = ModelCheckpoint("cifake_model.keras", monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)

    print("Starting training...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping]
    )
    
    print("\nTraining complete! Loading best model for ONNX conversion...")
    best_model = tf.keras.models.load_model("cifake_model.keras")
    
    print("Converting to ONNX format...")
    spec = (tf.TensorSpec((None, 32, 32, 3), tf.float32, name="input"),)
    tf2onnx.convert.from_keras(best_model, input_signature=spec, opset=13, output_path="cifake_model.onnx")
    
    print("\nSUCCESS! New high-accuracy 'cifake_model.onnx' has been generated and is ready to deploy!")

if __name__ == '__main__':
    main()
