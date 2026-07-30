import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================================
# ACTUAL TRAINING SCRIPT FOR CIFAKE DATASET
# ==========================================================
# 
# Dataset Link: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
# 
# Instructions:
# 1. Download the dataset and extract it.
# 2. Place the 'train' and 'test' folders in a directory named 'dataset'.
#    Example structure:
#    - dataset/
#        - train/
#            - REAL/
#            - FAKE/
#        - test/
#            - REAL/
#            - FAKE/
# 3. Run this script: python train.py
# 4. It will produce 'cifake_model.keras' which you can use with app.py
# ==========================================================

# Hyperparameters
IMG_SIZE = (32, 32) # CIFAKE image size is 32x32
BATCH_SIZE = 64
EPOCHS = 10
DATASET_DIR = "dataset"

def build_model():
    # Load MobileNetV2 without the top classification layer
    base_model = MobileNetV2(
        input_shape=(32, 32, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model to prevent destroying pre-trained weights during early training
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(1, activation='sigmoid')(x) # Binary classification (0: FAKE, 1: REAL)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
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

    # Data Augmentation & Loading
    # CIFAKE is already quite large, but slight augmentation helps
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        rotation_range=10
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['FAKE', 'REAL'] # Enforces FAKE=0, REAL=1
    )

    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['FAKE', 'REAL']
    )

    model = build_model()
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(
        "cifake_model.keras", 
        monitor='val_accuracy', 
        save_best_only=True, 
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_accuracy', 
        patience=3, 
        restore_best_weights=True
    )

    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stopping]
    )
    
    print("Training complete! The best model has been saved as 'cifake_model.keras'.")

if __name__ == '__main__':
    main()
