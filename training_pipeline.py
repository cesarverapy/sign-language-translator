from dotenv import load_dotenv
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

load_dotenv()

# Configuration settings
TRAIN_DIR = os.getenv("TRAIN_DIR")
VAL_DIR = os.getenv("VAL_DIR")
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 30
MODEL_PATH = os.getenv("MODEL_PATH")
NUM_CLASSES = len(os.listdir(TRAIN_DIR))  # Assumes one folder per class in TRAIN_DIR

def load_data():
    """Loads and preprocesses data with augmentation for training and validation."""
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return train_generator, val_generator

def build_model():
    """Builds the MobileNetV2 model with added custom layers for gesture classification."""
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    
    # Freeze base layers for initial training
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_generator, val_generator):
    """Trains the model with checkpointing and early stopping."""
    callbacks = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss', verbose=1),
        EarlyStopping(patience=5, monitor='val_loss', verbose=1)
    ]

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    return history

def fine_tune_model(model, train_generator, val_generator):
    """Unfreezes last few layers of base model and fine-tunes with a lower learning rate."""
    for layer in model.layers[-10:]:  # Unfreeze last few layers for fine-tuning
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate for fine-tuning
                  loss='categorical_crossentropy', metrics=['accuracy'])

    fine_tuning_history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        verbose=1
    )
    return fine_tuning_history

def main():
    # Load data
    train_generator, val_generator = load_data()

    # Build and train the model
    model = build_model()
    print("Starting initial training...")
    train_model(model, train_generator, val_generator)

    # Fine-tune the model
    print("Starting fine-tuning...")
    fine_tune_model(model, train_generator, val_generator)

    # Save the final model
    model.save(MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    main()
