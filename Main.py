# gesture_calculator.py
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -------------------------------
# 1. Dataset and Preprocessing
# -------------------------------

DATASET_DIR = 'Dataset/'  # path to your 14-class dataset
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 100
MODEL_PATH = 'gesture_calculator_model.h5'

# Image augmentation and train-validation split
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_generator.class_indices)
class_names = list(train_generator.class_indices.keys())

# -------------------------------
# 2. Build CNN Model
# -------------------------------

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE,3)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# -------------------------------
# 3. Train the Model
# -------------------------------
print("Starting training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# Save model
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")



from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = 'Dataset/'  # your dataset folder path here
IMG_SIZE = 64
BATCH_SIZE = 32

datagen = ImageDataGenerator(rescale=1./255)

generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Class indices found in dataset folder:")
print(generator.class_indices)









