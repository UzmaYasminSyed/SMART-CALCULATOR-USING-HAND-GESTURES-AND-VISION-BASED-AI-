import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os

dataset_path = "Dataset"
img_size = 64
batch_size = 32

# Data loading
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
)

train_gen = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    subset='validation'
)

num_classes = len(train_gen.class_indices)
print("Classes:", train_gen.class_indices)

# Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(train_gen, validation_data=val_gen, epochs=30)

# Save model + class map
model.save("sign_classifier.h5")

import json
with open("classes.json", "w") as f:
    json.dump(train_gen.class_indices, f)

print("Training complete, model saved as sign_classifier.h5")








import cv2
import numpy as np
import tensorflow as tf
import json

img_size = 64

# Load model
model = tf.keras.models.load_model("sign_classifier.h5")

# Load class labels
with open("classes.json", "r") as f:
    class_indices = json.load(f)

# reverse dict → index to class name
idx_to_class = {v: k for k, v in class_indices.items()}

cap = cv2.VideoCapture(0)

print("Press 'C' to capture | 'Q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('c'):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=(0, -1))  # shape: (1, 64, 64, 1)

        pred = model.predict(img)[0]
        index = np.argmax(pred)
        class_name = idx_to_class[index]
        confidence = pred[index]

        print(f"Prediction: {class_name} (confidence={confidence:.2f})")






###

import cv2
import os
from PIL import Image
import imagehash

dataset_path = "Dataset"

# --------------------------------------------
# LOAD DATASET HASHES
# --------------------------------------------

dataset_hashes = []   # list of (hash, class_name)

print("Loading dataset...")

for class_name in os.listdir(dataset_path):
    class_folder = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_folder):
        continue

    for file in os.listdir(class_folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):

            img_path = os.path.join(class_folder, file)

            img = Image.open(img_path).convert("L")  # grayscale
            h = imagehash.phash(img)

            dataset_hashes.append((h, class_name))

print("Loaded", len(dataset_hashes), "images hashed.")


# --------------------------------------------
# CAMERA: press C to capture then compare
# --------------------------------------------

cap = cv2.VideoCapture(0)
print("Press C to capture | Q to exit")

while True:

    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('c'):
        print("Capturing image...")

        # Save captured image as temp
        cv2.imwrite("capture.jpg", frame)

        # Convert to grayscale and hash
        img = Image.open("capture.jpg").convert("L")
        captured_hash = imagehash.phash(img)

        # Compare with dataset
        best_distance = 9999
        best_class = "Unknown"

        for h, class_name in dataset_hashes:
            dist = abs(captured_hash - h)
            if dist < best_distance:
                best_distance = dist
                best_class = class_name

        print("Most similar class:", best_class)
        print("Hash difference:", best_distance)
        print("---------------------------------")

cap.release()
cv2.destroyAllWindows()


##### 3


import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # mirror view
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # -------------------------------------------
    # SKIN COLOR RANGE (HSV)
    # Adjust if needed for your lighting
    # -------------------------------------------
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Clean noise
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = cv2.medianBlur(mask, 5)

    # Morphology to improve mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Create result: hand visible, background black
    hand_only = cv2.bitwise_and(frame, frame, mask=mask)

    # OPTIONAL: display mask
    cv2.imshow("Mask", mask)
    cv2.imshow("Hand Only (BG Black)", hand_only)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


