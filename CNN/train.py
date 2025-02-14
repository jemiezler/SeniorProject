import os
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Set image parameters
IMAGE_SIZE = (996, 1120)  # Resize images to 128x128
IMAGE_DIR = "images/"  # Folder containing images
LABELS_CSV = "data.csv"  # CSV file with "filename" and "weight" columns
# Load dataset labels
df = pd.read_csv(LABELS_CSV)

# Load images and corresponding weights
X = []
y = []

for index, row in df.iterrows():
    img_path = os.path.join(IMAGE_DIR, row['filename'])
    img = cv2.imread(img_path)

    if img is not None:
        img = cv2.resize(img, IMAGE_SIZE)  # Resize image
        X.append(img)
        y.append(row['weight'])

X = np.array(X)
y = np.array(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer for weight regression
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Save the trained model
model.save("cnn_weight_prediction.h5")

# Function to predict weight for a folder of images
def predict_weights_from_folder(folder_path):
    predictions = {}

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE)  # Resize
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            predicted_weight = model.predict(img)[0][0]
            predictions[filename] = predicted_weight

    return predictions

# Example usage: Predict weights for all images in a folder
predictions = predict_weights_from_folder("test_images/")
for filename, weight in predictions.items():
    print(f"Image: {filename} → Predicted Weight: {weight:.2f} kg")