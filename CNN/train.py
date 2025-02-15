import os
import numpy as np
import cv2
import pandas as pd
import tensorflow as tf
import logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set image parameters
IMAGE_SIZE = (996, 1120)  # Resize images
IMAGE_DIR = "images/"  # Folder containing images
LABELS_CSV = "data.csv"  # CSV file with "Label" (filename) and "Weight" columns

# Load dataset labels
logger.info("Loading dataset labels from CSV file...")
df = pd.read_csv(LABELS_CSV)

# Load images and corresponding weights
X = []
y = []

logger.info("Loading and preprocessing images...")
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Images"):
    img_path = os.path.join(IMAGE_DIR, row['Label'])
    img = cv2.imread(img_path)

    if img is not None:
        img = cv2.resize(img, IMAGE_SIZE)  # Resize image
        img = img / 255.0  # Normalize pixel values (0-1)
        X.append(img)
        y.append(row['Weight'])
    else:
        logger.warning(f"Failed to load image: {img_path}")

X = np.array(X)
y = np.array(y)

# Split dataset into training and testing sets (80% train, 20% test)
logger.info("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple CNN model
logger.info("Building CNN model...")
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
logger.info("Compiling the model...")
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
logger.info("Training the model...")
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
logger.info("Evaluating the model...")
loss, mae = model.evaluate(X_test, y_test)
logger.info(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

# Save the trained model
logger.info("Saving the trained model in Keras format...")
model.save("cnn_weight_prediction.keras")  # Use .keras instead of .h5

# Function to predict weight for all images in the dataset (no separate folder needed)
def predict_weights():
    predictions = {}
    
    logger.info(f"Predicting weights for {len(df)} images...")

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Predicting Weights", unit="img"):
        img_path = os.path.join(IMAGE_DIR, row['Label'])
        img = cv2.imread(img_path)

        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE)  # Resize
            img = img / 255.0  # Normalize
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            predicted_weight = model.predict(img)[0][0]
            predictions[row['Label']] = predicted_weight
        else:
            logger.warning(f"Failed to load image: {img_path}")

        percent_complete = (i + 1) / len(df) * 100
        logger.info(f"Progress: {percent_complete:.2f}%")
    
    return predictions

# Predict weights for all images in the dataset
predictions = predict_weights()
for filename, weight in predictions.items():
    logger.info(f"Image: {filename} → Predicted Weight: {weight:.2f} kg")
