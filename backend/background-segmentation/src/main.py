from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load trained model
MODEL_PATH = "models/deeplabv3plus_kale.h5"
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = (512, 512)

@app.post("/segment")
async def segment_image(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preprocess
    original_size = image.shape[:2]
    image_resized = cv2.resize(image, IMG_SIZE) / 255.0
    input_tensor = tf.convert_to_tensor(image_resized, dtype=tf.float32)[tf.newaxis, ...]

    # Predict segmentation
    predicted_mask = model.predict(input_tensor)[0]
    binary_mask = np.argmax(predicted_mask, axis=-1).astype(np.uint8)
    binary_mask = cv2.resize(binary_mask, (original_size[1], original_size[0]))

    # Return leaf count
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leaf_count = len(contours)

    return {"leaf_count": leaf_count}

# Run API
# Command: uvicorn main:app --host 0.0.0.0 --port 8000
