from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
import cv2
import io
from fastapi.responses import StreamingResponse
import logging
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

def resize_with_padding(image, target_size):
    """Resize image while maintaining aspect ratio with padding."""
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Compute padding
    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    padded = cv2.copyMakeBorder(resized, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded

# Load the trained DeepLabV3+ model
MODEL_PATH = "models/deeplabv3_trained.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"✅ Model successfully loaded from {MODEL_PATH}")
    model.summary()
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

# Function to preprocess the image
def preprocess_image(image: np.ndarray, target_size=(512, 512)):
    image = resize_with_padding(image, target_size[0])  # Resize with padding
    image = image.astype("float32") / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to remove background
def remove_background(image: np.ndarray, mask: np.ndarray):
    """Apply segmentation mask to remove background."""
    image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)  # Convert to RGBA
    image[:, :, 3] = mask  # Apply mask as alpha channel
    return image

# Endpoint for segmentation and background removal
@app.post("/segment/")
async def segment_image(file: UploadFile = File(...)):
    try:
        # Read image from upload
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)  # Read as OpenCV image
        
        if image is None:
            raise ValueError("Invalid image format")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        original_size = (image.shape[1], image.shape[0])  # Save original size
        logger.info(f"📸 Received image {file.filename} for segmentation")

        # Preprocess the image
        input_data = preprocess_image(image, target_size=(512, 512))

        # Predict segmentation mask
        prediction = model.predict(input_data)[0]  # Extract first batch output

        # Convert output to class index map (multi-class segmentation)
        predicted_mask = np.argmax(prediction, axis=-1).squeeze()
        
        # Debugging info
        logger.info(f"Prediction shape: {prediction.shape}, min: {prediction.min()}, max: {prediction.max()}")

        # Convert mask to uint8 image (match inference script)
        mask = (predicted_mask * 255).astype(np.uint8)
        mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

        # Remove background using mask
        segmented_image = remove_background(image, mask_resized)

        # Convert image to PNG bytes
        _, img_encoded = cv2.imencode(".png", segmented_image)
        img_bytes = io.BytesIO(img_encoded.tobytes())

        logger.info(f"✅ Segmentation and background removal completed for {file.filename}")

        # Return the segmented image with background removed
        return StreamingResponse(img_bytes, media_type="image/png")

    except Exception as e:
        logger.error(f"❌ Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
