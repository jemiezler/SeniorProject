from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from fastapi.responses import StreamingResponse
import logging
import tensorflow_hub as hub
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Load the trained DeepLabV3+ model
MODEL_PATH = "../models/deeplabv3_trained.keras"
try:
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={'KerasLayer': hub.KerasLayer})
    logger.info(f"✅ Model successfully loaded from {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

# Function to preprocess the image
def preprocess_image(image: Image.Image, target_size=(512, 512)):
    image = image.resize(target_size)  # Resize image to model input size
    image = np.array(image) / 255.0  # Normalize pixel values to [0,1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Endpoint for segmentation
@app.post("/segment/")
async def segment_image(file: UploadFile = File(...)):
    try:
        # Read image from upload
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        original_size = image.size  # Save original image size
        logger.info(f"📸 Received image {file.filename} for segmentation")

        # Preprocess the image
        input_data = preprocess_image(image, target_size=(512, 512))

        # Predict segmentation mask
        prediction = model.predict(input_data)[0]  # Extract first batch output

        # Convert output to class index map (for multi-class models)
        if prediction.shape[-1] > 1:  # Check if model outputs multiple channels (multi-class)
            mask = np.argmax(prediction, axis=-1)  # Convert to class index map
        else:
            mask = (prediction.squeeze() * 255).astype(np.uint8)  # Convert grayscale output

        # Convert mask to PIL image
        mask_image = Image.fromarray(mask)

        # Resize mask back to original image size
        mask_resized = mask_image.resize(original_size, Image.NEAREST)

        # Convert mask to PNG bytes
        img_bytes = io.BytesIO()
        mask_resized.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        logger.info(f"✅ Segmentation completed for {file.filename}")

        # Return the segmentation mask as a PNG image response
        return StreamingResponse(img_bytes, media_type="image/png")

    except Exception as e:
        logger.error(f"❌ Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)