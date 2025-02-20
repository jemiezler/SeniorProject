from fastapi import FastAPI
import tensorflow as tf
import numpy as np
from pydantic import BaseModel

# Load the exported model
MODEL_PATH = "lstm"
model = tf.saved_model.load(MODEL_PATH)

# Get the serving function
predict_fn = model.signatures["serving_default"]

# Define FastAPI
app = FastAPI()

# Define request body format
class PredictionInput(BaseModel):
    input_data: list

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        # Convert input data to a TensorFlow tensor
        input_tensor = tf.convert_to_tensor([input_data.input_data], dtype=tf.float32)

        # Run the model prediction
        prediction = predict_fn(input_tensor)
        
        # Get the output tensor name (dynamic depending on model)
        output_key = list(prediction.keys())[0]
        result = prediction[output_key].numpy().tolist()

        return {"prediction": result}
    
    except Exception as e:
        return {"error": str(e)}

# Run the server if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
