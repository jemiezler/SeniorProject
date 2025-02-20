import numpy as np
import tensorflow as tf

class PredictionService:
    def __init__(self, model_path: str):
        """
        Initializes the Prediction Service with the TensorFlow model.
        """
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, features: list) -> list:
        """
        Runs inference on input features.
        """
        input_data = np.array(features).reshape(1, -1)  # Reshape for model input
        prediction = self.model.predict(input_data)
        return prediction.flatten().tolist()
