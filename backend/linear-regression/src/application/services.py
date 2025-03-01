from domain.feature_extractor import FeatureExtractor
from infrastructure.model_loader import ModelLoader
from infrastructure.image_loader import ImageLoader
import pandas as pd
import numpy as np

# Load the trained model
model_loader = ModelLoader("models/model.pkl")
model_loader.load()

class AnalysisService:
    """Processes images to extract features and predict."""

    @staticmethod
    def analyze_image(image_bytes: bytes, temp: float):
        """Load image → Extract features → Scale features → Predict."""
        image = ImageLoader.load(image_bytes)
        features = FeatureExtractor.extract_all_features(image, temp)

        # ✅ Convert NumPy types to standard Python types
        features = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v for k, v in features.items()}

        # ✅ Convert to DataFrame
        features_df = pd.DataFrame([features])

        # ✅ Ensure correct column order
        if model_loader.expected_features is None:
            raise ValueError("❌ Model `expected_features` is not defined.")

        missing_features = [f for f in model_loader.expected_features if f not in features_df.columns]
        if missing_features:
            raise ValueError(f"❌ Missing required features: {missing_features}")

        filtered_features = features_df[model_loader.expected_features]

        # ✅ Apply the same scaler that was used in training
        prediction = model_loader.predict(filtered_features)

        return {
            "prediction": float(prediction),  # Convert to float to ensure proper JSON serialization
            "features": features  # Already converted to Python types
        }
