import joblib
import pandas as pd

class ModelLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.expected_features = None

    def load(self):
        """Load the model and define expected feature names."""
        self.model = joblib.load(self.model_path)

        # Extract feature names from the model if available
        if hasattr(self.model, "feature_names_in_"):
            self.expected_features = list(self.model.feature_names_in_)
        else:
            print("⚠️ Model does not have `feature_names_in_`. Defining manually.")
            # self.expected_features = [
            #     # HSV features
            #     'Mean_HSV_H', 'Mean_HSV_S', 'Mean_HSV_V', 'Std_HSV_H', 'Std_HSV_S', 'Std_HSV_V'
                
            #     # GLCM features
            #     'GLCM_contrast', 'GLCM_dissimilarity', 'GLCM_homogeneity', 'GLCM_energy', 'GLCM_correlation',

            #     # LBP features
            #     'LBP_0', 'LBP_1', 'LBP_2', 'LBP_3', 'LBP_4', 'LBP_5', 'LBP_6', 'LBP_7',

            #     # Other color and temperature features
            #     'Temp', 'Yellow', 'Cyan', 'Chroma'
            # ]
            self.model.feature_names_in_ = self.expected_features

    def predict(self, features: pd.DataFrame):
        """Predict without scaling."""
        if self.model is None:
            raise ValueError("Model is not loaded. Call `load()` before prediction.")

        # Ensure the input features match the expected ones
        if self.expected_features:
            features = features[self.expected_features]

        return self.model.predict(features)[0]
