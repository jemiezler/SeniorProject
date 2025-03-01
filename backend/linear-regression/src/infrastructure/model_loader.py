from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

class ModelLoader:
    def __init__(self, model_path: str, scaler_path: str = None):
        self.model_path = model_path
        self.model = None
        self.scaler = None  # Store the scaler
        self.expected_features = None

        if scaler_path:
            self.scaler = joblib.load(scaler_path)  # Load the same scaler used in training

    def load(self):
        """Load model and define expected feature names."""
        self.model = joblib.load(self.model_path)

        if hasattr(self.model, "feature_names_in_"):
            self.expected_features = list(self.model.feature_names_in_)
        else:
            print("⚠️ Model does not have `feature_names_in_`. Define manually.")
            self.expected_features = [
                'Mean_RGB_R', 'Std_RGB_R', 'Mean_RGB_G', 'Std_RGB_G', 'Mean_RGB_B', 'Std_RGB_B',
                'GLCM_contrast', 'GLCM_dissimilarity', 'GLCM_homogeneity', 'GLCM_energy', 'GLCM_correlation',
                'LBP_0', 'LBP_1', 'LBP_2', 'LBP_3', 'LBP_4', 'LBP_5', 'LBP_6', 'LBP_7', 'Temp', 'Cyan'
            ]
            self.model.feature_names_in_ = self.expected_features  # Manually define feature names

    def predict(self, features: pd.DataFrame):
        """Predict while applying scaling if necessary."""
        if self.scaler:
            features = self.scaler.transform(features)  # ✅ Scale features before prediction
        return self.model.predict(features)[0]
