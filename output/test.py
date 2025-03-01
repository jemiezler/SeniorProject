import joblib
import numpy as np

# ✅ Load trained model
model = joblib.load("best_ransacregressor.pkl")  # Change filename if needed

# ✅ Check the feature names (columns the model expects)
if hasattr(model, "feature_names_in_"):
    print("✅ Model feature names:", model.feature_names_in_)
    print("✅ Expected input shape:", (1, len(model.feature_names_in_)))
else:
    print("⚠️ Model does not store `feature_names_in_`. Check training data.")

# ✅ Check model coefficients shape (if applicable)
if hasattr(model, "coef_"):
    print("✅ Model coefficient shape:", model.coef_.shape)
else:
    print("⚠️ Model does not have coefficients (e.g., RANSACRegressor might not).")
