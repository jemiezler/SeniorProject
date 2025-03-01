import joblib

# Load model
model = joblib.load("model.pkl")

# Print feature names
if hasattr(model, "feature_names_in_"):
    print("✅ Model was trained with these features:", model.feature_names_in_)
else:
    print("⚠️ Model does NOT have feature names. Check training data.")
