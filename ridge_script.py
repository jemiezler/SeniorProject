import itertools
import pandas as pd
import os
import cv2
import numpy as np
import skimage.feature as skf
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# For the progress bars
from tqdm import tqdm

# ======================
# 1) Define selected feature groups
# ======================

df_color = pd.read_csv('resources/color_data.csv')
df_texture = pd.read_csv('resources/texture_data.csv')
df_weight = pd.read_csv('resources/weight_loss_data.csv')
df_color.sort_values(by='Filename', inplace=True, ignore_index=True)
df_texture.sort_values(by='Filename', inplace=True, ignore_index=True)
df_weight.sort_values(by='Filename', inplace=True, ignore_index=True)
df = pd.merge(df_weight, df_color, on='Filename')
df = pd.merge(df, df_texture, on='Filename')

    
df[['Day', 'Temp', 'Rep']] = df['Filename'].str.extract(r'(\d+)_(\d+)_(\d+)')
df[['Day', 'Temp', 'Rep']] = df[['Day', 'Temp', 'Rep']].astype(float).astype('Int64')
df["Yellow"] = df["Mean_RGB_R"] + df["Mean_RGB_G"]
df["Cyan"] = df["Mean_RGB_G"] + df["Mean_RGB_B"]
df["Magenta"] = df["Mean_RGB_R"] + df["Mean_RGB_B"]
df["Brightness"] = (df["Mean_RGB_R"] + df["Mean_RGB_G"] + df["Mean_RGB_B"]) / 3
df["Chroma"] = df[["Mean_RGB_R", "Mean_RGB_G", "Mean_RGB_B"]].max(axis=1) - df[["Mean_RGB_R", "Mean_RGB_G", "Mean_RGB_B"]].min(axis=1)


df.drop(columns=['Filename'], inplace=True)
features = {
    "HSV": [
        "Mean_HSV_H", "Std_HSV_H", "Mean_HSV_S", "Std_HSV_S", "Mean_HSV_V", "Std_HSV_V"
    ],
    "GLCM": [
        "GLCM_contrast", "GLCM_dissimilarity", "GLCM_homogeneity", 
        "GLCM_energy", "GLCM_correlation"
    ],
    "LBP": [
        "LBP_0", "LBP_1", "LBP_2", "LBP_3", "LBP_4", "LBP_5", "LBP_6", "LBP_7", "LBP_8", "LBP_9"
    ],
    "Temp": ["Temp"],
    "Yellow": ["Yellow"],
    "Cyan": ["Cyan"],
    "Chroma": ["Chroma"],
}

# A helper function for metrics
def calc_metrics(y_true, y_pred):
    return {
        "R2": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
    }

# 3) Your DataFrame 'df' must have all columns from `features` plus this target column
# df = pd.read_csv("your_data.csv")
target_column = "%_Weight_Loss"  # make sure this column exists

# Create a directory to store saved models
model_dir = "output/all/saved_models"
os.makedirs(model_dir, exist_ok=True)

# -----------------------------------------------------------
# 4) Train only on selected feature groups
# -----------------------------------------------------------
selected_cols = []
for k in features.keys():
    selected_cols.extend(features[k])

# X = selected columns, y = target
X = df[selected_cols]
y = df[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

# Define the feature group string
combo_str = "+".join(features.keys())
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Ridge model
model = Ridge()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test)

metric_values = calc_metrics(y_test, y_pred)

results = [{
    "Feature Groups": combo_str,
    "Model": "Ridge",
    "R2": metric_values["R2"],
    "MSE": metric_values["MSE"],
}]

# Save model
model_filename = f"{model_dir}/model_Ridge_{combo_str}.joblib"
dump(model, model_filename)

# ------------------------------------------------
# 5) Convert results to DataFrame & Save to CSV
# ------------------------------------------------
results_df = pd.DataFrame(results)
results_df = results_df[["Feature Groups", "Model", "R2", "MSE"]]

print("\nComplete Results:\n", results_df.head(), " ...\n")  # Show results preview

csv_filename = "output/feature_results.csv"
results_df.to_csv(csv_filename, index=False)
print(f"\nSaved to '{csv_filename}'")

# ------------------------------------------------
# 6) Load and Test with Specific Picture (Extracting Real Features)
# ------------------------------------------------
def extract_features_from_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_hsv = hsv.mean(axis=(0, 1))
    std_hsv = hsv.std(axis=(0, 1))
    
    # Convert to grayscale for GLCM and LBP
    gray = rgb2gray(image)
    gray = (gray * 255).astype(np.uint8)
    
    # Compute GLCM Features
    glcm = graycomatrix((gray * 255).astype(np.uint8), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    glcm_features = [
        graycoprops(glcm, 'contrast')[0, 0],
        graycoprops(glcm, 'dissimilarity')[0, 0],
        graycoprops(glcm, 'homogeneity')[0, 0],
        graycoprops(glcm, 'energy')[0, 0],
        graycoprops(glcm, 'correlation')[0, 0]
    ]
    
    # Compute LBP Features
    lbp = local_binary_pattern(gray, P=10, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(11), range=(0, 10))
    lbp_features = lbp_hist.tolist()
    
    # Extract additional features
    temp = [36.5]  # Replace with actual temperature extraction if applicable
    yellow = [np.mean(image[:, :, 2] - image[:, :, 1])]  # Difference between R and G
    cyan = [np.mean(image[:, :, 1] - image[:, :, 0])]  # Difference between G and B
    chroma = [np.std(image)]  # Standard deviation as a rough chroma measure
    
    # Combine features
    feature_vector = list(mean_hsv) + list(std_hsv) + glcm_features + lbp_features + temp + yellow + cyan + chroma
    return feature_vector

# Load trained model
model = load(model_filename)

# Test with a specific image
image_path = "resources/images/4_20_2.png"  # Change to actual image path
test_features = extract_features_from_image(image_path)
# Convert extracted features to DataFrame with correct column names
test_features_df = pd.DataFrame([test_features], columns=X_train.columns)  # ✅ Ensures correct feature names

# Make prediction
predicted_weight_loss = model.predict(test_features_df)[0]


print(f"Predicted % Weight Loss for {image_path}: {predicted_weight_loss:.2f}")