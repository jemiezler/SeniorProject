import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
from itertools import combinations
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Data
df = pd.read_csv('weight_color_data.csv')

# Handle missing values
df.dropna(inplace=True)

df[['Day', 'Temp', 'Rep']] = df['Label'].str.split('_', expand=True)
df['Day'] = df['Day'].astype(int)
df['Temp'] = df['Temp'].astype(int)
df['Rep'] = df['Rep'].astype(int)
df.drop(columns=['Label'], inplace=True)



# Define Feature Sets
base_features = {
    "L": ["L_Mean", "L_Std"],
    "a": ["a_Mean", "a_Std"],
    "b": ["b_Mean", "b_Std"],
    "H": ["H_Mean", "H_Std"],
    "S": ["S_Mean", "S_Std"],
    "V": ["V_Mean", "V_Std"],
    "R": ["R_Mean", "R_Std"],
    "G": ["G_Mean", "G_Std"],
    "B": ["B_Mean", "B_Std"],
    "Day": ["Day"],
    "Temp": ["Temp"]
}

def progressive_features(features_dict):
    feature_combinations = {}
    feature_groups = list(features_dict.keys())
    
    for i in range(1, len(feature_groups) + 1):
        for comb in combinations(feature_groups, i):
            combined_columns = sum([features_dict[key] for key in comb], [])
            feature_combinations[" ".join(comb)] = combined_columns
    
    logging.info(f"Generated {len(feature_combinations)} feature combinations.")
    return feature_combinations

feature_sets = progressive_features(base_features)

def mix_orange(df, red_weight=0.8, green_weight=0.2):
    """
    Create a new feature that represents a mixed color (Orange) from Red and Green.
    
    This function computes both:
    - Mixed color (Mean values)
    - Mixed color (Standard Deviation)
    - Converts the mixed color to LAB and HSV
    """
    # Compute the mixed color mean
    df["Mixed_RedGreen_Mean"] = (df["R_Mean"] * red_weight + df["G_Mean"] * green_weight).astype(int)

    # Compute the mixed color standard deviation
    df["Mixed_RedGreen_Std"] = (df["R_Std"] * red_weight + df["G_Std"] * green_weight).astype(int)

    # Convert Mixed RGB to LAB & HSV
    mixed_rgb = np.stack([df["Mixed_RedGreen_Mean"], df["Mixed_RedGreen_Mean"], np.zeros_like(df["Mixed_RedGreen_Mean"])], axis=1)
    mixed_bgr = np.array(mixed_rgb, dtype=np.uint8)[:, np.newaxis, :]  # Convert to BGR format
    
    mixed_lab = cv2.cvtColor(mixed_bgr, cv2.COLOR_RGB2LAB)[:, 0, :]
    mixed_hsv = cv2.cvtColor(mixed_bgr, cv2.COLOR_RGB2HSV)[:, 0, :]

    # Add Mean Values
    df["Mixed_L_Mean"] = mixed_lab[:, 0]
    df["Mixed_a_Mean"] = mixed_lab[:, 1]
    df["Mixed_b_Mean"] = mixed_lab[:, 2]

    df["Mixed_H_Mean"] = mixed_hsv[:, 0]
    df["Mixed_S_Mean"] = mixed_hsv[:, 1]
    df["Mixed_V_Mean"] = mixed_hsv[:, 2]

    # Add Standard Deviation (Since we're mixing, we use an estimated std based on weighted input stds)
    df["Mixed_L_Std"] = (df["L_Std"] * red_weight + df["L_Std"] * green_weight).astype(int)
    df["Mixed_a_Std"] = (df["a_Std"] * red_weight + df["a_Std"] * green_weight).astype(int)
    df["Mixed_b_Std"] = (df["b_Std"] * red_weight + df["b_Std"] * green_weight).astype(int)

    df["Mixed_H_Std"] = (df["H_Std"] * red_weight + df["H_Std"] * green_weight).astype(int)
    df["Mixed_S_Std"] = (df["S_Std"] * red_weight + df["S_Std"] * green_weight).astype(int)
    df["Mixed_V_Std"] = (df["V_Std"] * red_weight + df["V_Std"] * green_weight).astype(int)

    return df

base_features["Orange"] = ["Mixed_RedGreen_Mean", "Mixed_RedGreen_Std"]
base_features["Orange_Lab"] = ["Mixed_L_Mean", "Mixed_a_Mean", "Mixed_b_Mean", "Mixed_L_Std", "Mixed_a_Std", "Mixed_b_Std"]
base_features["Orange_HSV"] = ["Mixed_H_Mean", "Mixed_S_Mean", "Mixed_V_Mean", "Mixed_H_Std", "Mixed_S_Std", "Mixed_V_Std"]


# In[22]:


def mix_yellow(df, red_weight=0.5, green_weight=0.5):
    """
    Create a new feature that represents a mixed color (Yellow) from Red and Green.
    
    This function computes:
    - Mixed color (Mean values)
    - Mixed color (Standard Deviation)
    - Converts the mixed color to LAB and HSV
    """
    # Compute the mixed color mean
    df["Mixed_RedGreenYellow_Mean"] = (df["R_Mean"] * red_weight + df["G_Mean"] * green_weight).astype(int)

    # Compute the mixed color standard deviation
    df["Mixed_RedGreenYellow_Std"] = (df["R_Std"] * red_weight + df["G_Std"] * green_weight).astype(int)

    # Convert Mixed RGB to LAB & HSV
    mixed_rgb = np.stack([df["Mixed_RedGreenYellow_Mean"], df["Mixed_RedGreenYellow_Mean"], np.zeros_like(df["Mixed_RedGreenYellow_Mean"])], axis=1)
    mixed_bgr = np.array(mixed_rgb, dtype=np.uint8)[:, np.newaxis, :]  # Convert to BGR format
    
    mixed_lab = cv2.cvtColor(mixed_bgr, cv2.COLOR_RGB2LAB)[:, 0, :]
    mixed_hsv = cv2.cvtColor(mixed_bgr, cv2.COLOR_RGB2HSV)[:, 0, :]

    # Add Mean Values
    df["Mixed_Yellow_L_Mean"] = mixed_lab[:, 0]
    df["Mixed_Yellow_a_Mean"] = mixed_lab[:, 1]
    df["Mixed_Yellow_b_Mean"] = mixed_lab[:, 2]

    df["Mixed_Yellow_H_Mean"] = mixed_hsv[:, 0]
    df["Mixed_Yellow_S_Mean"] = mixed_hsv[:, 1]
    df["Mixed_Yellow_V_Mean"] = mixed_hsv[:, 2]

    # Add Standard Deviation (Since we're mixing, we use an estimated std based on weighted input stds)
    df["Mixed_Yellow_L_Std"] = (df["L_Std"] * red_weight + df["L_Std"] * green_weight).astype(int)
    df["Mixed_Yellow_a_Std"] = (df["a_Std"] * red_weight + df["a_Std"] * green_weight).astype(int)
    df["Mixed_Yellow_b_Std"] = (df["b_Std"] * red_weight + df["b_Std"] * green_weight).astype(int)

    df["Mixed_Yellow_H_Std"] = (df["H_Std"] * red_weight + df["H_Std"] * green_weight).astype(int)
    df["Mixed_Yellow_S_Std"] = (df["S_Std"] * red_weight + df["S_Std"] * green_weight).astype(int)
    df["Mixed_Yellow_V_Std"] = (df["V_Std"] * red_weight + df["V_Std"] * green_weight).astype(int)

    return df
base_features["Yellow"] = ["Mixed_RedGreenYellow_Mean", "Mixed_RedGreenYellow_Std"]
base_features["Yellow_LAB"] = ["Mixed_Yellow_L_Mean", "Mixed_Yellow_a_Mean", "Mixed_Yellow_b_Mean", "Mixed_Yellow_L_Std", "Mixed_Yellow_a_Std", "Mixed_Yellow_b_Std"]
base_features["Yellow_HSV"] = ["Mixed_Yellow_H_Mean", "Mixed_Yellow_S_Mean", "Mixed_Yellow_V_Mean", "Mixed_Yellow_H_Std", "Mixed_Yellow_S_Std", "Mixed_Yellow_V_Std"]

def create_interaction_features(df):
    """
    Creates interaction features (cross-multiplication) to enhance model learning.
    """
    df["R_G_Interaction"] = df["R_Mean"] * df["G_Mean"]
    df["R_B_Interaction"] = df["R_Mean"] * df["B_Mean"]
    df["G_B_Interaction"] = df["G_Mean"] * df["B_Mean"]
    
    df["L_H_Interaction"] = df["L_Mean"] * df["H_Mean"]
    df["a_S_Interaction"] = df["a_Mean"] * df["S_Mean"]
    df["b_V_Interaction"] = df["b_Mean"] * df["V_Mean"]

    return df

# Feature Engineering
df = mix_orange(df)
df = mix_yellow(df)
df = create_interaction_features(df)
# Prepare Results Storage
results = []

for feature_name, feature_list in feature_sets.items():
    logging.info(f"Training LSTM with feature set: {feature_name}")
    
    # Define Features and Target
    X = df[feature_list]
    y = df["Weight"]
    
    # Scale Data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    # Reshape for LSTM (samples, time steps, features)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_scaled, test_size=0.2, random_state=42)
    
    # Build LSTM Model
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    # Compile Model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Callbacks
    checkpoint = ModelCheckpoint(f"best_lstm_{feature_name}.h5", save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train Model
    history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), 
                        verbose=1, callbacks=[checkpoint, early_stopping])
    
    # Evaluate Model
    y_pred = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred)
    y_test = scaler_y.inverse_transform(y_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logging.info(f"Feature Set: {feature_name} - MSE: {mse:.4f}, R2 Score: {r2:.4f}")
    
    results.append({
        "Feature Set": feature_name,
        "MSE": mse,
        "R2 Score": r2
    })

# Save Results
df_results = pd.DataFrame(results)
df_results.to_csv("lstm_feature_comparison_results.csv", index=False)
logging.info("Training complete. Results saved.")
