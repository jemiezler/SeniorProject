#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries and Load data | Day_Treatment_Rep

# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[15]:


df = pd.read_csv('weight_color_data.csv')
df['Day'] = df['Label'].apply(lambda x: x.split('_')[0])
df['Temp'] = df['Label'].apply(lambda x: x.split('_')[1])
df.drop('Label', axis=1, inplace=True)
df.head()


# In[16]:


df.shape


# In[17]:


df.columns


# ## Training

# In[18]:


import cv2
import numpy as np
import pandas as pd
from itertools import combinations
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, QuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# In[19]:


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
    # "Day": ["Day"],
}

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Polynomial Regression": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Huber Regressor": HuberRegressor(),
    "Quantile Regressor": QuantileRegressor(quantile=0.5, alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
}


# In[20]:


def progressive_features(features_dict):
    feature_combinations = {}

    feature_groups = list(features_dict.keys())  # ["L", "a", "b", "H", ...]
    
    for i in range(1, len(feature_groups) + 1):
        for comb in combinations(feature_groups, i):  # Create feature group combinations
            combined_columns = sum([features_dict[key] for key in comb], [])  # Map to actual column names
            feature_combinations[",".join(comb)] = combined_columns

    print(f"Generated {len(feature_combinations)} feature combinations.")
    return feature_combinations


# In[21]:


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


# In[23]:


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


# In[24]:


def add_polynomial_features(X, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    return pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X.columns))


# In[ ]:


import sys
import tqdm
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

class SingleLineLogger:
    def __init__(self):
        self.last_msg = ''
    
    def write(self, msg):
        # Clear the last message by writing spaces
        if self.last_msg:
            sys.stdout.write('\r' + ' ' * len(self.last_msg) + '\r')
        # Write the new message
        sys.stdout.write('\r' + msg)
        sys.stdout.flush()
        self.last_msg = msg

    def flush(self):
        pass

class VSCodeFormatter(logging.Formatter):
    def format(self, record):
        # Simplified format for VS Code output
        return f"{record.levelname}: {record.getMessage()}"

def setup_logging():
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create custom stream handler with single line output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(VSCodeFormatter())
    logger.addHandler(console_handler)
    
    # File handler for complete logs
    file_handler = logging.FileHandler('training.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    return logger

def train_and_evaluate(X, y, feature_sets, models, output_csv_path):
    logger = setup_logging()
    results = []

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Calculate total steps
    total_steps = len(feature_sets) * len(models)

    # Custom progress bar format
    bar_format = '{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    
    with tqdm.tqdm(total=total_steps, desc="Training Progress", bar_format=bar_format, 
              file=sys.stdout, position=0, leave=True) as pbar:
        
        for feature_name, feature_list in feature_sets.items():
            X_train_subset = X_train[feature_list]
            X_test_subset = X_test[feature_list]

            for model_name, model in models.items():
                try:
                    current_msg = f"Training {model_name} with {feature_name}"
                    pbar.set_description(current_msg)
                    
                    # Train and evaluate
                    model.fit(X_train_subset, y_train)
                    y_pred = model.predict(X_test_subset)

                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    results.append({
                        "Feature Set": feature_name,
                        "Model": model_name,
                        "MSE": mse,
                        "R2 Score": r2
                    })

                    logger.info(f"{feature_name}, {model_name}: MSE={mse:.4f}, R2={r2:.4f}")

                except Exception as e:
                    logger.error(f"Error in {model_name} with {feature_name}: {str(e)}")
                
                finally:
                    pbar.update(1)

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv_path, index=False)
    logger.info(f"Training complete. Results saved to {output_csv_path}")

    return df_results


# In[26]:


# 🔹 Load Data
from multiprocessing import Pool


df = mix_orange(df)  # Apply color mixing
df = mix_yellow(df)
df = create_interaction_features(df)  # Add interaction features

# Define X (features) and y (target)
X = df[sum(base_features.values(), [])]
y = df["Weight"]

# Train & Evaluate Models
with Pool(processes=12) as pool:
    train_and_evaluate(X, y, progressive_features(base_features), models, "../output/train_csv/interact.csv")

