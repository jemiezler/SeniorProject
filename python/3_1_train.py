#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import logging
from itertools import combinations
from multiprocessing import Pool
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, QuantileRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Day'] = df['Label'].apply(lambda x: x.split('_')[0])
    df['Temp'] = df['Label'].apply(lambda x: x.split('_')[1])
    df.drop('Label', axis=1, inplace=True)
    return df

# Define base feature groups
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
}

def progressive_features(features_dict):
    feature_combinations = {}
    feature_groups = list(features_dict.keys())
    for i in range(1, len(feature_groups) + 1):
        for comb in combinations(feature_groups, i):
            combined_columns = sum([features_dict[key] for key in comb], [])
            feature_combinations[",".join(comb)] = combined_columns
    return feature_combinations

def mix_colors(df, color1, color2, name, weight1=0.5, weight2=0.5):
    df[f"Mixed_{name}_Mean"] = (df[f"{color1}_Mean"] * weight1 + df[f"{color2}_Mean"] * weight2).astype(int)
    df[f"Mixed_{name}_Std"] = (df[f"{color1}_Std"] * weight1 + df[f"{color2}_Std"] * weight2).astype(int)
    return df

def create_interaction_features(df):
    df["R_G_Interaction"] = df["R_Mean"] * df["G_Mean"]
    df["R_B_Interaction"] = df["R_Mean"] * df["B_Mean"]
    df["G_B_Interaction"] = df["G_Mean"] * df["B_Mean"]
    return df

def train_and_evaluate(X, y, feature_sets, models, output_csv):
    results = []
    for feature_name, feature_list in feature_sets.items():
        X_train, X_test, y_train, y_test = train_test_split(X[feature_list], y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            logging.info(f"{feature_name}, {model_name} - R² Score: {r2:.4f}, MSE: {mse:.4f}")
            results.append({"Feature Set": feature_name, "Model": model_name, "R2 Score": r2, "MSE": mse})
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print("Results saved to", output_csv)

if __name__ == "__main__":
    file_path = "weight_color_data.csv"
    output_csv = "model_feature_results.csv"
    
    df = load_data(file_path)
    df = mix_colors(df, "R", "G", "Orange", weight1=0.8, weight2=0.2)
    df = mix_colors(df, "R", "G", "Yellow", weight1=0.5, weight2=0.5)
    df = create_interaction_features(df)
    
    X = df[sum(base_features.values(), [])]
    y = df["Weight"]
    feature_sets = progressive_features(base_features)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42, verbose=-1)
    }
    
    train_and_evaluate(X, y, feature_sets, models, output_csv)
