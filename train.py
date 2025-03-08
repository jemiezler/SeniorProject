import os
import pandas as pd
import numpy as np
import joblib
from itertools import combinations
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers import Input

# ✅ Set GPU for TensorFlow
def setup_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure GPU usage

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print("✅ TensorFlow is using GPU:", gpus[0])
        except RuntimeError as e:
            print(e)

# ✅ Load Data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.sort_values(by=["Day"])
    return df

# ✅ Preprocess Data (Time-Series Lag Features & Split)
def preprocess_data(df, target="%_Weight_Loss", lags=3):
    for lag in range(1, lags + 1):
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    df = df.dropna()

    X = df.drop(columns=["Filename", "Weight", target, "Day"], errors="ignore")
    y = df[target]

    train_size = int(0.8 * len(X))
    return X.iloc[:train_size], X.iloc[train_size:], y.iloc[:train_size], y.iloc[train_size:]

# ✅ Train Linear Regression
def train_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

# ✅ Train XGBoost (GPU Enabled)
def train_xgboost(X_train, X_test, y_train, y_test):
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        learning_rate=0.1,
        tree_method="hist",
        device="cuda",  # Ensure GPU is used
    )
    X_train, X_test = np.array(X_train), np.array(X_test)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred

# ✅ Train LSTM Model
def train_lstm(X_train, X_test, y_train, y_test):
    X_train, X_test = X_train.values, X_test.values
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    model = Sequential()
    model.add(Input(shape=(1, X_train.shape[2])))
    model.add(LSTM(50, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.01), loss="mse")
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    y_pred = model.predict(X_test).flatten()
    return model, y_pred

# ✅ Evaluate Model
def evaluate_model(model_name, y_test, y_pred, feature_group, model_results):
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    model_results.append((model_name, feature_group, rmse, r2))
    return rmse

# ✅ Main Function
def main():
    setup_gpu()
    file_path = "resources/color_texture_weight_data.csv"
    df = load_data(file_path)
    X_train_full, X_test_full, y_train, y_test = preprocess_data(df)

    feature_groups = {
        "RGB": ["Mean_RGB_R", "Mean_RGB_G", "Mean_RGB_B", "Std_RGB_R", "Std_RGB_G", "Std_RGB_B"],
        "LAB": ["Mean_LAB_L", "Mean_LAB_A", "Mean_LAB_B", "Std_LAB_L", "Std_LAB_A", "Std_LAB_B"],
        "HSV": ["Mean_HSV_H", "Mean_HSV_S", "Mean_HSV_V", "Std_HSV_H", "Std_HSV_S", "Std_HSV_V"],
        "GLCM": ["GLCM_ASM", "GLCM_contrast", "GLCM_correlation", "GLCM_dissimilarity", "GLCM_energy"],
        "LBP": ["LBP_0", "LBP_1", "LBP_2", "LBP_3", "LBP_4"],
        "Yellow": ["Yellow"],
        "Cyan": ["Cyan"],
        "Magenta": ["Magenta"],
        "Brightness": ["Brightness"],
        "Chroma": ["Chroma"],
    }

    group_combinations = []
    for k in range(1, len(feature_groups) + 1):
        group_combinations.extend(combinations(feature_groups.keys(), k))

    model_results = []
    best_rmse = float("inf")
    best_model = None
    best_model_name = ""
    best_feature_group = ""

    # ✅ Train Models for Each Feature Group Combination
    for group_combo in tqdm(group_combinations, desc="Testing Models"):
        selected_features = [f for g in group_combo for f in feature_groups[g] if f in X_train_full.columns]
        if not selected_features:
            continue

        X_train, X_test = X_train_full[selected_features], X_test_full[selected_features]

        # ✅ Train and Evaluate Each Model
        for model_name, train_func in zip(["LinearRegression", "XGBoost", "LSTM"],
                                          [train_linear_regression, train_xgboost, train_lstm]):
            model, y_pred = train_func(X_train, X_test, y_train, y_test)
            rmse = evaluate_model(model_name, y_test, y_pred, " + ".join(group_combo), model_results)

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_model_name = model_name
                best_feature_group = " + ".join(group_combo)

    # ✅ Save Results
    results_df = pd.DataFrame(model_results, columns=["Model", "Feature_Groups", "RMSE", "R2_Score"])
    results_df.to_csv("feature_group_selection_results.csv", index=False)

    # ✅ Save Best Model
    if best_model_name == "LSTM":
        best_model.save("best_time_series_lstm_model.h5")
        best_model_path = "best_time_series_lstm_model.h5"
    else:
        joblib.dump(best_model, "best_time_series_model.pkl")
        best_model_path = "best_time_series_model.pkl"

    # ✅ Save Best Feature Groups
    pd.DataFrame([[best_model_name, best_feature_group]], columns=["Best_Model", "Best_Feature_Groups"]).to_csv(
        "best_selected_feature_groups.csv", index=False
    )

    print(f"Best Model: {best_model_name}")
    print(f"Best Feature Groups: {best_feature_group}")
    print(f"Best RMSE: {best_rmse}")
    print("Saved:")
    print("- feature_group_selection_results.csv")
    print(f"- {best_model_path}")
    print("- best_selected_feature_groups.csv")

# ✅ Run Script
if __name__ == "__main__":
    main()
