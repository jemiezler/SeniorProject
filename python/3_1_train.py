import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load data in chunks to prevent memory issues
def load_data(file_path, chunk_size=10000):
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

# Feature Engineering
def apply_transformations(df):
    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    df.fillna(0, inplace=True)  # Replace NaNs with 0
    
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64] and df[col].min() > 0:  # Apply log transformation only to positive numeric values
            df[col + "_Log"] = np.log1p(df[col])
    
    # Polynomial Features (Degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(df.drop("Weight", axis=1, errors='ignore'))
    df_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(df.drop("Weight", axis=1, errors='ignore').columns))
    
    return df_poly

# Train Model
def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standard Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train Model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"R² Score: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    
    return model

# Hyperparameter Tuning
def optimize_model(X, y):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X, y)
    print("Best Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

# Main Execution
if __name__ == "__main__":
    file_path = "weight_color_data.csv"  # Update with the correct file path
    df = load_data(file_path)
    df_poly = apply_transformations(df)
    model = train_and_evaluate(df_poly, df["Weight"])
    best_model = optimize_model(df_poly, df["Weight"])
