import pandas as pd
import sqlite3
import os
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- BULLETPROOF PATHING ---
# Since this script is inside the 'models/' folder, we go UP one level
# to the main project root, then route to the correct subfolders.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

DB_PATH = os.path.join(PROJECT_ROOT, "database", "energy_market.db")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_baseline.json")


def train_xgboost_baseline():
    print(f"🔋 Resolving database path: {DB_PATH}")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"CRITICAL: Database not found at {DB_PATH}. Check your pipeline.")

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        "SELECT * FROM master_features", conn, parse_dates=["timestamp"], index_col="timestamp"
    )
    # Safely drop only the rows where the future target hasn't happened yet
    df = df.dropna(subset=["target_price_24h_ahead"])
    conn.close()

    # Sort strictly by index to prevent lookahead bias
    df = df.sort_index()

    # Define features (X) and target (y)
    y = df["target_price_24h_ahead"]
    X = df.drop(columns=["target_price_24h_ahead"])

    # Institutional-grade Time Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)

    mae_scores = list()
    rmse_scores = list()
    dir_acc_scores = list()

    print("🧠 Training XGBoost Baseline across 5 chronological splits...")
    for train_index, test_index in tscv.split(X):
        # We use iloc to get the rows by their numerical index
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Initialize the Regressor
        model = xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=5, objective="reg:squarederror"
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Calculate Magnitude Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Calculate Directional Accuracy
        actual_direction = np.sign(y_test.values - X_test["price_eur_mwh"].values)
        predicted_direction = np.sign(preds - X_test["price_eur_mwh"].values)

        correct_trend = (actual_direction == predicted_direction).sum()
        dir_acc = correct_trend / len(y_test)

        mae_scores.append(mae)
        rmse_scores.append(rmse)
        dir_acc_scores.append(dir_acc)

    print("\n" + "=" * 50)
    print("       XGBOOST PRODUCTION VALIDATION RESULTS")
    print("=" * 50)
    print(f"Average MAE:  {np.mean(mae_scores):.2f} EUR/MWh")
    print(f"Average RMSE: {np.mean(rmse_scores):.2f} EUR/MWh")
    print(f"Average Directional Accuracy: {np.mean(dir_acc_scores) * 100:.2f}%")
    print("=" * 50)

    # Train the final model on the entire dataset to save for inference
    print("\n⚙️ Training final production model on all available data...")
    final_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    final_model.fit(X, y)

    # Ensure directory exists and save the model
    os.makedirs(MODEL_DIR, exist_ok=True)
    final_model.save_model(MODEL_PATH)
    print(f"✅ Final baseline model saved successfully to {MODEL_PATH}")
    print(f"📊 The model is now fully calibrated with {len(X.columns)} features.")


if __name__ == "__main__":
    train_xgboost_baseline()
