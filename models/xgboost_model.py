import pandas as pd
import sqlite3
import os
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Path to the SQLite database
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database', 'energy_market.db')

def train_xgboost_baseline():
    print("Loading master features from database...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM master_features", conn, parse_dates=['timestamp'], index_col='timestamp')
    conn.close()

    # Sort strictly by index to prevent lookahead bias
    df = df.sort_index()

    # Define features (X) and target (y)
    y = df['target_price_24h_ahead']
    X = df.drop(columns=['target_price_24h_ahead'])

    # Institutional-grade Time Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # BULLETPROOF FIX: Using list() instead of empty brackets so the AI doesn't delete them
    mae_scores = list()
    rmse_scores = list()
    dir_acc_scores = list()

    print("Training XGBoost Baseline across 5 chronological splits...")
    for train_index, test_index in tscv.split(X):
        # We use iloc to get the rows by their numerical index
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Initialize the Regressor
        model = xgb.XGBRegressor(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=5, 
            objective='reg:squarederror'
        )
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Calculate Magnitude Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        
        # Calculate Directional Accuracy (Did we correctly guess if the price will go UP or DOWN?)
        actual_direction = np.sign(y_test.values - X_test['price_eur_mwh'].values)
        predicted_direction = np.sign(preds - X_test['price_eur_mwh'].values)
        
        correct_trend = (actual_direction == predicted_direction).sum()
        dir_acc = correct_trend / len(y_test)

        mae_scores.append(mae)
        rmse_scores.append(rmse)
        dir_acc_scores.append(dir_acc)

    print("\n=== XGBoost Baseline Validation Results ===")
    print(f"Average MAE:  {np.mean(mae_scores):.2f} EUR/MWh")
    print(f"Average RMSE: {np.mean(rmse_scores):.2f} EUR/MWh")
    print(f"Average Directional Accuracy: {np.mean(dir_acc_scores) * 100:.2f}%")
    
    # Train the final model on the entire dataset to save for inference
    print("\nTraining final production model on all available data...")
    final_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    final_model.fit(X, y)
    
    # Save the model
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'xgb_baseline.json')
    final_model.save_model(model_path)
    print(f"Final baseline model saved successfully to {model_path}")

if __name__ == "__main__":
    train_xgboost_baseline()
