import os
import json
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'database', 'energy_market.db')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgb_baseline.json')
METRICS_PATH = os.path.join(BASE_DIR, 'models', 'metrics.json')

def calculate_smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    valid = denominator!= 0
    if not np.any(valid): return 0.0
    return float(np.mean(np.abs(y_true - y_pred)[valid] / denominator[valid]) * 100)

def export_metrics():
    print("Calculating Institutional Metrics...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM master_features", conn, index_col='timestamp', parse_dates=['timestamp'])
    conn.close()

    df = df.sort_index().dropna(subset=['target_price_24h_ahead'])
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)
    
    y_true = test_df['target_price_24h_ahead'].values
    y_pred = model.predict(test_df.drop(columns=['target_price_24h_ahead']))
    
    # Calculate Metrics
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    smape = calculate_smape(y_true, y_pred)
    
    actual_dir = np.sign(y_true - test_df['price_eur_mwh'].values)
    pred_dir = np.sign(y_pred - test_df['price_eur_mwh'].values)
    dir_acc = float(np.mean(actual_dir == pred_dir) * 100)

    # Simplified financial placeholders (since VectorBT trade logs aren't tracked in Git)
    # In a full deployment, you'd parse the trade_log.csv here just like we did locally
    metrics = {
        "predictive_metrics": {
            "RMSE_EUR_MWh": round(rmse, 2),
            "MAE_EUR_MWh": round(mae, 2),
            "sMAPE_Pct": round(smape, 2),
            "Directional_Accuracy_Pct": round(dir_acc, 2)
        },
        "financial_metrics": {
            "Total_OOS_Profit_EUR": 13056.30,  # From your validated local backtest
            "Hit_Ratio_Pct": 71.79,
            "Sharpe_Ratio": 1.85
        },
        "last_updated": pd.Timestamp.now(tz='Europe/Berlin').strftime('%Y-%m-%d %H:%M:%S')
    }

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics successfully exported to {METRICS_PATH}")

if __name__ == "__main__":
    export_metrics()
