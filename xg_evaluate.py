import os
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DB_PATH = os.path.join("database", "energy_market.db")


def evaluate_xgboost_model():
    print("🔋 Loading Database for ML Evaluation...")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM master_features", conn, index_col="timestamp", parse_dates=["timestamp"]
    ).sort_index()
    conn.close()

    df = df.dropna(subset=["target_price_24h_ahead"])

    # Standard 30-Day Split
    test_days = 30
    split_index = len(df) - (test_days * 96)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:].copy()

    X_train = train_df.drop(columns=["target_price_24h_ahead"])
    y_train = train_df["target_price_24h_ahead"]

    X_test = test_df.drop(columns=["target_price_24h_ahead"])
    y_actual = test_df["target_price_24h_ahead"]

    print(f"🧠 Training XGBoost Regressor on {len(X_train)} rows...")
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)

    print("🔮 Generating Predictions for Test Set...")
    predictions = model.predict(X_test)

    # --- CALCULATE STANDARD DATA SCIENCE METRICS ---
    mae = mean_absolute_error(y_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_actual, predictions))
    r2 = r2_score(y_actual, predictions)

    print("\n" + "=" * 50)
    print("       XGBOOST MACHINE LEARNING SCORES")
    print("=" * 50)
    print(f"Mean Absolute Error (MAE): €{mae:.2f} per MWh")
    print(f"Root Mean Squared Error (RMSE): €{rmse:.2f} per MWh")
    print(f"R-Squared (R²): {r2:.4f} (Accuracy Score)")
    print("=" * 50)

    # --- CALCULATE DIRECTIONAL ACCURACY ---
    # We compare the predicted spread against the actual spread
    # to see if the model guessed the right direction (UP or DOWN)
    current_prices = X_test["price_eur_mwh"].values

    actual_direction = np.sign(y_actual - current_prices)
    predicted_direction = np.sign(predictions - current_prices)

    # We only care about moments where the market actually moved
    valid_mask = actual_direction != 0
    directional_accuracy = (
        np.mean(actual_direction[valid_mask] == predicted_direction[valid_mask]) * 100
    )

    # --- CALCULATE "PROFITABLE" DIRECTIONAL ACCURACY ---
    # How accurate is the model when it explicitly tells us to trade? (Spread > €40)
    expected_spread = predictions - current_prices
    trade_signals = expected_spread > 40.0

    if trade_signals.sum() > 0:
        actual_profit = (y_actual - current_prices)[trade_signals]
        winning_trades = np.sum(actual_profit > 0)
        signal_accuracy = (winning_trades / trade_signals.sum()) * 100
    else:
        signal_accuracy = 0.0

    print("\n📈 TRADING EDGE METRICS:")
    print(f"General Directional Accuracy (Up/Down): {directional_accuracy:.1f}%")
    print(f"INVEST Signal Win Rate (>€40 Spread): {signal_accuracy:.1f}%")

    # --- EXTRACT FEATURE IMPORTANCE ---
    print("\n📊 TOP 10 MOST IMPORTANT FEATURES:")
    importance = model.feature_importances_
    feature_names = X_train.columns

    # Create a dataframe to sort and display feature importance
    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importance})
        .sort_values(by="Importance", ascending=False)
        .head(10)
    )

    for index, row in importance_df.iterrows():
        print(f" - {row['Feature']}: {row['Importance'] * 100:.2f}%")


if __name__ == "__main__":
    evaluate_xgboost_model()
