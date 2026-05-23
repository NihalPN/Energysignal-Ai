import pandas as pd
import sqlite3
import os
import xgboost as xgb
import vectorbt as vbt

# Paths
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "energy_market.db"
)


def run_institutional_backtest():
    print("Loading data...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM master_features", conn, parse_dates=["timestamp"], index_col="timestamp"
    )
    conn.close()

    df = df.sort_index()

    # Calculate 7-day rolling average
    df["rolling_7d_avg"] = df["price_eur_mwh"].rolling(window=672).mean()
    df = df.dropna()

    # THE FIX: Strict Separation of Train (80%) and Test (20%) Sets
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # Define training features and target
    X_train = train_df.drop(columns=["target_price_24h_ahead", "rolling_7d_avg"])
    y_train = train_df["target_price_24h_ahead"]

    # Define testing features and target
    X_test = test_df.drop(columns=["target_price_24h_ahead", "rolling_7d_avg"])

    # THE FIX: Train a fresh model ONLY on the in-sample training data
    print("Training XGBoost model exclusively on the in-sample training set...")
    model = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=5, objective="reg:squarederror"
    )
    model.fit(X_train, y_train)

    print("Generating strictly out-of-sample predictions...")
    test_df["predicted_price"] = model.predict(X_test)

    # --- TRADING SIGNAL LOGIC ---
    print("Applying high-conviction time-horizon constraints...")

    EXPECTED_PROFIT_MARGIN = 40.0

    raw_entries = test_df["predicted_price"] > (test_df["price_eur_mwh"] + EXPECTED_PROFIT_MARGIN)

    # Crossover logic to prevent rapid churning
    entries = raw_entries & (~raw_entries.shift(1, fill_value=False))

    # 24-Hour Time Exit (96 steps)
    exits = entries.shift(96, fill_value=False)

    # Price Offset to prevent VectorBT negative price crash
    offset_price = test_df["price_eur_mwh"] + 500.0

    # --- VECTORBT BACKTEST ENGINE ---
    print("\nExecuting VectorBT Engine...")

    portfolio = vbt.Portfolio.from_signals(
        close=offset_price,
        entries=entries,
        exits=exits,
        init_cash=100000.0,
        size=10,
        size_type="amount",
        fees=0.0,
        fixed_fees=1.50,
        freq="15min",
    )

    print("\n==================================================")
    print("       INSTITUTIONAL BACKTEST RESULTS (OOS)       ")
    print("==================================================")

    stats_dict = portfolio.stats().to_dict()

    # Using VectorBT's dedicated profit function for accuracy
    profit_val = portfolio.total_profit()
    win_val = stats_dict.get("Win Rate [%]", 0.0)
    trades_val = stats_dict.get("Total Trades", 0)

    print(f"Total Profit (EUR):     €{profit_val:,.2f}")
    print(f"Win Rate:               {win_val:.2f}%")
    print(f"Total Trades Executed:  {trades_val}")

    print("\nGenerating Trade Log...")
    trade_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_log.csv")
    portfolio.trades.records_readable.to_csv(trade_log_path, index=False)
    print(f"Full trade log saved to {trade_log_path}")


if __name__ == "__main__":
    run_institutional_backtest()
