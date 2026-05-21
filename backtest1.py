import os
import sqlite3
import pandas as pd
import xgboost as xgb
import vectorbt as vbt
import warnings

# Suppress pandas deprecation warnings for cleaner terminal output
warnings.simplefilter(action='ignore', category=FutureWarning)

DB_PATH = os.path.join("database", "energy_market.db")

def run_institutional_backtest():
    print("🔋 Initializing VectorBT Institutional Backtest...")
    
    # 1. Load the pristine feature set
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM master_features", 
        conn, 
        index_col="timestamp", 
        parse_dates=["timestamp"]
    ).sort_index()
    conn.close()

    df = df.dropna(subset=["target_price_24h_ahead"])
    
    # 2. Split Data: Train on older data, Backtest on the last 30 days
    test_days = 30
    split_index = len(df) - (test_days * 96)  # 96 blocks per day
    
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:].copy()

    print(f"🧠 Training XGBoost on {len(train_df)} rows...")
    X_train = train_df.drop(columns=["target_price_24h_ahead"])
    y_train = train_df["target_price_24h_ahead"]
    
    X_test = test_df.drop(columns=["target_price_24h_ahead"])
    y_actual = test_df["target_price_24h_ahead"]

    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    model.fit(X_train, y_train)

    print("🔮 Generating Backtest Predictions...")
    predictions = model.predict(X_test)
    test_df["predicted_future_price"] = predictions
    
    # 3. Simulate Strategy Logic (Bypassing negative price constraints)
    EXPECTED_MARGIN = 40.0
    VOLUME_MWH = 10.0
    STARTING_CAPITAL = 100000.0

    test_df["current_entry_price"] = test_df["price_eur_mwh"]
    test_df["expected_spread"] = test_df["predicted_future_price"] - test_df["current_entry_price"]

    # Generate the exact INVEST signals from your UI
    entries = (test_df["expected_spread"] > EXPECTED_MARGIN) & (test_df["current_entry_price"] > 0)
    
    print("📊 Executing Institutional Portfolio Simulation...")
    
    # Calculate strict cash profit for each trade manually to allow for negative power prices
    test_df["trade_pnl"] = 0.0
    
    # Profit = (Actual Exit Price - Actual Entry Price) * 10 MWh
    test_df.loc[entries, "trade_pnl"] = (test_df.loc[entries, "target_price_24h_ahead"] - test_df.loc[entries, "current_entry_price"]) * VOLUME_MWH
    
    # Convert absolute cash profit into a percentage return
    test_df["strategy_return"] = test_df["trade_pnl"] / STARTING_CAPITAL
    
    # 4. Generate the Tear Sheet safely using VectorBT's native Returns Accessor
    print("\n" + "="*50)
    print("       VECTORBT STRATEGY TEAR SHEET")
    print("="*50)
    
    # THE FIX: Direct accessor bypasses the Portfolio object entirely
    tear_sheet = test_df["strategy_return"].vbt.returns.stats()
    print(tear_sheet)
    print("="*50)
    
    # 5. Print a quick summary of the raw cash execution
    total_trades = entries.sum()
    winning_trades = (test_df["trade_pnl"] > 0).sum()
    total_profit = test_df["trade_pnl"].sum()
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    print(f"\n💰 RAW CASH SUMMARY (Past 30 Days):")
    print(f"Total Trades Executed: {total_trades}")
    print(f"Winning Trades: {winning_trades} ({win_rate:.1f}% Win Rate)")
    print(f"Net Cash Profit: €{total_profit:,.2f}")

if __name__ == "__main__":
    run_institutional_backtest()
