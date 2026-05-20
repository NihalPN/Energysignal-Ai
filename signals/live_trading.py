import sys
import os
import pandas as pd
import sqlite3
import xgboost as xgb

# Force Python to recognize the root directory so it can find the 'utils' folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from utils.notifier import send_alert
except ImportError:
    # Failsafe in case the telegram notifier isn't configured yet
    def send_alert(msg):
        pass


# Paths
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "energy_market.db"
)


def generate_live_signals():
    print("Loading latest market data...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM master_features", conn, parse_dates=["timestamp"], index_col="timestamp"
    )
    conn.close()

    df = df.sort_index()

    # Dynamically identify the latest available day in the database
    latest_timestamp = df.index.max()
    latest_day = latest_timestamp.floor("D")

    print(f"Latest market data found for: {latest_day.strftime('%Y-%m-%d')}")

    historical_df = df.dropna(subset=["target_price_24h_ahead"]).copy()
    live_df = df[df.index >= latest_day].copy()

    X_train = historical_df.drop(columns=["target_price_24h_ahead"])
    y_train = historical_df["target_price_24h_ahead"]

    print("Training Production XGBoost model on all available historical data...")
    model = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=5, objective="reg:squarederror"
    )
    model.fit(X_train, y_train)

    X_live = live_df.drop(columns=["target_price_24h_ahead"])

    prediction_day = latest_day + pd.Timedelta(days=1)
    print(f"Predicting prices for tomorrow: {prediction_day.strftime('%Y-%m-%d')}...\n")

    live_df["predicted_price_tomorrow"] = model.predict(X_live)

    # --- FORMATTED FULL 24-HOUR TABLE OUTPUT ---
    print("===============================================================================")
    print(f"       FULL 24-HOUR PRICE FORECAST FOR {prediction_day.strftime('%Y-%m-%d')}          ")
    print("===============================================================================")
    print(
        f"{'Delivery Time Block':<22} | {'Entry Price':<14} | {'Predicted Exit':<16} | {'Action'}"
    )
    print("-" * 79)

    EXPECTED_PROFIT_MARGIN = 40.0
    trades_found = 0

    for timestamp, row in live_df.iterrows():
        # The exact 15-minute delivery block for the trade
        target_delivery_time = timestamp + pd.Timedelta(hours=24)
        block_end = target_delivery_time + pd.Timedelta(minutes=15)

        # Format the time block label (e.g., "14:15 - 14:30")
        time_label = f"{target_delivery_time.strftime('%H:%M')} - {block_end.strftime('%H:%M')}"

        current_price = row["price_eur_mwh"]
        predicted_price = row["predicted_price_tomorrow"]
        expected_profit = predicted_price - current_price

        if (expected_profit > EXPECTED_PROFIT_MARGIN) and (current_price > 0):
            action = f"BUY (+€{expected_profit:.2f} spread)"
            trades_found += 1

            # Fire off the asynchronous Telegram alert for valid trades
            alert_msg = (
                f"📈 *XGBOOST TRADE SIGNAL* 📈\n\n"
                f"⚡ *Action:* BUY 10 MWh\n"
                f"🕒 *Delivery Block:* {target_delivery_time.strftime('%Y-%m-%d %H:%M')}\n"
                f"💶 *Entry Price:* €{current_price:.2f}\n"
                f"🎯 *Predicted Exit:* €{predicted_price:.2f}\n"
                f"💰 *Expected Gross Profit:* €{expected_profit * 10:.2f}"
            )
            send_alert(alert_msg)
        else:
            action = "IGNORE"

        # Print every single row
        print(f"{time_label:<22} | €{current_price:<13.2f} | €{predicted_price:<15.2f} | {action}")

    print("-" * 79)
    print(f"Total High-Conviction Trades Executed: {trades_found}")


if __name__ == "__main__":
    generate_live_signals()
