import os
import sqlite3
import pandas as pd
import requests
import logging
from dotenv import load_dotenv

load_dotenv()

# Setup Logging
log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "anomaly.log")
logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s - %(message)s")

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "energy_market.db"
)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_alert(message: str):
    """Phase 4: Asynchronous push notification to your phone."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram credentials missing in.env file.")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}

    try:
        requests.post(url, json=payload, timeout=5)
        logging.info("Anomaly alert dispatched to Telegram.")
    except Exception as e:
        logging.error(f"Failed to send Telegram alert: {e}")


def detect_grid_anomalies():
    """Phase 2: Detects sudden supply shocks in the German power grid."""
    print("Scanning recent grid physics for anomalies...")

    conn = sqlite3.connect(DB_PATH)
    gen_df = pd.read_sql_query(
        "SELECT * FROM generation_mix", conn, parse_dates=["timestamp"], index_col="timestamp"
    )
    load_df = pd.read_sql_query(
        "SELECT * FROM actual_load", conn, parse_dates=["timestamp"], index_col="timestamp"
    )
    conn.close()

    # Merge physical data
    df = gen_df.join(load_df, how="inner").sort_index()
    if df.empty:
        print("No data available for anomaly detection.")
        return

    # Calculate Residual Load (The actual strain on fossil fuel plants)
    df["total_renewable"] = df["wind_onshore"] + df["wind_offshore"] + df["solar"]
    df["residual_load"] = df["load_mw"] - df["total_renewable"]

    # We use a 24-hour rolling window (96 steps of 15-mins) to calculate normal behavior
    rolling_mean = df["residual_load"].rolling(window=96).mean()
    rolling_std = df["residual_load"].rolling(window=96).std()

    # Calculate the Z-Score (How many standard deviations away from normal is the current state?)
    df["z_score"] = (df["residual_load"] - rolling_mean) / rolling_std

    # Look at the most recent 15-minute block
    latest_time = df.index[-1]
    latest_data = df.iloc[-1]
    z_score = latest_data["z_score"]

    print(f"Latest Block: {latest_time} | Residual Load Z-Score: {z_score:.2f}")

    # A Z-Score above 3.0 indicates a massive grid shock
    if z_score > 3.0:
        alert_msg = (
            f"🚨 *GRID ANOMALY DETECTED* 🚨\n\n"
            f"🕒 *Time:* {latest_time}\n"
            f"⚠️ *Type:* Severe Supply Shock (Z-Score: {z_score:.2f})\n"
            f"📈 *Residual Load:* {latest_data['residual_load']:.0f} MW\n"
            f"🌬️ *Renewables:* {latest_data['total_renewable']:.0f} MW\n\n"
            f"_Action:_ Volatility incoming. Triggering defensive trading protocols."
        )
        print("Anomaly detected! Firing alert...")
        send_telegram_alert(alert_msg)

    elif z_score < -3.0:
        alert_msg = (
            f"⚠️ *GRID ANOMALY DETECTED* ⚠️\n\n"
            f"🕒 *Time:* {latest_time}\n"
            f"📉 *Type:* Massive Oversupply (Z-Score: {z_score:.2f})\n"
            f"☀️ *Renewables:* {latest_data['total_renewable']:.0f} MW\n\n"
            f"_Action:_ Negative pricing likely. Halt standard LONG positions."
        )
        print("Anomaly detected! Firing alert...")
        send_telegram_alert(alert_msg)
    else:
        print("Grid operating within normal statistical parameters. No alerts fired.")


if __name__ == "__main__":
    detect_grid_anomalies()
