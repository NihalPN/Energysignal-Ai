import pandas as pd
import os
import holidays
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Force Python to read the .env file locally
load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("❌ CRITICAL: Missing DATABASE_URL in environment variables.")

engine = create_engine(DB_URL)


def calculate_technical_features():
    print("Fetching Day-Ahead Prices from Cloud Database...")

    # EXACT ORIGINAL LOGIC, just using engine instead of conn
    prices_df = pd.read_sql_query(
        "SELECT * FROM day_ahead_prices", engine, parse_dates=["timestamp"], index_col="timestamp"
    )

    # Time-based features
    prices_df["hour"] = prices_df.index.hour
    prices_df["minute"] = prices_df.index.minute
    prices_df["day_of_week"] = prices_df.index.dayofweek
    prices_df["month"] = prices_df.index.month

    # German Public Holidays Flag
    de_holidays = holidays.DE()
    prices_df["is_holiday"] = prices_df.index.map(lambda x: 1 if x in de_holidays else 0)

    # THE REGIME SHIFT FLAG
    # 0 before Oct 1, 2025 (Hourly Era), 1 after Oct 1, 2025 (15-min Era)
    prices_df["is_15min_era"] = (prices_df.index >= "2025-10-01").astype(int)

    # Lagged Prices (15-min resolution: 1hr = 4 steps, 24hr = 96 steps, 168hr = 672 steps)
    prices_df["price_lag_1h"] = prices_df["price_eur_mwh"].shift(4)
    prices_df["price_lag_24h"] = prices_df["price_eur_mwh"].shift(96)
    prices_df["price_lag_168h"] = prices_df["price_eur_mwh"].shift(672)

    # Rolling Volatility (24hr rolling window = 96 steps)
    prices_df["volatility_24h"] = prices_df["price_eur_mwh"].rolling(window=96).std()

    return prices_df
