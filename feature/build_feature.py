import sqlite3
import os
import pandas as pd
from technical_features import calculate_technical_features
from renewable_features import calculate_renewable_features

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "energy_market.db")


def build_and_store_master_features():
    print("Calculating Technical Features...")
    tech_df = calculate_technical_features()

    print("Calculating Renewable Features...")
    ren_df = calculate_renewable_features()

    print("Fetching Weather Data...")
    conn = sqlite3.connect(DB_PATH)
    weather_df = pd.read_sql_query(
        "SELECT * FROM weather_data", conn, parse_dates=["timestamp"], index_col="timestamp"
    )

    print("Merging master feature dataset...")
    master_df = tech_df.join(ren_df, how="outer").join(weather_df, how="outer")

    # THE FIX: Drop rows where the REAL exchange price doesn't exist yet (e.g., tomorrow's weather forecast)
    master_df = master_df.dropna(subset=['price_eur_mwh'])

    # NOW safely forward-fill the missing grid actuals (load/generation) up to 8 hours
    master_df = master_df.ffill(limit=32)

    # Define the Target Variable: The price 24 hours (96 steps) into the future
    master_df["target_price_24h_ahead"] = master_df["price_eur_mwh"].shift(-96)

    # THE FIX 3 (CRITICAL): Stop deleting tomorrow's prices! 
    # Only drop rows if Technical (Prices) or Weather are missing (clears historical lag NaNs). 
    # We allow Renewable/Load features to be NaN in the future so the model doesn't go blind.
    safe_cols_to_check = list(tech_df.columns) + list(weather_df.columns)
    master_df = master_df.dropna(subset=safe_cols_to_check)

    master_df.to_sql("master_features", con=conn, if_exists="replace")
    conn.close()

    print(f"Feature engineering complete. Master table shape: {master_df.shape}")


if __name__ == "__main__":
    build_and_store_master_features()
