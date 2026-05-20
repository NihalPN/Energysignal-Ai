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
    master_df = tech_df.join(ren_df, how="inner").join(weather_df, how="inner")

    # Define the Target Variable: The price 24 hours (96 steps) into the future
    master_df["target_price_24h_ahead"] = master_df["price_eur_mwh"].shift(-96)

    # THE FIX: Drop NaNs from the historical lags, but KEEP the NaNs in the target variable
    cols_to_check = [col for col in master_df.columns if col != "target_price_24h_ahead"]
    master_df = master_df.dropna(subset=cols_to_check)

    master_df.to_sql("master_features", con=conn, if_exists="replace")
    conn.close()

    print(f"Feature engineering complete. Master table shape: {master_df.shape}")


if __name__ == "__main__":
    build_and_store_master_features()
