import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from technical_features import calculate_technical_features
from renewable_features import calculate_renewable_features

# --- CLOUD DATABASE SETUP ---
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("❌ CRITICAL: Missing DATABASE_URL in environment variables.")

# Instantiate the cloud connection
engine = create_engine(DB_URL)


def build_and_store_master_features():
    print("Calculating Technical Features...")
    tech_df = calculate_technical_features()

    print("Calculating Renewable Features...")
    ren_df = calculate_renewable_features()

    print("Fetching Weather Data from Neon Postgres...")
    # Read directly from the cloud database via SQLAlchemy
    weather_df = pd.read_sql_query(
        "SELECT * FROM weather_data", engine, parse_dates=["timestamp"], index_col="timestamp"
    )

    print("Merging master feature dataset...")
    master_df = tech_df.join(ren_df, how="outer").join(weather_df, how="outer")

    # THE FIX: Drop rows where the REAL exchange price doesn't exist yet
    master_df = master_df.dropna(subset=["price_eur_mwh"])

    # NOW safely forward-fill the missing grid actuals (load/generation) up to 8 hours
    master_df = master_df.ffill(limit=32)

    # --- THE SOLAR FIX (Bulletproof Math) ---
    solar_col = next(
        (col for col in master_df.columns if "solar" in col.lower() or "radiation" in col.lower()),
        None,
    )

    if solar_col:
        print(f"Dynamically detected Solar column: '{solar_col}'")

        # Safely calculate total grid load based on known ENTSO-E residual data
        if "residual_load" in master_df.columns and "total_renewable" in master_df.columns:
            total_grid_load = master_df["residual_load"] + master_df["total_renewable"]
            print("Calculated actual load from residual_load + total_renewable")
        else:
            # Absolute fallback to prevent crashes
            total_grid_load = 50000
            print("Could not find residual load. Using safe fallback divisor.")

        # 1. Solar Penetration Ratio
        master_df["solar_penetration_ratio"] = master_df[solar_col] / (total_grid_load + 1)

        # 2. Non-Linear Solar Penalty
        master_df["solar_squared"] = master_df[solar_col] ** 2
    else:
        print("⚠️ WARNING: Could not find solar column. Skipping solar feature engineering.")
        master_df["solar_penetration_ratio"] = 0
        master_df["solar_squared"] = 0

    # 3. Peak Daylight Flag: Binary feature telling the AI it is prime crash time (11:00 to 15:00)
    master_df["is_peak_solar_window"] = (
        (master_df.index.hour >= 11) & (master_df.index.hour <= 15)
    ).astype(int)
    # -------------------------------------------

    # Define the Target Variable: The price 24 hours (96 steps) into the future
    master_df["target_price_24h_ahead"] = master_df["price_eur_mwh"].shift(-96)

    # Stop deleting tomorrow's prices!
    safe_cols_to_check = list(tech_df.columns) + list(weather_df.columns)
    master_df = master_df.dropna(subset=safe_cols_to_check)

    print("Pushing master features back to Neon Postgres...")
    # Write the finished dataset back to the cloud using the SQLAlchemy engine
    master_df.to_sql("master_features", con=engine, if_exists="replace")

    print(f"✅ Feature engineering complete. Master table shape: {master_df.shape}")


if __name__ == "__main__":
    build_and_store_master_features()
