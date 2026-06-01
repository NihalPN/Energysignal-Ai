import pandas as pd
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Force Python to read the .env file locally
load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("❌ CRITICAL: Missing DATABASE_URL in environment variables.")

engine = create_engine(DB_URL)


def calculate_renewable_features():
    print("Fetching Generation and Load Data from Cloud Database...")

    # EXACT ORIGINAL LOGIC, just using engine instead of conn
    gen_df = pd.read_sql_query(
        "SELECT * FROM generation_mix", engine, parse_dates=["timestamp"], index_col="timestamp"
    )
    load_df = pd.read_sql_query(
        "SELECT * FROM actual_load", engine, parse_dates=["timestamp"], index_col="timestamp"
    )

    # Merge on timestamp
    df = gen_df.join(load_df, how="inner")

    # Calculate Total Renewables
    df["total_renewable"] = df["wind_onshore"] + df["wind_offshore"] + df["solar"]

    # Calculate Penetration Ratio
    df["renewable_penetration"] = df["total_renewable"] / df["load_mw"]

    # Calculate Residual Load (Demand that must be covered by thermal/gas plants)
    df["residual_load"] = df["load_mw"] - df["total_renewable"]

    return df[["total_renewable", "renewable_penetration", "residual_load"]]
