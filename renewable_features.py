import pandas as pd
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "energy_market.db")


def calculate_renewable_features():
    conn = sqlite3.connect(DB_PATH)

    # Load generation and load data
    gen_df = pd.read_sql_query(
        "SELECT * FROM generation_mix", conn, parse_dates=["timestamp"], index_col="timestamp"
    )
    load_df = pd.read_sql_query(
        "SELECT * FROM actual_load", conn, parse_dates=["timestamp"], index_col="timestamp"
    )

    # Merge on timestamp
    df = gen_df.join(load_df, how="inner")

    # Calculate Total Renewables
    df["total_renewable"] = df["wind_onshore"] + df["wind_offshore"] + df["solar"]

    # Calculate Penetration Ratio
    df["renewable_penetration"] = df["total_renewable"] / df["load_mw"]

    # Calculate Residual Load (Demand that must be covered by thermal/gas plants)
    df["residual_load"] = df["load_mw"] - df["total_renewable"]

    conn.close()
    return df[["total_renewable", "renewable_penetration", "residual_load"]]
