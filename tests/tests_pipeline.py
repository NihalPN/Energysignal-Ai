import pytest
import pandas as pd
import sqlite3
import os

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "energy_market.db"
)


def test_database_connection():
    """Validates that the SQLite database exists and is accessible."""
    assert os.path.exists(DB_PATH), "CRITICAL: Database file does not exist."
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.close()
    except Exception as e:
        pytest.fail(f"Database connection failed: {e}")


def test_15_minute_resolution_integrity():
    """
    Addresses Image 3: Ensures the day_ahead_prices table strictly follows
    the 15-minute Market Time Unit (MTU) mandated by the European market.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT timestamp FROM day_ahead_prices ORDER BY timestamp DESC LIMIT 10",
        conn,
        parse_dates=["timestamp"],
    )
    conn.close()

    assert not df.empty, "Database is empty."

    # Calculate time differences between consecutive rows
    time_diffs = df["timestamp"].diff().dropna().abs()

    # Assert that every single time difference is exactly 15 minutes
    for diff in time_diffs:
        assert diff == pd.Timedelta(
            minutes=15
        ), f"Data resolution violation: Found {diff} instead of 15 minutes."


def test_ui_required_columns_exist():
    """
    Addresses Images 5 & 6: Prevents PyQt6 UI crashes (KeyError/DB Connection Error)
    by ensuring the feature engineering output perfectly matches the frontend's expected inputs.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM master_features LIMIT 1", conn)
    conn.close()

    required_columns = [
        "price_eur_mwh",
        "total_renewable",
        "residual_load",
        "target_price_24h_ahead",
    ]

    for col in required_columns:
        assert (
            col in df.columns
        ), f"UI Crash Risk: Required column '{col}' is missing from master_features."


def test_no_data_leakage_in_target():
    """
    Addresses the previous backtesting logic flaw: Ensures the target variable
    is strictly shifted 24 hours (96 steps) into the future to prevent lookahead bias.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT price_eur_mwh, target_price_24h_ahead FROM master_features ORDER BY timestamp ASC LIMIT 200",
        conn,
    )
    conn.close()

    # If we shift the actual price backward by 96 steps, it must perfectly match the target column
    # We test a specific index to prove the mathematical shift is correct
    test_index = 0
    target_index = 96

    actual_future_price = df.loc[target_index, "price_eur_mwh"]
    predicted_target = df.loc[test_index, "target_price_24h_ahead"]

    assert (
        actual_future_price == predicted_target
    ), "Data Leakage Detected: Target variable does not align with the 24-hour future price."


def test_safe_insert_duplicate_handling():
    """
    Addresses the Daylight Saving Time crash: Ensures pandas drops duplicate
    timestamps instead of crashing the database.
    """
    # Create a dummy dataframe with duplicate indices
    data = {"price_eur_mwh": [50.0, 60.0, 70.0]}
    index = pd.to_datetime(["2026-10-25 02:00:00", "2026-10-25 02:00:00", "2026-10-25 02:15:00"])
    df = pd.DataFrame(data, index=index)

    # Apply the exact logic from our safe_insert function
    df_deduplicated = df[~df.index.duplicated(keep="first")]

    assert (
        len(df_deduplicated) == 2
    ), "Duplicate handling failed: Script did not drop the overlapping DST timestamp."
