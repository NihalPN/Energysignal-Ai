import os
import sqlite3
from sqlalchemy import create_engine

DB_PATH = os.path.join(os.path.dirname(__file__), "energy_market.db")
engine = create_engine(f"sqlite:///{DB_PATH}")


def init_db():
    """Initialize the SQLite database with strict 15-min schemas."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS day_ahead_prices (
        timestamp TEXT PRIMARY KEY,
        price_eur_mwh REAL
    )"""
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS generation_mix (
        timestamp TEXT PRIMARY KEY,
        wind_onshore REAL,
        wind_offshore REAL,
        solar REAL,
        nuclear REAL,
        fossil_gas REAL,
        fossil_hard_coal REAL
    )"""
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS actual_load (
        timestamp TEXT PRIMARY KEY,
        load_mw REAL
    )"""
    )

    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS weather_data (
        timestamp TEXT PRIMARY KEY,
        temperature_2m REAL,
        wind_speed_10m REAL,
        solar_irradiance REAL
    )"""
    )

    conn.commit()
    conn.close()
    print(f"Database schema initialized successfully at {DB_PATH}")


if __name__ == "__main__":
    init_db()
