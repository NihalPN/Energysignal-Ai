import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

# --- CLOUD DATABASE SETUP ---
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("❌ CRITICAL: Missing DATABASE_URL in environment variables.")

# Create the SQLAlchemy engine for Neon Postgres
engine = create_engine(DB_URL)


def init_db():
    """Initialize the Postgres database with strict 15-min schemas."""

    print("Connecting to Neon Postgres...")

    # Using engine.begin() automatically commits the transaction upon success
    with engine.begin() as conn:

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS day_ahead_prices (
            timestamp TIMESTAMP PRIMARY KEY,
            price_eur_mwh REAL
        )"""))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS generation_mix (
            timestamp TIMESTAMP PRIMARY KEY,
            wind_onshore REAL,
            wind_offshore REAL,
            solar REAL,
            nuclear REAL,
            fossil_gas REAL,
            fossil_hard_coal REAL
        )"""))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS actual_load (
            timestamp TIMESTAMP PRIMARY KEY,
            load_mw REAL
        )"""))

        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS weather_data (
            timestamp TIMESTAMP PRIMARY KEY,
            temperature_2m REAL,
            wind_speed_10m REAL,
            solar_irradiance REAL
        )"""))

    print("✅ Database schema initialized successfully on Neon Postgres!")


if __name__ == "__main__":
    init_db()
