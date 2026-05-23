import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from entsoe import EntsoePandasClient

# Configuration
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")
DB_PATH = "energy_data.db"
COUNTRY_CODE = "DE_LU" # German Day-Ahead Market

def get_last_timestamp(cursor):
    """Fetches the most recent timestamp currently in the database."""
    try:
        # Adjust 'prices' and 'timestamp' to match your actual table/column names
        cursor.execute("SELECT MAX(timestamp) FROM prices")
        result = cursor.fetchone()[0]
        if result:
            return pd.to_datetime(result)
    except sqlite3.OperationalError:
        pass # Table might not exist yet
    
    # If DB is empty, default to fetching the last 24 hours
    return pd.Timestamp.now(tz="Europe/Berlin") - pd.Timedelta(hours=24)

def backfill_data():
    if not ENTSOE_API_KEY:
        raise ValueError("ENTSOE_API_KEY environment variable is missing.")

    client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Identify the gap
    start_time = get_last_timestamp(cursor)
    end_time = pd.Timestamp.now(tz="Europe/Berlin")

    # Add a small buffer to start_time to catch overlapping MTUs
    start_time = start_time - pd.Timedelta(minutes=15) 

    print(f"Checking for missing data between {start_time} and {end_time}...")

    # 2. Fetch missing window from ENTSO-E
    try:
        # query_day_ahead_prices returns a Pandas Series
        ts_data = client.query_day_ahead_prices(COUNTRY_CODE, start=start_time, end=end_time)
        
        if ts_data.empty:
            print("No new data found.")
            return

        # Convert Series to DataFrame for SQLite insertion
        df = ts_data.reset_index()
        df.columns = ["timestamp", "price"]
        
        # 3. Clean the data (Drop any NaNs caused by ENTSO-E API drops)
        df = df.dropna(subset=["price"])

        # 4. Insert into SQLite using UPSERT to prevent duplicates
        # Requires SQLite 3.24+ 
        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO prices (timestamp, price) 
                VALUES (?, ?)
                ON CONFLICT(timestamp) DO UPDATE SET price=excluded.price;
            """, (row["timestamp"].strftime("%Y-%m-%d %H:%M:%S"), row["price"]))
        
        conn.commit()
        print(f"Successfully backfilled {len(df)} rows.")

    except Exception as e:
        print(f"ENTSO-E API Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    backfill_data()
