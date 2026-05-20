import sqlite3
import pandas as pd

DB_PATH = "database/energy_market.db"
conn = sqlite3.connect(DB_PATH)

print("=== RAW TABLE ROW COUNTS ===")
tables = ["day_ahead_prices", "generation_mix", "actual_load", "weather_data"]
for t in tables:
    # The fix:.iloc gets the first row and first column of the result
    count = pd.read_sql(f"SELECT COUNT(*) as c FROM {t}", conn).iloc
    print(f"{t}: {count} rows")

print("\n=== CHECKING WEATHER DATA FOR NaNs ===")
weather = pd.read_sql("SELECT * FROM weather_data", conn)
if len(weather) > 0:
    print(weather.isna().sum())
else:
    print("Weather table is completely empty!")

print("\n=== CHECKING GENERATION DATA FOR NaNs ===")
gen = pd.read_sql("SELECT * FROM generation_mix", conn)
if len(gen) > 0:
    print(gen.isna().sum())
else:
    print("Generation table is completely empty!")

conn.close()
