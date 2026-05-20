import sys
import os
import sqlite3
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

# Ensure Python can find your folders
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

print("========================================")
print(" ⚡ ENERGY SIGNAL - API & DATA TESTER ⚡ ")
print("========================================\n")

# --- TEST 1: THE DATABASE & ENTSO-E ---
print("--- 1. Testing Database (Did ENTSO-E data save?) ---")
try:
    conn = sqlite3.connect("database/energy_market.db")

    # Get the 5 most recent rows
    df = pd.read_sql_query(
        "SELECT timestamp, price_eur_mwh, total_renewable FROM master_features ORDER BY timestamp DESC LIMIT 5",
        conn,
    )

    if df.empty:
        print("❌ Database connected, but the 'master_features' table is EMPTY.")
    else:
        print("✅ Database read successfully. Most recent 5 rows:")
        print(df.to_string(index=False))

        # Test the timezone logic from the dashboard
        berlin_tz = ZoneInfo("Europe/Berlin")
        today_str = datetime.now(berlin_tz).strftime("%Y-%m-%d")
        print(f"\nLooking for today's date: {today_str}")

        # Check if today's date actually exists in those recent rows
        if df["timestamp"].str.contains(today_str).any():
            print("✅ SUCCESS: Today's data IS in the database!")
        else:
            print(
                f"❌ FAILURE: Data stops before {today_str}. The GitHub Action is NOT pulling live data."
            )

    conn.close()
except Exception as e:
    print(f"❌ Database Error: {e}")


print("\n--- 2. Testing LLM API ---")
try:
    # We will try to import your functions exactly as the dashboard does
    from llm.rag_analyst import analyze_market_condition
    from llm.llm_classifier import fetch_live_german_energy_news

    print("✅ LLM Modules Imported Successfully.")

    print("Fetching live news...")
    news = fetch_live_german_energy_news()
    print(f"News Result: {news}")

    print("\nTesting AI Analyst generation (this takes a few seconds)...")
    test_scenario = (
        "Current Market Reality: The price is 75.00 EUR/MWh. Renewable generation is 40000.00 MW."
    )
    analysis = analyze_market_condition(test_scenario)

    print(f"✅ AI Response Received:\n{analysis}")

except ImportError as e:
    print(f"❌ IMPORT ERROR: {e}")
    print("Check your requirements.txt or folder structure.")
except Exception as e:
    print(f"❌ API ERROR: {e}")
    print("Check your API keys (Groq/OpenAI) or network connection.")
