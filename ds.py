import os
import pandas as pd
from entsoe import EntsoePandasClient
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()
API_KEY = os.getenv("ENTSOE_API_KEY")

if not API_KEY:
    print("❌ ERROR: ENTSOE_API_KEY is missing from your .env file!")
    exit()

# Print a masked version of the key just to be sure it's loading correctly
masked_key = f"{API_KEY[:4]}...{API_KEY[-4:]}" if len(API_KEY) > 8 else "INVALID_LENGTH"
print(f"Loaded API Key: {masked_key}")
print("Pinging ENTSO-E servers...\n")

try:
    client = EntsoePandasClient(api_key=API_KEY)

    # Set our time boundaries
    tz = "Europe/Berlin"
    now = pd.Timestamp.now(tz=tz)
    yesterday = now - pd.Timedelta(days=1)
    tomorrow = now.floor("D") + pd.Timedelta(days=1)

    # 1. Test Historical/Live Data (Actual Load up to NOW)
    print("1. Testing 'Actual Load' endpoint (Past 24 hours up to NOW)...")
    load = client.query_load("DE_LU", start=yesterday, end=now)
    print(f"   ✅ Success! Fetched {len(load)} rows.")
    print(f"   🕒 Latest data point received: {load.index.max()}")

    # 2. Test Future Data (Day-Ahead Prices up to TOMORROW)
    print("\n2. Testing 'Day-Ahead Prices' endpoint (Future data for TOMORROW)...")
    prices = client.query_day_ahead_prices("DE_LU", start=now.floor("D"), end=tomorrow)
    print(f"   ✅ Success! Fetched {len(prices)} rows.")
    print(f"   🔮 Latest future price received: {prices.index.max()}")

    print("\n🎉 ALL TESTS PASSED! Your API key is 100% active and working perfectly.")

except Exception as e:
    error_msg = str(e).lower()
    print(f"\n❌ API CONNECTION FAILED:")
    print(f"Error Details: {e}\n")

    if "401" in error_msg or "unauthorized" in error_msg:
        print("💡 DIAGNOSIS: Your API key is invalid, expired, or hasn't been approved yet.")
        print("   Fix: Log into the ENTSO-E Transparency Platform and generate a new key.")
    elif "429" in error_msg or "rate" in error_msg:
        print("💡 DIAGNOSIS: Your key is valid, but you are in a temporary rate-limit timeout.")
        print("   Fix: Wait about 15-30 minutes for the servers to cool down, then try again.")
    else:
        print(
            "💡 DIAGNOSIS: The ENTSO-E servers might be down or missing data for this specific minute."
        )
