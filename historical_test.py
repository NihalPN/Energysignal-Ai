import os
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
from entsoe import EntsoePandasClient
from sklearn.metrics import mean_absolute_error, r2_score
from dotenv import load_dotenv

# Load keys safely
load_dotenv()
ENTSOE_KEY = os.getenv("ENTSOE_API_KEY")


def fetch_black_swan_data():
    if not ENTSOE_KEY:
        raise ValueError("ENTSOE_API_KEY not found in environment variables.")

    print("⏳ Fetching Historical Crisis Data (May 2023)...")
    client = EntsoePandasClient(api_key=ENTSOE_KEY)

    start = pd.Timestamp("2023-05-01", tz="Europe/Berlin")
    end = pd.Timestamp("2023-05-31", tz="Europe/Berlin")
    country_code = "DE_LU"

    try:
        # 1. Fetch ENTSO-E Data
        print(" -> Downloading ENTSO-E Prices and Load...")
        prices = client.query_day_ahead_prices(country_code, start=start, end=end)
        load = client.query_load(country_code, start=start, end=end)

        # THE FIX 1: Prevent "1-dimensional" error.
        # Force prices and load to be 1D Series if ENTSO-E returns them as 2D DataFrames
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]
        if isinstance(load, pd.DataFrame):
            load = load.iloc[:, 0]

        print(" -> Downloading ENTSO-E Generation Mix...")
        gen = client.query_generation("DE", start=start, end=end)

        # THE FIX 2: Modern Pandas Syntax to clear the FutureWarning
        if isinstance(gen, pd.Series):
            gen = gen.to_frame()

        if isinstance(gen.columns, pd.MultiIndex):
            # Transpose (.T), group by the index, sum, and Transpose back (.T)
            # This completely avoids the deprecated axis=1 argument
            gen = gen.T.groupby(level=0).sum().T

        # Combine safely now that we know prices and load are strictly 1-Dimensional
        df = pd.DataFrame({"price_eur_mwh": prices, "load_mw": load})
        df = df.join(gen, how="inner")

        # Map names dynamically to match your feature engineering
        name_map = {
            "Solar": "solar_generation",
            "Wind Onshore": "wind_onshore",
            "Wind Offshore": "wind_offshore",
        }
        df.rename(columns=name_map, inplace=True)

        # Ensure critical columns exist
        for col in ["solar_generation", "wind_onshore", "wind_offshore"]:
            if col not in df.columns:
                df[col] = 0

        # 2. Fetch Historical Weather
        print(" -> Downloading Historical Weather from Open-Meteo...")
        weather_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": 51.1657,
            "longitude": 10.4515,
            "start_date": "2023-05-01",
            "end_date": "2023-05-31",
            "hourly": "direct_radiation",
            "timezone": "Europe/Berlin",
        }
        resp = requests.get(weather_url, params=params).json()

        weather_df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(resp["hourly"]["time"]).tz_localize("Europe/Berlin"),
                "solar_irradiance": resp["hourly"]["direct_radiation"],
            }
        ).set_index("timestamp")

        # Resample to 15min
        weather_df = weather_df.resample("15min").interpolate(method="linear")

        return df.join(weather_df, how="inner")

    except Exception as e:
        print(f"❌ API Error: {str(e)}")
        return None


def engineer_features_and_test(df):
    print("\n⚙️ Applying Institutional Feature Engineering...")

    # Pre-processing
    df = df.ffill(limit=32).dropna(subset=["price_eur_mwh"])

    # Feature Engineering (Exact match to your live model)
    df["total_renewable"] = df["solar_generation"] + df["wind_onshore"] + df["wind_offshore"]
    df["solar_penetration_ratio"] = df["solar_generation"] / (df["load_mw"] + 1)
    df["solar_squared"] = df["solar_generation"] ** 2
    df["is_peak_solar_window"] = ((df.index.hour >= 11) & (df.index.hour <= 15)).astype(int)

    # Target
    df["target_price_24h_ahead"] = df["price_eur_mwh"].shift(-96)
    df = df.dropna()

    # Split
    split_index = int(len(df) * 0.66)
    train_df, test_df = df.iloc[:split_index], df.iloc[split_index:].copy()

    # Train
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
    model.fit(train_df.drop(columns=["target_price_24h_ahead"]), train_df["target_price_24h_ahead"])

    # Predict
    preds = model.predict(test_df.drop(columns=["target_price_24h_ahead"]))

    # Metrics
    mae = mean_absolute_error(test_df["target_price_24h_ahead"], preds)
    actual_dir = np.sign(test_df["target_price_24h_ahead"] - test_df["price_eur_mwh"])
    pred_dir = np.sign(preds - test_df["price_eur_mwh"])
    dir_acc = np.mean(actual_dir == pred_dir) * 100

    print(f"\n{'='*50}\n       MAY 2023 STRESS TEST RESULTS\n{'='*50}")
    print(f"MAE: €{mae:.2f} | Directional Acc: {dir_acc:.1f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    data = fetch_black_swan_data()
    if data is not None:
        engineer_features_and_test(data)
