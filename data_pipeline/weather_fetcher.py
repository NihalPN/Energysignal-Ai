import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.schema import engine

logging.basicConfig(filename="logs/pipeline.log", level=logging.INFO)

cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

LAT = 52.5200
LON = 13.4050


def safe_insert(df, table_name):
    """Institutional-grade insert: checks DB, drops DST duplicates, and prevents overlapping timestamps."""
    if df.empty:
        return

    # THE DST FIX: Drop duplicate timestamps within the incoming data itself
    df = df[~df.index.duplicated(keep="first")]

    min_ts = df.index.min()
    max_ts = df.index.max()
    existing_df = pd.read_sql(
        f"SELECT timestamp FROM {table_name} WHERE timestamp >= '{min_ts}' AND timestamp <= '{max_ts}'",
        con=engine,
    )
    df_new = df[~df.index.isin(existing_df["timestamp"])]
    if not df_new.empty:
        df_new.to_sql(table_name, con=engine, if_exists="append", index_label="timestamp")
        logging.info(f"Inserted {len(df_new)} new rows into {table_name}.")


def fetch_and_store_weather(start_date: str, end_date: str):
    logging.info(f"Fetching Open-Meteo data from {start_date} to {end_date}")

    # THE FIX: Point to the dedicated Historical Forecast API
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date,
        "end_date": end_date,
        "minutely_15": ["temperature_2m", "wind_speed_10m", "shortwave_radiation"],
        "models": "icon_d2",  # Explicitly request the 15-min German DWD model
        "timezone": "Europe/Berlin",
    }

    try:
        responses = openmeteo.weather_api(url, params=params)

        # Using int("0") to safely extract the first location
        first_item_index = int("0")
        response = responses[first_item_index]

        minutely_15 = response.Minutely15()
        minutely_15_data = {
            "timestamp": pd.date_range(
                start=pd.to_datetime(minutely_15.Time(), unit="s", utc=True).tz_convert(
                    "Europe/Berlin"
                ),
                end=pd.to_datetime(minutely_15.TimeEnd(), unit="s", utc=True).tz_convert(
                    "Europe/Berlin"
                ),
                freq=pd.Timedelta(seconds=minutely_15.Interval()),
                inclusive="left",
            )
        }

        minutely_15_data["temperature_2m"] = minutely_15.Variables(0).ValuesAsNumpy()
        minutely_15_data["wind_speed_10m"] = minutely_15.Variables(1).ValuesAsNumpy()
        minutely_15_data["solar_irradiance"] = minutely_15.Variables(2).ValuesAsNumpy()

        df_weather = pd.DataFrame(data=minutely_15_data)
        df_weather["timestamp"] = df_weather["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        df_weather.set_index("timestamp", inplace=True)

        safe_insert(df_weather, "weather_data")
        print(f"Weather chunk {start_date} to {end_date} safely stored.")

    except Exception as e:
        logging.error(f"Weather Fetcher failed: {e}")
        print(f"Error fetching weather data: {e}")
        raise
