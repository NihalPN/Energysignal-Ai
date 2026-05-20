import os
import pandas as pd
from entsoe import EntsoePandasClient
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt
import logging
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.schema import engine

load_dotenv()
logging.basicConfig(
    filename="logs/pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

API_KEY = os.getenv("ENTSOE_API_KEY")
if not API_KEY:
    raise ValueError("Missing ENTSOE_API_KEY in.env file")

client = EntsoePandasClient(api_key=API_KEY)
TZ = "Europe/Berlin"
COUNTRY_CODE = "10Y1001A1001A82H"


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


@retry(wait=wait_exponential(multiplier=2, min=2, max=30), stop=stop_after_attempt(5))

def fetch_and_store_entsoe_data(start_date: str, end_date: str):
    start = pd.Timestamp(start_date, tz=TZ)
    end = pd.Timestamp(end_date, tz=TZ)
    
    # NEW: Define a strict cutoff for "Actuals" so we don't ask for the future
    now_berlin = pd.Timestamp.now(tz=TZ)
    actuals_end = end if end < now_berlin else now_berlin

    try:
        # 1. Fetch Day-Ahead Prices (Uses DE_LU) - Can pull the future
        logging.info(f"Fetching DA prices from {start} to {end}")
        da_prices = client.query_day_ahead_prices("DE_LU", start=start, end=end)

        if isinstance(da_prices, pd.Series):
            df_prices = da_prices.to_frame(name="price_eur_mwh")
        else:
            df_prices = da_prices.copy()
            df_prices = df_prices.iloc[:, :1]
            df_prices.columns = ["price_eur_mwh"]

        df_prices = df_prices.resample("15min").ffill()
        df_prices.index = df_prices.index.strftime("%Y-%m-%d %H:%M:%S")
        safe_insert(df_prices, "day_ahead_prices")

        # 2. Fetch Actual Load (Uses DE_LU) - STRICTLY up to 'actuals_end'
        logging.info(f"Fetching actual load from {start} to {actuals_end}")
        load = client.query_load("DE_LU", start=start, end=actuals_end)

        if isinstance(load, pd.Series):
            df_load = load.to_frame(name="load_mw")
        else:
            df_load = load.copy()
            df_load = df_load.iloc[:, :1]
            df_load.columns = ["load_mw"]

        df_load = df_load.resample("15min").ffill()
        df_load.index = df_load.index.strftime("%Y-%m-%d %H:%M:%S")
        safe_insert(df_load, "actual_load")

        # 3. Fetch Generation by Type (Uses DE Country Code) - STRICTLY up to 'actuals_end'
        logging.info(f"Fetching generation mix from {start} to {actuals_end}")
        generation = client.query_generation("DE", start=start, end=actuals_end)

        if isinstance(generation.columns, pd.MultiIndex):
            first_level = int("0")
            generation.columns = generation.columns.get_level_values(first_level)
        generation = generation.loc[:, ~generation.columns.duplicated()]

        gen_mapping = {
            "Wind Onshore": "wind_onshore",
            "Wind Offshore": "wind_offshore",
            "Solar": "solar",
            "Nuclear": "nuclear",
            "Fossil Gas": "fossil_gas",
            "Fossil Hard coal": "fossil_hard_coal",
        }
        df_gen = generation.rename(columns=gen_mapping)
        cols_to_keep = [col for col in df_gen.columns if col in gen_mapping.values()]
        df_gen = df_gen[cols_to_keep]
        df_gen = df_gen.resample("15min").ffill()
        df_gen.index = df_gen.index.strftime("%Y-%m-%d %H:%M:%S")
        safe_insert(df_gen, "generation_mix")

        print(f"ENTSO-E chunk {start_date} to {end_date} safely stored.")

    except Exception as e:
        logging.error(f"ENTSO-E Fetcher failed for {start_date} to {end_date}: {e}")
        print(f"API Error. Tenacity will retry. Error: {e}")
        raise
