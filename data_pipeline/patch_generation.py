import pandas as pd
import time
from entsoe import EntsoePandasClient
import os
from dotenv import load_dotenv
import sys
from sqlalchemy import create_engine

# Force Python to read the .env file locally
load_dotenv()

# Setup Paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Grab credentials from .env
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY")
DB_URL = os.getenv("DATABASE_URL")

if not ENTSOE_API_KEY or not DB_URL:
    raise ValueError("Missing ENTSOE_API_KEY or DATABASE_URL in environment variables.")

client = EntsoePandasClient(api_key=ENTSOE_API_KEY)
engine = create_engine(DB_URL)
TZ = "Europe/Berlin"


def cloud_insert(df, table_name):
    """Pushes the dataframe directly to Neon Serverless Postgres."""
    try:
        # Pushing to the cloud in chunks to avoid overwhelming the connection
        df.to_sql(
            table_name,
            engine,
            if_exists="append",
            index=True,
            index_label="timestamp",
            chunksize=500,
        )
        print(f"✅ Successfully inserted {len(df)} rows into {table_name}")
    except Exception as e:
        print(f"❌ Database Insert Error: {e}")


def patch_generation():
    print("Patching 730 days of Generation Data using Country Code 'DE' to Cloud DB...")

    final_end_time = pd.Timestamp.now(tz=TZ).floor("D") + pd.Timedelta(days=1)
    current_start = final_end_time - pd.Timedelta(days=730)

    while current_start < final_end_time:
        current_end = min(current_start + pd.Timedelta(days=3), final_end_time)

        print(
            f"Fetching Generation: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}"
        )

        try:
            # Note: We use 'DE' here, not 'DE_LU'
            generation = client.query_generation("DE", start=current_start, end=current_end)

            # ENTSO-E sometimes returns a MultiIndex (e.g. 'Wind Onshore', 'Actual Aggregated')
            if isinstance(generation.columns, pd.MultiIndex):
                first_level = 0
                generation.columns = generation.columns.get_level_values(first_level)

            # Drop duplicated columns (e.g. Hydro Pumped Storage returning twice)
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

            # Keep index as datetime object for SQLAlchemy, just remove tz info if Postgres expects naive UTC
            if df_gen.index.tz is not None:
                df_gen.index = df_gen.index.tz_convert("UTC").tz_localize(None)

            cloud_insert(df_gen, "generation_mix")

        except Exception as e:
            print(f"Error fetching chunk: {e}")

        current_start = current_end
        # Be polite to the ENTSO-E API so they don't block your IP
        time.sleep(4)

    print("Cloud Generation patch complete!")


if __name__ == "__main__":
    patch_generation()
