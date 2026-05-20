import pandas as pd
import time
from entsoe import EntsoePandasClient
import os
from dotenv import load_dotenv
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.schema import engine
from data_pipeline.entsoe_fetcher import safe_insert

load_dotenv()
client = EntsoePandasClient(api_key=os.getenv("ENTSOE_API_KEY"))
TZ = "Europe/Berlin"


def patch_generation():
    print("Patching 30 days of Generation Data using Country Code 'DE'...")

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
                # Using int("0") to prevent the AI formatter from deleting the index bracket
                first_level = int("0")
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
            df_gen.index = df_gen.index.strftime("%Y-%m-%d %H:%M:%S")

            safe_insert(df_gen, "generation_mix")

        except Exception as e:
            print(f"Error fetching chunk: {e}")

        current_start = current_end
        time.sleep(4)

    print("Generation patch complete!")


if __name__ == "__main__":
    patch_generation()
