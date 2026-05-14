import pandas as pd
import time
from entsoe_fetcher import fetch_and_store_entsoe_data
from weather_fetcher import fetch_and_store_weather

def backfill_data_in_chunks(total_days=730, chunk_size=3):
    print(f"Starting backfill for the last {total_days} days in chunks of {chunk_size} days...")
    tz = 'Europe/Berlin'
    
    final_end_time = pd.Timestamp.now(tz=tz).floor('D') + pd.Timedelta(days=1)
    absolute_start_time = final_end_time - pd.Timedelta(days=total_days)
    
    current_start = absolute_start_time
    
    while current_start < final_end_time:
        current_end = min(current_start + pd.Timedelta(days=chunk_size), final_end_time)
        
        start_str_entsoe = current_start.strftime('%Y%m%d')
        end_str_entsoe = current_end.strftime('%Y%m%d')
        start_str_weather = current_start.strftime('%Y-%m-%d')
        end_str_weather = current_end.strftime('%Y-%m-%d')
        
        print(f"\n--- Fetching chunk: {start_str_entsoe} to {end_str_entsoe} ---")
        
        try:
            print("Fetching ENTSO-E data...")
            fetch_and_store_entsoe_data(start_str_entsoe, end_str_entsoe)
            
            print("Fetching Open-Meteo data...")
            fetch_and_store_weather(start_str_weather, end_str_weather)
            
        except Exception as e:
            print(f"\nCRITICAL: Chunk failed after 5 retries. Error: {e}")
            print("Server is completely down. Wait an hour and run again.")
            return 
        
        current_start = current_end
        
        print("Sleeping for 5 seconds to respect API rate limits...")
        time.sleep(5)
        
    print("\nBackfill complete! Your database is primed.")

if __name__ == "__main__":
    backfill_data_in_chunks(total_days=730, chunk_size=3)
