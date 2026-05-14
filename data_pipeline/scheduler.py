import pandas as pd
from entsoe_fetcher import fetch_and_store_entsoe_data
from weather_fetcher import fetch_and_store_weather
import logging
import os

log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'pipeline.log')
logging.basicConfig(filename=log_path, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def run_daily_pipeline():
    tz = 'Europe/Berlin'
    today = pd.Timestamp.now(tz=tz).floor('D')
    
    # We want data from yesterday up to the end of TOMORROW (since tomorrow's auction clears today at noon)
    yesterday = today - pd.Timedelta(days=1)
    end_boundary = today + pd.Timedelta(days=2) 
    
    start_str_entsoe = yesterday.strftime('%Y%m%d')
    end_str_entsoe = end_boundary.strftime('%Y%m%d')
    start_str_weather = yesterday.strftime('%Y-%m-%d')
    end_str_weather = end_boundary.strftime('%Y-%m-%d')
    
    logging.info("=== STARTING DAILY PIPELINE RUN ===")
    try:
        fetch_and_store_entsoe_data(start_str_entsoe, end_str_entsoe)
        fetch_and_store_weather(start_str_weather, end_str_weather)
        logging.info("=== PIPELINE RUN COMPLETE ===")
        print(f"Pipeline successfully fetched data up to {end_boundary.strftime('%Y-%m-%d')}")
    except Exception as e:
        logging.critical(f"PIPELINE CRASHED: {e}")
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    run_daily_pipeline()
