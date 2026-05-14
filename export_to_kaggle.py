import pandas as pd
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database', 'energy_market.db')

def export_data():
    print("Connecting to database...")
    conn = sqlite3.connect(DB_PATH)
    
    print("Extracting master features...")
    df = pd.read_sql_query("SELECT * FROM master_features", conn, index_col='timestamp')
    
    export_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kaggle_master_features.csv')
    df.to_csv(export_path)
    
    print(f"Data successfully exported to: {export_path}")
    conn.close()

if __name__ == "__main__":
    export_data()
