import sqlite3
import pandas as pd

# Connect to your database
conn = sqlite3.connect("database/energy_market.db")

# Run the query and load it into a dataframe
query = "SELECT timestamp, price_eur_mwh FROM master_features ORDER BY timestamp DESC LIMIT 5"
df = pd.read_sql_query(query, conn)

print(df)
conn.close()
