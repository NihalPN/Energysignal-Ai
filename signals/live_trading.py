import pandas as pd
import sqlite3
import os
import xgboost as xgb

# Paths
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database', 'energy_market.db')

def generate_live_signals():
    print("Loading latest market data...")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM master_features", conn, parse_dates=['timestamp'], index_col='timestamp')
    conn.close()

    df = df.sort_index()

    # THE FIX: Dynamically identify the latest available day in the database
    latest_timestamp = df.index.max()
    latest_day = latest_timestamp.floor('D')
    
    print(f"Latest market data found for: {latest_day.strftime('%Y-%m-%d')}")

    historical_df = df.dropna(subset=['target_price_24h_ahead']).copy()
    
    # Strictly isolate the most recent 24-hour calendar day
    live_df = df[df.index >= latest_day].copy()

    X_train = historical_df.drop(columns=['target_price_24h_ahead'])
    y_train = historical_df['target_price_24h_ahead']

    print("Training Production XGBoost model on all available historical data...")
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, objective='reg:squarederror')
    model.fit(X_train, y_train)

    X_live = live_df.drop(columns=['target_price_24h_ahead'])
    
    # Calculate what day we are actually predicting
    prediction_day = latest_day + pd.Timedelta(days=1)
    print(f"Predicting prices for tomorrow: {prediction_day.strftime('%Y-%m-%d')}...")
    
    live_df['predicted_price_tomorrow'] = model.predict(X_live)

    print("\n==================================================")
    print(f"       LIVE TRADING SIGNALS FOR {prediction_day.strftime('%Y-%m-%d')}          ")
    print("==================================================")
    
    EXPECTED_PROFIT_MARGIN = 40.0  
    trades_found = 0
    
    for timestamp, row in live_df.iterrows():
        # The exact 15-minute delivery block for the trade
        target_delivery_time = timestamp + pd.Timedelta(hours=24)
        
        current_price = row['price_eur_mwh']
        predicted_price = row['predicted_price_tomorrow']
        
        if (predicted_price > (current_price + EXPECTED_PROFIT_MARGIN)) and (current_price > 0):
            print(f" SIGNAL: BUY 10 MWh for delivery at {target_delivery_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"    Baseline Entry Price: €{current_price:.2f} | Predicted Exit Price: €{predicted_price:.2f}")
            print(f"    Expected Gross Profit: €{(predicted_price - current_price) * 10:.2f}\n")
            trades_found += 1

    if trades_found == 0:
        print(f"\nNo high-conviction trades detected for {prediction_day.strftime('%Y-%m-%d')}.")
        print("STATUS: Preserving capital.")

if __name__ == "__main__":
    generate_live_signals()
