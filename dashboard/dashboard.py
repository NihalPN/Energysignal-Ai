import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import os
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="EnergySignal AI Terminal", layout="wide", page_icon="⚡")
DB_PATH = 'database/energy_market.db'
MODEL_PATH = 'models/xgb_baseline.json'
ANALYSIS_PATH = 'models/latest_analysis.json'

# --- 2. SHARED DATA LOGIC ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    if os.path.exists(MODEL_PATH):
        model.load_model(MODEL_PATH)
    return model

@st.cache_data(ttl=60) # Refreshes every 60 seconds automatically
def fetch_master_features():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM master_features", conn, index_col='timestamp', parse_dates=['timestamp'])
    conn.close()
    return df.sort_index()

def color_profit(val):
    """Pandas styler to color table cells like PyQt"""
    if isinstance(val, str) and "INVEST" in val:
        return 'color: #00ff00; font-weight: bold;'
    elif isinstance(val, (int, float)) and val < 0:
        return 'color: #ff4444;'
    return ''

# --- 3. MAIN UI ---
st.title("⚡ EnergySignal AI - Institutional Terminal")

df_full = fetch_master_features()
model = load_model()

if df_full.empty:
    st.warning("Database is empty or missing. Please run the data pipeline.")
else:
    # --- MATH & PREDICTIONS (Same as PyQt DataWorker) ---
    history_df = df_full.tail(672) # Last 7 days
    latest_price = float(history_df['price_eur_mwh'].tail(1).item())
    
    live_df = df_full.tail(96)
    # Drop target for inference
    X_live = live_df.drop(columns=['target_price_24h_ahead'], errors='ignore') 
    predictions = model.predict(X_live)
    target_price_24h_now = float(predictions[-1])
    
    EXPECTED_MARGIN = 40.0
    current_spread = target_price_24h_now - latest_price
    
    if current_spread > EXPECTED_MARGIN and latest_price > 0:
        signal_text = "🟢 BUY 10 MWh"
    else:
        signal_text = "🟠 PRESERVE CAPITAL"

    # --- TABS LAYOUT ---
    tab_strategy, tab_preds, tab_live = st.tabs(["Strategy & AI Analyst", "XGBoost 24h Forecasts", "Live Market Monitor"])

    # === TAB 1: STRATEGY & AI ===
    with tab_strategy:
        # KPIs
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Last Cleared Price", value=f"€{latest_price:.2f}")
        with col2:
            st.metric(label="XGBoost 24h Forecast", value=f"€{target_price_24h_now:.2f}", delta=f"{current_spread:.2f} Spread")
        with col3:
            st.metric(label="Algorithmic Signal", value=signal_text)
            
        st.divider()
        
        # Chart & AI Layout
        chart_col, ai_col = st.columns([2, 1]) # Chart takes 2/3 space, AI takes 1/3
        
        with chart_col:
            st.subheader("Market Timeline (7 Days + 24h Forecast)")
            # Combine history and future for a seamless chart
            hist_series = history_df['price_eur_mwh']
            future_dates = [hist_series.index[-1] + timedelta(minutes=15 * i) for i in range(1, 97)]
            future_series = pd.Series(predictions, index=future_dates)
            
            chart_df = pd.DataFrame({"Historical": hist_series, "Forecast": future_series})
            st.line_chart(chart_df, color=["#00d2ff", "#ffaa00"]) # Blue history, Orange future
            
        with ai_col:
            st.subheader("🤖 RAG Market Analyst")
            if os.path.exists(ANALYSIS_PATH):
                with open(ANALYSIS_PATH, 'r') as f:
                    ai_data = json.load(f)
                st.caption(f"Last updated: {ai_data.get('timestamp', 'Unknown')}")
                st.markdown(f"> {ai_data.get('analysis', '')}")
            else:
                st.info("Waiting for background worker to generate analysis...")

    # === TAB 2: PREDICTIONS ===
    with tab_preds:
        st.subheader("Model Predictions & Profit Opportunities (Next 24 Hours)")
        
        # Build the dataframe like the PyQt Table
        pred_data = []
        hist_length = len(history_df)
        
        for i, pred_price in enumerate(predictions):
            block_start = history_df.index[-1] + timedelta(minutes=15 * (i + 1))
            today_price_index = hist_length - 96 + i
            if today_price_index >= 0:
                today_price = float(history_df['price_eur_mwh'].iloc[today_price_index])
                expected_profit = float(pred_price) - today_price
                
                status = f"€{pred_price:.2f}"
                if expected_profit > EXPECTED_MARGIN and today_price > 0:
                    status = f"€{pred_price:.2f} | INVEST (+€{expected_profit:.2f})"
                
                pred_data.append({
                    "Delivery Time Block": block_start.strftime('%Y-%m-%d %H:%M'),
                    "Predicted Price (€/MWh)": status
                })
        
        pred_df = pd.DataFrame(pred_data)
        # Apply the color styling and display
        st.dataframe(pred_df.style.map(color_profit, subset=["Predicted Price (€/MWh)"]), use_container_width=True, hide_index=True)

    # === TAB 3: LIVE MARKET ===
    with tab_live:
        st.subheader("Today's Clearing Prices")
        # Ensure you are using the same timezone as your data pipeline
        berlin_tz = ZoneInfo("Europe/Berlin")
        today_str = datetime.now(berlin_tz).strftime('%Y-%m-%d')
        today_df = df_full[df_full.index.astype(str).str.startswith(today_str)].copy()
        
        if not today_df.empty:
            display_df = today_df[['price_eur_mwh']].reset_index()
            display_df.columns = ["Time Block", "Clearing Price (€/MWh)"]
            display_df['Time Block'] = display_df['Time Block'].dt.strftime('%H:%M')
            
            st.dataframe(display_df.style.map(color_profit, subset=["Clearing Price (€/MWh)"]), use_container_width=True, hide_index=True)
        else:
            st.info("No prices cleared for today yet.")
