import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit.components.v1 as components

def live_berlin_clock():
    """Renders a live ticking clock in the browser using zero server resources."""
    clock_html = """
    <div style="display: flex; justify-content: flex-end; align-items: center; padding: 5px;">
        <div style="font-family: monospace; font-size: 1.2rem; font-weight: bold; color: #4CAF50; background-color: #1E1E1E; padding: 8px 15px; border-radius: 5px; border: 1px solid #333;">
            🇩🇪 Berlin: <span id="berlin-time"></span>
        </div>
    </div>

    <script>
    function updateClock() {
        const now = new Date();
        // Force the time to calculate based on the Europe/Berlin timezone
        const options = { 
            timeZone: 'Europe/Berlin', 
            hour: '2-digit', 
            minute: '2-digit', 
            second: '2-digit', 
            hour12: false 
        };
        const formatter = new Intl.DateTimeFormat('en-GB', options);
        document.getElementById('berlin-time').innerText = formatter.format(now);
    }
    // Update the clock every 1000 milliseconds (1 second)
    setInterval(updateClock, 1000);
    updateClock(); // Run immediately on load
    </script>
    """
    # Render the HTML component
    components.html(clock_html, height=60)




# --- LLM Imports ---
try:
    from llm.rag_analyst_cloud import analyze_market_condition
    from llm.llm_classifier import fetch_live_german_energy_news
except ImportError:
    st.error("Error importing LLM modules. Ensure 'llm/rag_analyst.py' and 'llm/llm_classifier.py' exist.")

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="EnergySignal AI Terminal", layout="wide", page_icon="⚡")
DB_PATH = 'database/energy_market.db'
MODEL_PATH = 'models/xgb_baseline.json'

# --- 2. SHARED DATA & AI LOGIC ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    if os.path.exists(MODEL_PATH):
        model.load_model(MODEL_PATH)
    return model

@st.cache_data(ttl=60) # Refreshes database reads every 60 seconds
def fetch_master_features():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM master_features", conn, index_col='timestamp', parse_dates=['timestamp'])
    conn.close()
    return df.sort_index()

@st.cache_data(ttl=3600, show_spinner="🤖 AI Analyst is reading the market...") # Runs only once per hour
def get_hourly_ai_analysis(latest_price, wind_generation, residual_load):
    try:
        live_news = fetch_live_german_energy_news()
        news_text = " | ".join(live_news) if live_news else "No major news today."
        
        actual_load = wind_generation + residual_load
        real_scenario = (f"Current Market Reality: The price is {latest_price:.2f} EUR/MWh. "
                         f"Renewable generation is {wind_generation:.2f} MW against a grid load of {actual_load:.2f} MW. "
                         f"Latest Live News: {news_text}")
                         
        analysis = analyze_market_condition(real_scenario)
        return analysis
    except Exception as e:
        return f"AI Analysis temporarily unavailable: {str(e)}"

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
    st.warning("Database is empty or missing. Please ensure your data pipeline has run and the database is pushed to GitHub.")
else:
    # --- MATH & PREDICTIONS ---
    history_df = df_full.tail(672) # Last 7 days
    latest_price = float(history_df['price_eur_mwh'].tail(1).item())
    
    live_df = df_full.tail(96)
    # Drop target for inference
    X_live = live_df.drop(columns=['target_price_24h_ahead'], errors='ignore') 
    
    # Ensure all required columns are present for XGBoost
    if 'price_eur_mwh' in model.feature_names_in_ and 'price_eur_mwh' not in X_live.columns:
        X_live['price_eur_mwh'] = history_df['price_eur_mwh'].tail(96).values

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
            st.line_chart(chart_df, color=["#00d2ff", "#ffaa00"]) 
            
        with ai_col:
            st.subheader("🤖 RAG Market Analyst")
            
            # Extract current grid physics for the LLM
            latest_wind = float(history_df['total_renewable'].tail(1).item()) if 'total_renewable' in history_df else 0.0
            latest_residual = float(history_df['residual_load'].tail(1).item()) if 'residual_load' in history_df else 0.0
            
            # Call the cached LLM function
            ai_text = get_hourly_ai_analysis(latest_price, latest_wind, latest_residual)
            st.markdown(f"> {ai_text}")

    # === TAB 2: PREDICTIONS ===
    with tab_preds:
        st.subheader("Model Predictions & Profit Opportunities (Next 24 Hours)")
        
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
        
        if pred_data:
            pred_df = pd.DataFrame(pred_data)
            st.dataframe(pred_df.style.map(color_profit, subset=["Predicted Price (€/MWh)"]), use_container_width=True, hide_index=True)
        else:
            st.info("Generating future predictions...")

    # === TAB 3: LIVE MARKET ===

    with tab_live:
        st.subheader("Today's Clearing Prices")
        
        berlin_tz = ZoneInfo("Europe/Berlin")
        today_str = datetime.now(berlin_tz).strftime('%Y-%m-%d')
        
        try:
            today_mask = df_full.index.astype(str).str.startswith(today_str)
            today_df = df_full[today_mask].copy()
            
            if not today_df.empty:
                display_df = today_df[['price_eur_mwh']].reset_index()
                display_df.columns = ["Time Block", "Clearing Price (€/MWh)"]
                display_df['Time Block'] = display_df['Time Block'].dt.strftime('%H:%M')
                
                # THE FIX: Removed the .style.map() logic that was silently crashing Streamlit
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
            else:
                st.info(f"No prices cleared for today ({today_str}) yet. Ensure the database has been updated.")
                
        except Exception as e:
            st.error(f"Error loading live data: {e}")
