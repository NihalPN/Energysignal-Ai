import streamlit as st
import sqlite3
import pandas as pd
import xgboost as xgb
import os
import sys

# Fixing module paths for relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Adding noqa: E402 so Flake8 ignores these imports coming after executable code
from datetime import datetime, timedelta  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402


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
    setInterval(updateClock, 1000);
    updateClock(); 
    </script>
    """
    components.html(clock_html, height=60)


# --- LLM Imports ---
try:
    from llm.rag_analyst_cloud import analyze_market_condition  # noqa: E402
    from llm.llm_classifier import fetch_live_german_energy_news  # noqa: E402
except ImportError:
    st.error(
        "Error importing LLM modules. Ensure 'llm/rag_analyst.py' and 'llm/llm_classifier.py' exist."
    )

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="EnergySignal AI Terminal", layout="wide", page_icon="⚡")
DB_PATH = "database/energy_market.db"
MODEL_PATH = "models/xgb_baseline.json"


# --- 2. SHARED DATA & AI LOGIC ---
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    if os.path.exists(MODEL_PATH):
        model.load_model(MODEL_PATH)
    return model


@st.cache_data(ttl=60)
def fetch_master_features():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM master_features", conn, index_col="timestamp", parse_dates=["timestamp"]
    )
    conn.close()

    # Streamlit inherently creates timezone-aware indexes here
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    df.index = df.index.tz_convert("Europe/Berlin")
    return df.sort_index()


@st.cache_data(ttl=3600, show_spinner="🤖 AI Analyst is reading the market...")
def get_hourly_ai_analysis(latest_price, wind_generation, residual_load):
    try:
        live_news = fetch_live_german_energy_news()
        news_text = " | ".join(live_news) if live_news else "No major news today."

        actual_load = wind_generation + residual_load
        real_scenario = (
            f"Current Market Reality: The price is {latest_price:.2f} EUR/MWh. "
            f"Renewable generation is {wind_generation:.2f} MW against a grid load of {actual_load:.2f} MW. "
            f"Latest Live News: {news_text}"
        )

        analysis = analyze_market_condition(real_scenario)
        return analysis
    except Exception as e:
        return f"AI Analysis temporarily unavailable: {str(e)}"


def color_profit(val):
    if isinstance(val, str) and "INVEST" in val:
        return "color: #00ff00; font-weight: bold;"
    elif isinstance(val, str) and "Err:" in val:
        return "color: #ffaa00;"
    elif isinstance(val, str) and "Negative" in val:
        return "color: #ff4444;"
    elif isinstance(val, (int, float)) and val < 0:
        return "color: #ff4444;"
    return ""


# --- 3. MAIN UI ---
st.title("⚡ EnergySignal AI - Institutional Terminal")
live_berlin_clock()

df_full = fetch_master_features()
model = load_model()

if df_full.empty:
    st.warning("Database is empty or missing. Please ensure your data pipeline has run.")
else:
    # --- TIME ANCHORS ---
    berlin_now = pd.Timestamp.now(tz="Europe/Berlin")
    today_midnight = berlin_now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_midnight = today_midnight - pd.Timedelta(days=1)

    # Filter out the Day-Ahead Leak strictly for Tab 1 so it stays anchored to the physical present
    df_current = df_full[df_full.index <= berlin_now]
    if df_current.empty:
        df_current = df_full

    # --- TAB 1 MATH (7-Day Analyst View) ---
    history_df = df_current.tail(672)  # Last 7 days
    latest_price = float(history_df["price_eur_mwh"].iloc[-1])

    live_df = df_current.tail(96)
    X_live = live_df.drop(columns=["target_price_24h_ahead"], errors="ignore")

    if "price_eur_mwh" in model.feature_names_in_ and "price_eur_mwh" not in X_live.columns:
        X_live["price_eur_mwh"] = history_df["price_eur_mwh"].tail(96).values

    predictions = model.predict(X_live)
    target_price_24h_now = float(predictions[-1])

    EXPECTED_MARGIN = 40.0
    current_spread = target_price_24h_now - latest_price

    if current_spread > EXPECTED_MARGIN and latest_price > 0:
        signal_text = "🟢 BUY 10 MWh"
    else:
        signal_text = "🟠 PRESERVE CAPITAL"

    # --- TABS LAYOUT ---
    tab_strategy, tab_preds, tab_live = st.tabs(
        ["Strategy & AI Analyst", "Xg Boost Forecast Monitor", "Live Market Monitor"]
    )

    # === TAB 1: STRATEGY & AI (UNCHANGED 7-DAY VIEW) ===
    with tab_strategy:
        col1, col2, col3 = st.columns(3)
        last_dt = history_df.index[-1].strftime("%Y-%m-%d %H:%M")
        tomorrow_dt = (history_df.index[-1] + timedelta(days=1)).strftime("%H:%M")

        with col1:
            st.metric(label=f"Price at {last_dt}", value=f"€{latest_price:.2f}")
        with col2:
            st.metric(
                label=f"Forecast ({tomorrow_dt} Tomorrow)", value=f"€{target_price_24h_now:.2f}"
            )
        with col3:
            st.metric(label="Algorithmic Signal", value=signal_text)

        st.divider()

        chart_col, ai_col = st.columns([2, 1])

        with chart_col:
            st.subheader("DE-LU 15-Minute Resolution")

            future_dates = [history_df.index[-1] + timedelta(minutes=15 * i) for i in range(1, 97)]
            full_index = history_df.index.append(pd.Index(future_dates))
            chart_df = pd.DataFrame(index=full_index)

            chart_df["Actual Prices (7 Days)"] = history_df["price_eur_mwh"]

            cross_check = pd.Series(index=history_df.index, dtype=float)
            if len(history_df) > 96:
                X_past = history_df.iloc[:-96].drop(
                    columns=["target_price_24h_ahead"], errors="ignore"
                )
                past_preds = model.predict(X_past)
                cross_check.iloc[96:] = past_preds
            chart_df["Model Cross-Check"] = cross_check

            forecast_series = pd.Series(predictions, index=future_dates)
            chart_df["XGBoost Forecast (Next 24h)"] = forecast_series

            st.line_chart(chart_df, color=["#00d2ff", "#ff00ff", "#ffaa00"])

        with ai_col:
            st.subheader("🤖 RAG Market Analyst")
            latest_wind = (
                float(history_df["total_renewable"].tail(1).item())
                if "total_renewable" in history_df
                else 0.0
            )
            latest_residual = (
                float(history_df["residual_load"].tail(1).item())
                if "residual_load" in history_df
                else 0.0
            )

            ai_text = get_hourly_ai_analysis(latest_price, latest_wind, latest_residual)
            st.markdown(f"> {ai_text}")

    # === TAB 2: PREDICTIONS & CROSS-CHECK (3-DAY HORIZON) ===
    with tab_preds:
        st.subheader("Execution  & AI Cross-Check")

        # 1. Generate 3-Day Horizon Data
        feature_df = df_full[df_full.index >= yesterday_midnight].copy()
        X_to_predict = feature_df.drop(columns=["target_price_24h_ahead"], errors="ignore")
        raw_preds = model.predict(X_to_predict)

        pred_dates = X_to_predict.index + pd.Timedelta(days=1)
        pred_df = pd.DataFrame({"AI Forecast": raw_preds}, index=pred_dates)

        actuals_df = df_full[df_full.index >= today_midnight][["price_eur_mwh"]]
        actuals_df.rename(columns={"price_eur_mwh": "Actual Price"}, inplace=True)

        combined_df = pred_df.join(actuals_df, how="outer")
        end_of_d2 = today_midnight + pd.Timedelta(days=3) - pd.Timedelta(minutes=15)
        combined_df = combined_df[
            (combined_df.index >= today_midnight) & (combined_df.index <= end_of_d2)
        ]

        # 2. Render Graph
        st.markdown("#### 📉 Visual Tracker: AI Prediction vs Actual Market")
        # Streamlit automatically handles the NaNs natively, drawing the actual line as far as it goes,
        # while the forecast line maps all the way out to D+2.
        st.line_chart(combined_df[["Actual Price", "AI Forecast"]], color=["#ffffff", "#00ff00"])
        st.divider()

        # 3. Render Table
        pred_data = []
        for dt, row in combined_df.iterrows():
            time_label = dt.strftime("%b %d, %H:%M")
            act = row["Actual Price"]
            pred = row["AI Forecast"]

            if pd.notna(act):
                # Has Cleared Actual Market Data
                if pd.notna(pred):
                    err = pred - act
                    status = f"€{pred:.2f} (Actual: €{act:.2f} | Err: €{err:+.2f})"
                else:
                    status = f"Actual: €{act:.2f} (No AI Forecast)"
                time_label += " (Cross-Check)"
            else:
                # Future Unknown (Pure Forecast)
                if pd.notna(pred):
                    expected_profit = pred - latest_price
                    if pred < 0:
                        status = f"€{pred:.2f} (Negative)"
                    elif expected_profit > EXPECTED_MARGIN and latest_price > 0:
                        status = f"€{pred:.2f} | INVEST (+€{expected_profit:.2f})"
                    else:
                        status = f"€{pred:.2f} (Spread: €{expected_profit:.2f})"
                else:
                    status = "Awaiting Weather Data..."
                time_label += " (Forecast)"

            pred_data.append({"Delivery Time Block": time_label, "Predicted Price (€/MWh)": status})

        if pred_data:
            st.markdown("#### 📊 Execution Tape (Spread & Signals)")
            pred_df_table = pd.DataFrame(pred_data)
            st.dataframe(
                pred_df_table.style.map(color_profit, subset=["Predicted Price (€/MWh)"]),
                use_container_width=True,
                hide_index=True,
            )

    # === TAB 3: LIVE MARKET MONITOR ===
    with tab_live:
        st.subheader("Live Market Horizon")

        try:
            # Grab everything from today midnight onwards to show Today + Tomorrow's leak
            live_market_df = df_full[df_full.index >= today_midnight].copy()
            live_market_df = live_market_df.dropna(subset=["price_eur_mwh"])

            if not live_market_df.empty:
                display_df = live_market_df[["price_eur_mwh"]].reset_index()
                display_df.columns = ["Delivery Time Block", "Clearing Price (€/MWh)"]
                display_df["Delivery Time Block"] = display_df["Delivery Time Block"].dt.strftime(
                    "%b %d - %H:%M"
                )

                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                st.info("No prices cleared yet. Ensure the database has been updated.")

        except Exception as e:
            st.error(f"Error loading live data: {e}")
