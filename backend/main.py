import os
import sys
import sqlite3
import pandas as pd
import xgboost as xgb
import asyncio
import math
from fastapi import FastAPI, BackgroundTasks  # noqa: F402
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from datetime import timedelta

# Setup Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from llm.rag_analyst_cloud import analyze_market_condition
from llm.llm_classifier import fetch_live_german_energy_news

app = FastAPI(title="EnergySignal AI API")

# Allow your frontend URLs
app.add_middleware(
    CORSMiddleware,  # noqa: F821
    allow_origins=["https://energysignalai.onrender.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.path.join(BASE_DIR, "database", "energy_market.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_baseline.json")

# Load Model
model = xgb.XGBRegressor()
if os.path.exists(MODEL_PATH):
    model.load_model(MODEL_PATH)

# --- SERVER SIDE STATE (Locally Stored in RAM) ---
# This dictionary holds the AI's thoughts so it never makes the user wait
server_state = {
    "rag_analysis": "System initializing... Booting AI Trading Desk...",
    "last_ai_update": None,
}


def fetch_master_features():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM master_features", conn, index_col="timestamp", parse_dates=["timestamp"]
    )
    conn.close()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("Europe/Berlin")
    return df.sort_index()


def replace_nan(val):
    if pd.isna(val) or math.isnan(val):
        return None
    return float(val)


# --- BACKGROUND AI WORKER ---
async def run_ai_loop():
    """Runs immediately on startup, then sleeps for 1 hour, infinitely."""
    while True:
        try:
            print("Running hourly Background AI Task...")
            df_full = fetch_master_features()
            if not df_full.empty:
                berlin_now = pd.Timestamp.now(tz="Europe/Berlin")
                df_current = df_full[df_full.index <= berlin_now]
                history_df = df_current.tail(1)

                latest_price = float(history_df["price_eur_mwh"].iloc[0])
                wind = (
                    float(history_df["total_renewable"].iloc[0])
                    if "total_renewable" in history_df
                    else 0.0
                )
                residual = (
                    float(history_df["residual_load"].iloc[0])
                    if "residual_load" in history_df
                    else 0.0
                )

                live_news = fetch_live_german_energy_news()
                news_text = " | ".join(live_news) if live_news else "No major news today."

                real_scenario = (
                    f"Current Market Reality: The price is {latest_price:.2f} EUR/MWh. "
                    f"Renewable generation is {wind:.2f} MW against a grid load of {wind+residual:.2f} MW. "
                    f"Latest Live News: {news_text}"
                )

                # Fetch from LLM and store locally in RAM
                analysis = analyze_market_condition(real_scenario)
                server_state["rag_analysis"] = analysis
                server_state["last_ai_update"] = berlin_now.strftime("%H:%M")
                print("Background AI Task Complete. Saved to Server RAM.")
        except Exception as e:
            print(f"Background AI Error: {e}")
            server_state["rag_analysis"] = "Temporary AI Disconnect. Retrying next cycle..."

        # Sleep for exactly 1 hour (3600 seconds)
        await asyncio.sleep(3600)


@app.on_event("startup")
async def startup():
    # 1. Initialize the Endpoint Cache for the Math
    FastAPICache.init(InMemoryBackend())
    # 2. Fire the AI Background Loop to defeat Cold Starts
    asyncio.create_task(run_ai_loop())


# --- ENDPOINTS ---
@app.get("/api/v1/rag-analysis")
async def get_rag_analysis():
    """Instantly returns the AI thought stored in Server RAM."""
    return {"analysis": server_state["rag_analysis"], "last_update": server_state["last_ai_update"]}


@app.get("/api/v1/dashboard-data")
@cache(expire=3600)  # Caches the complex XGBoost math for 1 hour
async def get_dashboard_data():
    """Your main dashboard logic. It will only actually run once per hour."""
    df_full = fetch_master_features()
    if df_full.empty:
        return {"error": "Database empty"}

    berlin_now = pd.Timestamp.now(tz="Europe/Berlin")
    today_midnight = berlin_now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_midnight = today_midnight - pd.Timedelta(days=1)

    df_current = df_full[df_full.index <= berlin_now]
    if df_current.empty:
        df_current = df_full

    history_df = df_current.tail(672)
    latest_price = float(history_df["price_eur_mwh"].iloc[-1])
    last_dt_str = history_df.index[-1].strftime("%Y-%m-%d %H:%M")

    X_live = df_current.tail(96).drop(columns=["target_price_24h_ahead"], errors="ignore")
    if "price_eur_mwh" in model.feature_names_in_ and "price_eur_mwh" not in X_live.columns:
        X_live["price_eur_mwh"] = history_df["price_eur_mwh"].tail(96).values

    predictions = model.predict(X_live)
    target_price_24h_now = float(predictions[-1])

    EXPECTED_MARGIN = 40.0
    current_spread = target_price_24h_now - latest_price
    signal_text = (
        "🟢 BUY 10 MWh"
        if (current_spread > EXPECTED_MARGIN and latest_price > 0)
        else "🟠 PRESERVE CAPITAL"
    )

    future_dates = [history_df.index[-1] + timedelta(minutes=15 * i) for i in range(1, 97)]
    t1_timestamps = [
        ts.strftime("%Y-%m-%d %H:%M") for ts in history_df.index.append(pd.Index(future_dates))
    ]
    t1_actuals = [replace_nan(x) for x in history_df["price_eur_mwh"]] + [None] * len(future_dates)

    X_hist = history_df.drop(columns=["target_price_24h_ahead"], errors="ignore")
    hist_preds = model.predict(X_hist)
    t1_cross_check = [replace_nan(x) for x in hist_preds] + [None] * len(future_dates)

    last_pink_point = replace_nan(hist_preds[-1])
    t1_forecast = (
        [None] * (len(history_df) - 1) + [last_pink_point] + [replace_nan(x) for x in predictions]
    )

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

    t2_timestamps = [ts.strftime("%b %d, %H:%M") for ts in combined_df.index]
    t2_actuals = [replace_nan(x) for x in combined_df["Actual Price"]]
    t2_forecasts = [replace_nan(x) for x in combined_df["AI Forecast"]]

    t2_table = []
    for dt, row in combined_df.iterrows():
        time_label = dt.strftime("%b %d, %H:%M")
        act = replace_nan(row["Actual Price"])
        pred = replace_nan(row["AI Forecast"])
        if act is not None:
            if pred is not None:
                err = pred - act
                status = f"€{pred:.2f} (Actual: €{act:.2f} | Err: €{err:+.2f})"
                color = "magenta"
            else:
                status = f"Actual: €{act:.2f} (No AI Forecast)"
                color = "magenta"
            time_label += " (Cross-Check)"
            time_color = "gray"
        else:
            if pred is not None:
                expected_profit = pred - latest_price
                if pred < 0:
                    status = f"€{pred:.2f} (Negative)"
                    color = "red"
                elif expected_profit > EXPECTED_MARGIN and latest_price > 0:
                    status = f"€{pred:.2f} | INVEST (+€{expected_profit:.2f})"
                    color = "green"
                    time_color = "green"
                else:
                    status = f"€{pred:.2f} (Spread: €{expected_profit:.2f})"
                    color = "orange"
                time_color = "white"
            else:
                status = "Awaiting Weather Data..."
                color = "gray"
                time_color = "white"
            time_label += " (Forecast)"
        t2_table.append(
            {
                "time": time_label,
                "status": status,
                "color": color,
                "time_color": time_color if "time_color" in locals() else "gray",
            }
        )

    live_market_df = df_full[df_full.index >= today_midnight].copy()
    live_market_df = live_market_df.dropna(subset=["price_eur_mwh"])
    t3_table = [
        {"time": dt.strftime("%b %d - %H:%M"), "price": f"€{row['price_eur_mwh']:.2f}"}
        for dt, row in live_market_df.iterrows()
    ]

    return {
        "kpis": {
            "latest_price": latest_price,
            "last_time": last_dt_str,
            "tomorrow_time": (history_df.index[-1] + timedelta(days=1)).strftime("%H:%M"),
            "forecast_price": target_price_24h_now,
            "signal": signal_text,
        },
        "tab1": {
            "timestamps": t1_timestamps,
            "actuals": t1_actuals,
            "cross_check": t1_cross_check,
            "forecast": t1_forecast,
        },
        "tab2": {
            "timestamps": t2_timestamps,
            "actuals": t2_actuals,
            "forecasts": t2_forecasts,
            "table": t2_table,
        },
        "tab3": {"table": t3_table},
    }
