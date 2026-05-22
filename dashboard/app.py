import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import pyqtgraph as pg
import qdarktheme
import xgboost as xgb
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QFrame,
    QGridLayout,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm.rag_analyst import analyze_market_condition
from llm.llm_classifier import fetch_live_german_energy_news

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "energy_market.db"
)
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "xgb_baseline.json"
)


class DataWorker(QThread):
    """Calculates independent timelines for Tab 1 (7-Day) and Tab 2 (3-Day)."""

    data_loaded = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def run(self):
        try:
            conn = sqlite3.connect(DB_PATH)
            df_full = pd.read_sql_query(
                "SELECT * FROM master_features",
                conn,
                index_col="timestamp",
                parse_dates=["timestamp"],
            )
            conn.close()

            if df_full.empty:
                raise ValueError("Database is empty. Please run the pipeline first.")

            df_full = df_full.sort_index()

            train_df = df_full.dropna(subset=["target_price_24h_ahead"])
            X_train = train_df.drop(columns=["target_price_24h_ahead"])
            y_train = train_df["target_price_24h_ahead"]

            model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
            model.fit(X_train, y_train)

            # THE FIX: .tz_localize(None) strips the timezone awareness so it can safely
            # be compared to the timezone-naive datetime64[ns] index from SQLite.
            berlin_now = pd.Timestamp.now(tz="Europe/Berlin").tz_localize(None)
            today_midnight = berlin_now.floor("D")

            # ==========================================
            # TAB 1 LOGIC: 7 Days Past + 24 Hours Future
            # ==========================================
            df_current = df_full[df_full.index <= berlin_now]
            if df_current.empty:
                df_current = df_full

            history_df = df_current.tail(672)  # Last 7 days
            t1_hist_prices = history_df["price_eur_mwh"].values
            t1_hist_preds = model.predict(
                history_df.drop(columns=["target_price_24h_ahead"], errors="ignore")
            )

            live_df = df_current.tail(96)
            t1_future_prices = model.predict(
                live_df.drop(columns=["target_price_24h_ahead"], errors="ignore")
            )

            latest_price = float(history_df["price_eur_mwh"].iloc[-1])
            latest_time_str = history_df.index[-1].strftime("%Y-%m-%d %H:%M")
            target_price_24h_now = float(t1_future_prices[-1])

            EXPECTED_MARGIN = 40.0
            current_spread = target_price_24h_now - latest_price

            if current_spread > EXPECTED_MARGIN and latest_price > 0:
                signal_text = "BUY 10 MWh"
                signal_color = "#00ff00"
            else:
                signal_text = "PRESERVE CAPITAL"
                signal_color = "#ffaa00"

            # ==========================================
            # TAB 2 LOGIC: 3-Day Forward Horizon
            # (Today, Tomorrow, Day After Tomorrow)
            # ==========================================
            yesterday_midnight = today_midnight - pd.Timedelta(days=1)

            # Extract features to build predictions starting precisely at Today Midnight
            feature_df = df_full[df_full.index >= yesterday_midnight].copy()
            X_to_predict = feature_df.drop(columns=["target_price_24h_ahead"], errors="ignore")
            raw_preds = model.predict(X_to_predict)

            # Shift index forward 24h to align predictions with their target time
            pred_dates = X_to_predict.index + pd.Timedelta(days=1)
            pred_df = pd.DataFrame({"ai_forecast": raw_preds}, index=pred_dates)

            actuals_df = df_full[df_full.index >= today_midnight][["price_eur_mwh"]]

            combined_df = pred_df.join(actuals_df, how="outer")
            end_of_d2 = today_midnight + pd.Timedelta(days=3) - pd.Timedelta(minutes=15)
            combined_df = combined_df[
                (combined_df.index >= today_midnight) & (combined_df.index <= end_of_d2)
            ]

            payload = {
                "t1_hist_prices": t1_hist_prices,
                "t1_hist_preds": t1_hist_preds,
                "t1_future_prices": t1_future_prices,
                "latest_price": latest_price,
                "latest_time_str": latest_time_str,
                "target_price_24h_now": target_price_24h_now,
                "sig_text": signal_text,
                "sig_color": signal_color,
                "t2_timestamps": combined_df.index,
                "t2_actuals": combined_df["price_eur_mwh"].values,
                "t2_forecasts": combined_df["ai_forecast"].values,
            }

            self.data_loaded.emit(payload)

        except Exception as e:
            self.error_signal.emit(str(e))


class RAGWorker(QThread):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def run(self):
        try:
            self.update_signal.emit("📡 Fetching live news from SMARD.de...")
            live_news = fetch_live_german_energy_news()
            news_text = " | ".join(live_news) if live_news else "No major news today."
            self.update_signal.emit(f"📰 Live Headlines: {news_text}\n")

            self.update_signal.emit("📊 Extracting latest grid physics from local database...")
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query(
                "SELECT * FROM master_features WHERE total_renewable IS NOT NULL AND residual_load IS NOT NULL ORDER BY timestamp DESC LIMIT 1",
                conn,
            )
            conn.close()

            if not df.empty:
                price = float(df["price_eur_mwh"].head(1).item())
                wind = float(df["total_renewable"].head(1).item())
                residual = float(df["residual_load"].head(1).item())
                actual_load = wind + residual

                real_scenario = (
                    f"Current Market Reality: The price is {price:.2f} EUR/MWh. "
                    f"Renewable generation is {wind:.2f} MW against a grid load of {actual_load:.2f} MW. "
                    f"Latest Live News: {news_text}"
                )
            else:
                real_scenario = f"Latest Live News: {news_text}"

            self.update_signal.emit(
                "🧠 Searching LanceDB for historical precedents and prompting Groq...\n"
            )
            analysis = analyze_market_condition(real_scenario)
            self.update_signal.emit(f"=== 🤖 LIVE AI TRADING ANALYSIS ===\n\n{analysis}")

        except Exception as e:
            self.update_signal.emit(f"\n❌ Error connecting to AI: {str(e)}")
        finally:
            self.finished_signal.emit()


class TradingTerminal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EnergySignal AI - Institutional Terminal")
        self.resize(1300, 900)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.tab_strategy = QWidget()
        self.tab_predictions = QWidget()
        self.tab_live = QWidget()

        self.tabs.addTab(self.tab_strategy, "Strategy & AI Analyst")
        self.tabs.addTab(self.tab_predictions, "3-Day Trading Horizon")
        self.tabs.addTab(self.tab_live, "Live Market Monitor")

        self.setup_strategy_tab()
        self.setup_predictions_tab()
        self.setup_live_tab()

        self.today_date_str = None
        self.current_live_block = None

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_live_clock)
        self.timer.start(1000)

        self.data_worker = DataWorker()
        self.data_worker.data_loaded.connect(self.on_data_loaded)
        self.data_worker.error_signal.connect(self.on_data_error)
        self.data_worker.start()

    def setup_strategy_tab(self):
        layout = QVBoxLayout(self.tab_strategy)

        kpi_frame = QFrame()
        kpi_frame.setStyleSheet("QFrame { background-color: #1e1e2e; border-radius: 8px; }")
        kpi_layout = QGridLayout(kpi_frame)

        self.lbl_current_title = QLabel("Last Cleared Price")
        self.lbl_current_price = QLabel("Loading...")
        self.lbl_forecast_title = QLabel("XGBoost Target Forecast")
        self.lbl_forecast_price = QLabel("Loading...")
        self.lbl_signal_title = QLabel("Algorithmic Signal")
        self.lbl_signal = QLabel("Loading...")

        title_font = QFont("Arial", 11, QFont.Weight.Bold)
        value_font = QFont("Arial", 18, QFont.Weight.Bold)

        for lbl in [self.lbl_current_title, self.lbl_forecast_title, self.lbl_signal_title]:
            lbl.setFont(title_font)
            lbl.setStyleSheet("color: #8c8c8c;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        for lbl in [self.lbl_current_price, self.lbl_forecast_price, self.lbl_signal]:
            lbl.setFont(value_font)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        kpi_layout.addWidget(self.lbl_current_title, 0, 0)
        kpi_layout.addWidget(self.lbl_current_price, 1, 0)
        kpi_layout.addWidget(self.lbl_forecast_title, 0, 1)
        kpi_layout.addWidget(self.lbl_forecast_price, 1, 1)
        kpi_layout.addWidget(self.lbl_signal_title, 0, 2)
        kpi_layout.addWidget(self.lbl_signal, 1, 2)
        layout.addWidget(kpi_frame)

        middle_layout = QHBoxLayout()

        chart_frame = QFrame()
        chart_layout = QVBoxLayout(chart_frame)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("DE-LU 15-Minute Resolution", color="w", size="12pt")
        self.plot_widget.setLabel("left", "Price", units="€/MWh")
        self.plot_widget.setLabel("bottom", "Time Steps")
        self.plot_widget.addLegend()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        chart_layout.addWidget(self.plot_widget)
        middle_layout.addWidget(chart_frame, stretch=2)

        ai_frame = QFrame()
        ai_layout = QVBoxLayout(ai_frame)
        ai_label = QLabel("RAG Market Analyst")
        ai_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))

        self.ai_output = QTextEdit()
        self.ai_output.setReadOnly(True)
        self.ai_output.setStyleSheet("QTextEdit { font-size: 14px; line-height: 1.5; }")
        self.ai_output.setText(
            "System ready. Click below to fetch live news and generate AI context."
        )

        self.btn_run_ai = QPushButton("Run Live AI Analysis")
        self.btn_run_ai.setMinimumHeight(45)
        self.btn_run_ai.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 14px; background-color: #4169E1; }"
        )
        self.btn_run_ai.clicked.connect(self.trigger_ai_analysis)

        ai_layout.addWidget(ai_label)
        ai_layout.addWidget(self.ai_output)
        ai_layout.addWidget(self.btn_run_ai)
        middle_layout.addWidget(ai_frame, stretch=1)

        layout.addLayout(middle_layout)

    def setup_predictions_tab(self):
        layout = QVBoxLayout(self.tab_predictions)

        self.lbl_pred_header = QLabel("3-Day Execution Tape & AI Cross-Check")
        self.lbl_pred_header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.lbl_pred_header.setStyleSheet("color: #ffffff; margin-bottom: 5px;")

        self.cross_check_plot = pg.PlotWidget()
        self.cross_check_plot.setTitle(
            "📉 Visual Tracker: AI Prediction vs Actual Market", color="w", size="12pt"
        )
        self.cross_check_plot.setLabel("left", "Price", units="€/MWh")
        self.cross_check_plot.setLabel("bottom", "Today, Tomorrow, Day After Tomorrow (15m Blocks)")
        self.cross_check_plot.addLegend()
        self.cross_check_plot.showGrid(x=True, y=True, alpha=0.3)
        self.cross_check_plot.setFixedHeight(280)

        self.pred_table = QTableWidget()
        self.pred_table.setColumnCount(2)
        self.pred_table.setHorizontalHeaderLabels(
            ("Delivery Time Block", "Predicted Price (€/MWh)")
        )
        self.pred_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.pred_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.pred_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.pred_table.setStyleSheet(
            "QTableWidget { font-size: 16px; background-color: #1e1e2e; color: #ffffff; gridline-color: #333333; } "
            "QHeaderView::section { background-color: #282a36; font-weight: bold; font-size: 14px; padding: 5px; border: 1px solid #444; }"
        )

        layout.addWidget(self.lbl_pred_header)
        layout.addWidget(self.cross_check_plot)
        layout.addWidget(self.pred_table)

    def setup_live_tab(self):
        layout = QVBoxLayout(self.tab_live)

        self.lbl_live_time = QLabel("Berlin Time: --:--:--")
        self.lbl_live_time.setFont(QFont("Courier", 18, QFont.Weight.Bold))
        self.lbl_live_time.setStyleSheet("color: #00d2ff; margin-bottom: 10px;")
        self.lbl_live_time.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.price_table = QTableWidget()
        self.price_table.setColumnCount(2)
        self.price_table.setHorizontalHeaderLabels(
            ("Date & Time Block", "Actual Clearing Price (€/MWh)")
        )
        self.price_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.price_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.price_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.price_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.price_table.setStyleSheet(
            "QTableWidget { font-size: 16px; background-color: #1e1e2e; color: #ffffff; gridline-color: #333333; } "
            "QHeaderView::section { background-color: #282a36; font-weight: bold; font-size: 14px; padding: 5px; border: 1px solid #444; }"
        )

        layout.addWidget(self.lbl_live_time)
        layout.addWidget(self.price_table)

    def on_data_loaded(self, payload):
        # === TAB 1: MACRO GRAPH (Restored to 7-Day + 24h) ===
        self.plot_widget.clear()

        t1_hist = payload["t1_hist_prices"]
        t1_preds = payload["t1_hist_preds"]
        t1_future = payload["t1_future_prices"]

        x_hist = np.arange(len(t1_hist))
        self.plot_widget.plot(
            x_hist, t1_hist, pen=pg.mkPen("#00d2ff", width=2), name="Actual Prices (7 Days)"
        )
        self.plot_widget.plot(
            x_hist,
            t1_preds,
            pen=pg.mkPen("#ff55ff", width=2, style=Qt.PenStyle.DotLine),
            name="Model Cross-Check",
        )

        x_future = np.arange(len(t1_hist), len(t1_hist) + len(t1_future))
        self.plot_widget.plot(
            x_future,
            t1_future,
            pen=pg.mkPen("#ffaa00", width=2, style=Qt.PenStyle.DashLine),
            name="XGBoost Forecast (Next 24h)",
        )

        self.lbl_current_title.setText(f"Price at {payload['latest_time_str']}")
        self.lbl_current_price.setText(f"€{payload['latest_price']:.2f}")

        target_time = pd.to_datetime(payload["latest_time_str"]) + pd.Timedelta(hours=24)
        self.lbl_forecast_title.setText(f"Forecast ({target_time.strftime('%H:%M')} Tomorrow)")
        self.lbl_forecast_price.setText(f"€{payload['target_price_24h_now']:.2f}")

        self.lbl_signal.setText(payload["sig_text"])
        self.lbl_signal.setStyleSheet(
            f"color: {payload['sig_color']}; font-weight: bold; font-size: 18px;"
        )

        # === TAB 2: 3-DAY HORIZON GRAPH & TABLE ===
        self.cross_check_plot.clear()

        t2_ts = payload["t2_timestamps"]
        t2_act = payload["t2_actuals"]
        t2_fcast = payload["t2_forecasts"]
        x_cross = np.arange(len(t2_ts))

        # Add midnight separators for Today, Tomorrow, D+2
        self.cross_check_plot.addItem(
            pg.InfiniteLine(
                pos=96, angle=90, pen=pg.mkPen("w", style=Qt.PenStyle.DashLine, alpha=100)
            )
        )
        self.cross_check_plot.addItem(
            pg.InfiniteLine(
                pos=192, angle=90, pen=pg.mkPen("w", style=Qt.PenStyle.DashLine, alpha=100)
            )
        )

        valid_act = ~pd.isna(t2_act)
        valid_fcast = ~pd.isna(t2_fcast)

        if valid_act.any():
            self.cross_check_plot.plot(
                x_cross[valid_act],
                t2_act[valid_act],
                pen=pg.mkPen("#ffffff", width=2),
                name="Actual Price",
            )
        if valid_fcast.any():
            self.cross_check_plot.plot(
                x_cross[valid_fcast],
                t2_fcast[valid_fcast],
                pen=pg.mkPen("#00ff00", width=2),
                name="AI Forecast",
            )

        self.pred_table.setRowCount(len(t2_ts))
        EXPECTED_MARGIN = 40.0
        latest_price = payload["latest_price"]

        for i in range(len(t2_ts)):
            time_label = pd.Timestamp(t2_ts[i]).strftime("%b %d, %H:%M")
            pred = t2_fcast[i]
            act = t2_act[i]

            item_time = QTableWidgetItem(time_label)
            item_time.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            if not pd.isna(act):
                # Has Cleared Actual Market Data
                item_time.setForeground(QColor("#888888"))
                if not pd.isna(pred):
                    error = pred - act
                    price_label = f"€{pred:.2f} (Actual: €{act:.2f} | Err: €{error:+.2f})"
                else:
                    price_label = f"Actual: €{act:.2f} (No AI Forecast)"

                item_price = QTableWidgetItem(price_label)
                item_price.setForeground(QColor("#ff55ff"))
            else:
                # Future Unknown (Pure Forecast)
                if not pd.isna(pred):
                    expected_profit = pred - latest_price
                    if pred < 0:
                        price_label = f"€{pred:.2f} (Negative)"
                        item_price = QTableWidgetItem(price_label)
                        item_price.setForeground(QColor("#ff4444"))
                    elif expected_profit > EXPECTED_MARGIN and latest_price > 0:
                        price_label = f"€{pred:.2f} | INVEST (+€{expected_profit:.2f})"
                        item_price = QTableWidgetItem(price_label)
                        item_price.setForeground(QColor("#00ff00"))
                        item_time.setForeground(QColor("#00ff00"))
                        font = QFont()
                        font.setBold(True)
                        item_price.setFont(font)
                        item_time.setFont(font)
                    else:
                        price_label = f"€{pred:.2f} (Spread: €{expected_profit:.2f})"
                        item_price = QTableWidgetItem(price_label)
                        item_price.setForeground(QColor("#ffaa00"))
                else:
                    price_label = "Awaiting Weather Data..."
                    item_price = QTableWidgetItem(price_label)
                    item_price.setForeground(QColor("#888888"))

            item_price.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.pred_table.setItem(i, 0, item_time)
            self.pred_table.setItem(i, 1, item_price)

    def on_data_error(self, err_msg):
        self.plot_widget.setTitle(f"Data Error: {err_msg}", color="r")

    def update_live_clock(self):
        berlin_time = pd.Timestamp.now(tz="Europe/Berlin")
        block_start = berlin_time.floor("15min")

        self.lbl_live_time.setText(
            f"Berlin Time: {berlin_time.strftime('%Y-%m-%d %H:%M:%S')}  |  Active Block: {block_start.strftime('%H:%M')}"
        )

        current_date_str = berlin_time.strftime("%Y-%m-%d")
        if self.today_date_str != current_date_str:
            self.today_date_str = current_date_str
            self.load_horizon_prices(current_date_str)

        self.highlight_current_block(block_start)

    def load_horizon_prices(self, date_str):
        try:
            conn = sqlite3.connect(DB_PATH)
            # Fetch EVERYTHING from Today's midnight onwards (Today + Tomorrow actuals if leaked)
            query = f"SELECT timestamp, price_eur_mwh FROM day_ahead_prices WHERE timestamp >= '{date_str} 00:00:00' AND price_eur_mwh IS NOT NULL ORDER BY timestamp ASC"
            df = pd.read_sql_query(query, conn)
            conn.close()

            self.price_table.setRowCount(len(df))

            for i, row in df.iterrows():
                ts = pd.Timestamp(row["timestamp"])
                time_label = f"{ts.strftime('%b %d - %H:%M')}"

                price = row["price_eur_mwh"]
                price_str = f"€{price:.2f}"

                item_time = QTableWidgetItem(time_label)
                item_price = QTableWidgetItem(price_str)
                item_time.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item_price.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                if price < 0:
                    item_price.setForeground(QColor("#ff4444"))
                else:
                    item_price.setForeground(QColor("#00ff00"))

                item_time.setData(Qt.ItemDataRole.UserRole, ts.strftime("%Y-%m-%d %H:%M:%S"))

                self.price_table.setItem(i, 0, item_time)
                self.price_table.setItem(i, 1, item_price)

        except Exception as e:
            print(f"Table Load Error: {e}")

    def highlight_current_block(self, block_start):
        block_str = block_start.strftime("%Y-%m-%d %H:%M:%S")
        for row in range(self.price_table.rowCount()):
            item = self.price_table.item(row, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) == block_str:
                self.price_table.selectRow(row)
                self.price_table.scrollToItem(item)
                break

    def trigger_ai_analysis(self):
        self.btn_run_ai.setEnabled(False)
        self.ai_output.clear()
        self.worker = RAGWorker()
        self.worker.update_signal.connect(self.append_ai_text)
        self.worker.finished_signal.connect(self.ai_finished)
        self.worker.start()

    def append_ai_text(self, text):
        self.ai_output.append(text)

    def ai_finished(self):
        self.btn_run_ai.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    qdarktheme.setup_theme("dark", corner_shape="sharp")
    window = TradingTerminal()
    window.show()
    sys.exit(app.exec())
