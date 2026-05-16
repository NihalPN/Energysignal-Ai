EnergySignal AI: Institutional Power Trading & Grid Monitoring
A production-grade algorithmic trading and real-time anomaly detection system for the German (DE-LU) wholesale electricity market. Built to execute within strict hardware constraints (8GB RAM) while handling the volatile physics of the European power grid.

Core Architecture
This system is composed of five decoupled layers:

Data Pipeline: Automated, rate-limited ingestion of ENTSO-E market data and Open-Meteo DWD ICON atmospheric models.

Machine Learning: XGBoost regression predicting day-ahead clearing prices in 15-minute intervals.

Backtesting Engine: VectorBT implementation executing fixed 10 MWh physical lots while factoring in slippage and exchange execution fees.

Grid Anomaly Detection: Real-time rolling Z-score monitor pushing asynchronous Telegram alerts during physical supply shocks.

AI RAG Analyst: LanceDB embedded vector database and Groq gpt-oss-safeguard-20b integration to classify live market news and cross-reference current grid physics against historical price spikes.

Key Engineering Achievements
Handling the SDAC 15-Minute Market Transition: On October 1, 2025, the European Single Day-Ahead Coupling transitioned to 15-minute Market Time Units. The data pipeline automatically forward-fills legacy hourly data to maintain uniform tensor shapes for the ML model without introducing interpolation-based lookahead bias.

Bypassing Negative Price Engine Crashes: The German power market experiences hundreds of hours of negative pricing annually due to renewable oversupply. To prevent the VectorBT engine from corrupting position sizing during negative price events, the backtester utilizes a strict price-offset mechanism to maintain accurate absolute P&L.

Zero Data Leakage: Backtesting implements a strict chronological TimeSeriesSplit. The production model achieved a verified 71.79% directional accuracy on out-of-sample data.

Hardware Optimization: Replaced RAM-heavy FAISS with LanceDB to enable disk-based vector retrieval, preventing Out-Of-Memory (OOM) crashes on 8GB local systems during concurrent LLM API streaming and Pandas manipulation.

Interface
Includes a multi-threaded PyQt6 institutional desktop terminal powered by PyQtGraph for zero-latency charting of high-frequency market data.
