# EnergySignal AI: Institutional Power Trading & Grid Monitoring

A production-grade algorithmic trading and real-time anomaly detection system for the **German (DE-LU) wholesale electricity market**.

Built to execute within strict hardware constraints (**8GB RAM**) while handling the volatile physics of the European power grid.

---

## Core Architecture

This system is composed of **five decoupled layers**:

### Data Pipeline
Automated, rate-limited ingestion of:

- **ENTSO-E** wholesale electricity market data
- **Open-Meteo DWD ICON** atmospheric forecast models

---

### Machine Learning Engine
Price forecasting pipeline using:

- **XGBoost Regression**
- Predicts **day-ahead clearing prices**
- Supports **15-minute market intervals**

---

### Backtesting Engine
Institutional-grade simulation using **VectorBT**:

- Executes fixed **10 MWh physical lots**
- Includes:
  - slippage modeling
  - exchange execution fees
  - realistic trade accounting

---

### Grid Anomaly Detection
Real-time grid instability monitoring:

- Rolling **Z-score anomaly detection**
- Detects abnormal physical supply shocks
- Sends asynchronous **Telegram alerts**

---

### AI RAG Market Analyst
AI-assisted market intelligence layer:

- **LanceDB** embedded vector database
- **Groq gpt-oss-safeguard-20b** integration
- Capabilities:
  - classify live market news
  - cross-reference current grid physics
  - compare against historical price spikes

---

# Key Engineering Achievements

## SDAC 15-Minute Market Transition Handling
On **October 1, 2025**, the European **Single Day-Ahead Coupling (SDAC)** transitioned to **15-minute Market Time Units (MTUs)**.

The data pipeline automatically:

- forward-fills legacy hourly data
- maintains uniform tensor shapes
- avoids interpolation-based lookahead bias

---

## Negative Price Stability Protection
The German electricity market frequently enters **negative pricing regimes** due to renewable oversupply.

To prevent simulation corruption:

- backtester applies a strict **price offset mechanism**
- preserves correct position sizing
- maintains accurate absolute P&L during negative pricing

---

## Zero Data Leakage Validation
Backtesting enforces strict chronological validation:

- **TimeSeriesSplit**
- no future leakage
- realistic production evaluation

### Verified Performance
- **71.79% directional accuracy**
- measured on out-of-sample market data

---

## Hardware Optimization
Designed specifically for constrained local systems.

Optimization decisions:

- replaced memory-heavy **FAISS**
- migrated to disk-based **LanceDB**
- prevents OOM crashes during:
  - concurrent LLM API streaming
  - Pandas-heavy processing
  - vector retrieval workloads

Target hardware:
**8GB RAM systems**

---

# Interface

Includes a multi-threaded **PyQt6 institutional desktop terminal** featuring:

- **PyQtGraph zero-latency charting**
- high-frequency market visualization
- live monitoring dashboard
- responsive trading workstation UI

---

# Technology Stack

**ML / Data Science**
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- VectorBT

**AI / Retrieval**
- LanceDB
- Groq API
- RAG pipeline

**Market Data**
- ENTSO-E API
- Open-Meteo API

**Monitoring / Alerts**
- Telegram Bot API

**Desktop Interface**
- PyQt6
- PyQtGraph

---

# Design Philosophy

Institutional-grade architecture focused on:

- robustness under volatile market conditions
- strict anti-leakage modeling
- realistic execution simulation
- hardware-efficient deployment
- modular decoupled system design
