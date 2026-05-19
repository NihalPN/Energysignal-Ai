# Use a highly optimized, slim Python runtime
FROM python:3.10-slim

# Prevent Python from writing pyc files to disc and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the API logic, trained model, and data necessary for the /metrics endpoint
COPY api/./api/
COPY models/xgb_baseline.json ./models/
COPY database/energy_market.db ./database/
COPY signals/trade_log.csv ./signals/

# Expose the FastAPI port
EXPOSE 8000

# Launch the Uvicorn ASGI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
