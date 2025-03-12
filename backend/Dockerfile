FROM python:3.10-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Environment setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONPATH=/app \
    PORT=8000

WORKDIR /app

# Install Python dependencies first for caching
COPY requirements.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    /opt/venv/bin/pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Copy application
COPY . .

# Fix permissions
RUN chmod a+rx -R ./resources/

# Start script
RUN echo '#!/bin/sh\n/opt/venv/bin/uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}' > start.sh && \
    chmod +x start.sh

CMD ["./start.sh"]