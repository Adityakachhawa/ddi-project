# Base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONPATH=/app \
    PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu && \
    /opt/venv/bin/pip install -r requirements.txt

# Copy application files
COPY . .

# Ensure resources are accessible
RUN chmod a+rx -R ./resources/

# Create start script
RUN echo '#!/bin/sh\nuvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}' > start.sh && \
    chmod +x start.sh

# Runtime configuration
ENV PATH="/opt/venv/bin:$PATH"
EXPOSE $PORT

# Start command using environment variable
CMD ["./start.sh"]