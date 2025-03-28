# Use Python 3.10 slim as the base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    SKLEARN_WARN_IGNORE=1

# Set working directory
WORKDIR /app

# Copy dependency files first for caching
COPY requirements.txt .

# Create and activate virtual environment, install dependencies
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt && \
    /opt/venv/bin/pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# Ensure virtual environment is used in PATH
ENV PATH="/opt/venv/bin:$PATH"

# Copy the entire application (force cache invalidation with a build arg if needed)
ARG BUILD_TIMESTAMP
COPY . .

# Fix permissions for resources directory (if it exists)
RUN chmod -R a+rx ./resources/ || true

# Create and configure start script
RUN echo '#!/bin/sh\nuvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1' > start.sh && \
    chmod +x start.sh

# Expose the port (optional, for documentation)
EXPOSE ${PORT}

# Set the default command
CMD ["./start.sh"]