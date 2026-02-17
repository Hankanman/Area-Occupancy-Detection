# Use Python 3.13 slim image for smaller size
FROM python:3.13-slim

# Set working directory to project root
WORKDIR /app

# Install build dependencies needed for compiling Python packages (e.g., lru-dict)
# These will be removed after installation to keep the image small
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy simulator pyproject.toml first for better layer caching
COPY simulator/pyproject.toml /app/simulator/pyproject.toml

# Install Python dependencies from simulator pyproject.toml
# This file contains all dependencies needed: Flask, Home Assistant, etc.
# Note: Home Assistant includes many dependencies like SQLAlchemy, PyYAML, etc.
RUN pip install --no-cache-dir /app/simulator/

# Remove build dependencies to reduce image size
RUN apt-get purge -y --auto-remove \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files needed for the simulator
# Copy custom_components (required for imports)
COPY custom_components/ /app/custom_components/
# Copy simulator directory
COPY simulator/ /app/simulator/
# Copy main.py entry point
COPY main.py /app/main.py

# Create non-root user for security
RUN useradd -m -u 1000 simulator && \
    chown -R simulator:simulator /app

# Switch to non-root user
USER simulator

# Expose the default port (configurable via PORT env var)
EXPOSE 10000

# Set default environment variables
ENV PORT=10000
ENV FLASK_DEBUG=0
ENV FLASK_HOST=0.0.0.0

# Health check (checks if Flask app is responding, using PORT env var)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", \"10000\")}/api/get-purposes').read()" || exit 1

# Run the Flask application
CMD ["python", "main.py"]

