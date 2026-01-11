# Use Python 3.13 slim image to match project requirements
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project configuration files
COPY pyproject.toml .

# Install Python dependencies
# We use pip to install directly from pyproject.toml
# Note: In a more complex setup, we might use uv or poetry, but pip works for standard pyproject.toml
RUN pip install --upgrade pip && \
    pip install .

# Copy application code
COPY . .

# Expose the application port
EXPOSE 8000

# Create a non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Run the application
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
