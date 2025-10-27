FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file
COPY requirements.txt .

# Install dependencies to a clean isolated location
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --prefix=/install -r requirements.txt

FROM python:3.11-slim

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app

# Working directory
WORKDIR $APP_HOME

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy application code and static assets
COPY app ./app
COPY static ./static

# (Optional) copy other files if needed, e.g. templates/
# COPY templates ./templates

# Create non-root user
RUN useradd -m appuser
USER appuser

# Expose the FastAPI default port
EXPOSE 8000

# Start FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
