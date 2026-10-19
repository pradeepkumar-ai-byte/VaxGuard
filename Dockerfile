FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run multi-worker production ASGI server (bypasses Python GIL)
ENV WEB_CONCURRENCY=4
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "vaxguard.api.app:app", "--bind", "0.0.0.0:8000", "--timeout", "30", "--access-logfile", "-"]
