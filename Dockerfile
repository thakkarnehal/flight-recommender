FROM python:3.12-slim

WORKDIR /app

# System deps for pandas/pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Torch CPU-only first (large wheel, separate layer for cache efficiency)
RUN pip install --no-cache-dir \
    torch==2.11.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Rest of dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code and baked-in artifacts
COPY src/ src/
COPY models/ models/
COPY data/processed/ data/processed/

ENV PYTHONPATH=/app/src

EXPOSE 8080

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
