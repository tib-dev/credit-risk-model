# Use a slim but pre-built image to save compile time
FROM python:3.11-slim

# 1. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

# 2. Install minimal system dependencies 
# (Adding --no-install-recommends keeps the image small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. CACHE LAYER: Copy only requirements and install
# This layer is cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the code (This changes often)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "credit_risk_model.api.main:app", "--host", "0.0.0.0", "--port", "8000"]