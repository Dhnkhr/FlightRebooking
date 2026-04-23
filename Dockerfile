FROM python:3.10-slim

# Install system dependencies for bitsandbytes
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY app.py .
COPY environment.py .
COPY tasks.py .
COPY openenv.yaml .
COPY frontend/ ./frontend/
COPY flight-rebooking-lora/ ./flight-rebooking-lora/

# Create a place for HF cache
RUN mkdir -p /app/.cache && chmod 777 /app/.cache
ENV HF_HOME=/app/.cache

EXPOSE 7860

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]