FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Runtime-only dependencies for serving the environment/API.
COPY requirements.runtime.txt /app/requirements.runtime.txt
RUN pip install --no-cache-dir -r /app/requirements.runtime.txt

# Copy only runtime files required by the Space/API app.
COPY app.py /app/app.py
COPY environment.py /app/environment.py
COPY tasks.py /app/tasks.py
COPY openenv.yaml /app/openenv.yaml
COPY frontend /app/frontend

# Keep inference modules available in container for optional local checks.
COPY inference.py /app/inference.py
COPY baseline.py /app/baseline.py
COPY ml_policy.py /app/ml_policy.py

# Keep lightweight report + current runtime ML artifact.
COPY artifacts/ml_policy.pkl /app/artifacts/ml_policy.pkl
COPY artifacts/ml_policy_report.json /app/artifacts/ml_policy_report.json

EXPOSE 7860

# Hugging Face Spaces (Docker SDK) listens on port 7860.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]