FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Install standard dependencies
RUN pip install --no-cache-dir pydantic openai pyyaml

# Environment variables (override at runtime)
ENV API_BASE_URL=https://api.together.xyz/v1
ENV MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo

# Command to run the agent evaluation
CMD ["python", "baseline.py"]
