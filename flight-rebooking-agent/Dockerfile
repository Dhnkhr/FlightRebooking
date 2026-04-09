FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 7860

# Hugging Face Spaces (Docker SDK) listens on port 7860.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
