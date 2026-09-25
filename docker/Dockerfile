FROM python:3.11-slim

# tesseract-ocr: optional OCR on ingested images. libgl1/libglib: Pillow/OpenCV-adjacent runtime deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY app/ ./app/
COPY frontend/ ./frontend/
COPY evaluation/ ./evaluation/
COPY tests/ ./tests/
COPY pytest.ini .

ENV DATA_DIR=/srv/data \
    STORAGE_DIR=/srv/storage \
    OLLAMA_HOST=http://ollama:11434 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /srv/data /srv/storage

EXPOSE 8000 8501

CMD ["python", "-m", "app.api"]
