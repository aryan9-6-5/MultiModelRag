# Single-container Hugging Face Space: FastAPI backend + Streamlit UI.
# Copy this file to the repository root of the Space as `Dockerfile`.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
ENV HOME=/home/user PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY --chown=user app/ ./app/
COPY --chown=user frontend/ ./frontend/
COPY --chown=user evaluation/ ./evaluation/
COPY --chown=user deploy/huggingface/start.sh ./start.sh
RUN chmod +x start.sh && chown -R user:user /home/user/app

USER user

ENV DATA_DIR=/home/user/app/data \
    STORAGE_DIR=/home/user/app/storage \
    API_BASE=http://127.0.0.1:8000 \
    LLM_PROVIDER=groq \
    DEMO_MODE=true

# Build the demo index at image build time: download a COCO val2017 subset,
# embed it with OpenCLIP, caption it with BLIP, and persist FAISS + DuckDB.
RUN mkdir -p data storage \
    && python -m evaluation.datasets.download_coco_subset --n-images 100 \
    && python -m app.ingest --path evaluation/datasets/coco_subset/images \
    && rm -f evaluation/datasets/coco_subset/captions_val2017.json

EXPOSE 7860
CMD ["./start.sh"]
