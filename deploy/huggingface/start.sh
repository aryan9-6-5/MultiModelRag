#!/bin/sh
# FastAPI on 8000 (internal), Streamlit on 7860 (the port Spaces exposes).
python -m app.api &
exec streamlit run frontend/app.py \
    --server.address=0.0.0.0 --server.port=7860 --server.headless=true \
    --server.enableXsrfProtection=false --server.enableCORS=false
