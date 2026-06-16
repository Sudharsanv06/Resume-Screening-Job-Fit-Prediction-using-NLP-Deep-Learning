FROM python:3.10-slim

WORKDIR /app

# curl is needed for the Render health check probe
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies ──────────────────────────────────────────────────────────────
# Copy requirements first so Docker layer-caches them.
# Rebuilds only when requirements.txt changes, not when source changes.
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
COPY src/     ./src/
COPY backend/ ./backend/

# ── Model files ───────────────────────────────────────────────────────────────
# classifier_v2.pkl and label_encoder_v2.pkl are small (<5 MB) and committed to git.
# The sentence encoder is downloaded below rather than committed (~80 MB).
COPY models/classifier_v2.pkl    ./models/classifier_v2.pkl
COPY models/label_encoder_v2.pkl ./models/label_encoder_v2.pkl

# Download and cache sentence-transformers model into the image.
# This makes the container fully self-contained — no HuggingFace call at runtime.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models/sentence_encoder')"

# ── Runtime config ────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app:/app/backend
ENV PORT=8000

EXPOSE 8000

# PORT env var is set automatically by Render — falls back to 8000 locally
CMD sh -c "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"
