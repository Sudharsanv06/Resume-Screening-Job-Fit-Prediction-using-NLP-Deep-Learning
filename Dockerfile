FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Memory optimisation for Render free tier (512MB RAM) ─────────────────────
# Prevents PyTorch and tokenizers from spinning up extra threads
# that each consume memory, keeping us under the 512MB limit.
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_OFFLINE=0
ENV PYTHONPATH=/app:/app/backend
ENV PORT=8000

COPY backend/requirements.txt ./requirements.txt

# CPU-only torch — no CUDA libraries, ~190MB instead of 2.5GB
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY src/     ./src/
COPY backend/ ./backend/
COPY models/classifier_v2.pkl    ./models/classifier_v2.pkl
COPY models/label_encoder_v2.pkl ./models/label_encoder_v2.pkl

# ── Download model at BUILD time so runtime has zero network overhead ─────────
# The TRANSFORMERS_CACHE env keeps it in a predictable location.
ENV SENTENCE_TRANSFORMERS_HOME=/app/models/sentence_encoder
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8000

CMD sh -c "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"
