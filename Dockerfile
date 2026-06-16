FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt

# ── THE FIX ───────────────────────────────────────────────────────────────────
# Install CPU-only PyTorch BEFORE the main requirements.
# Without this, pip resolves torch to the GPU version and pulls 2.5GB of CUDA
# libraries (cudnn, cublas, cufft, cusolver, nccl, triton...) which kills the
# build on Render's free 512MB tier. CPU-only torch is ~200MB and works fine
# for sentence-transformers inference on CPU.
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Now install everything else — sentence-transformers sees torch already present
# and skips re-downloading it, avoiding all GPU packages entirely.
RUN pip install --no-cache-dir -r requirements.txt

# ── Source code ───────────────────────────────────────────────────────────────
COPY src/     ./src/
COPY backend/ ./backend/

# ── Model files ───────────────────────────────────────────────────────────────
COPY models/classifier_v2.pkl    ./models/classifier_v2.pkl
COPY models/label_encoder_v2.pkl ./models/label_encoder_v2.pkl

# Download and cache sentence-transformer model into the image at build time.
# Uses the CPU torch already installed above — no GPU needed.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2', cache_folder='./models/sentence_encoder')"

# ── Runtime ───────────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app:/app/backend
ENV PORT=8000

EXPOSE 8000

CMD sh -c "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"
