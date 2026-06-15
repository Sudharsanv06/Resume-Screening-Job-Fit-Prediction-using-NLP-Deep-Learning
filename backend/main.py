import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# Ensure the project root is on the path so src/ imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.predict_v2 import predict, is_models_loaded, _load_assets
from src.career_dna import get_resume_dna
from parser import extract_text

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("resume_screener")

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── Lifespan (model preload) ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading ML models into memory...")
    _load_assets()
    logger.info("Models loaded. Server ready.")
    yield
    logger.info("Shutting down.")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Resume Screener API",
    description="Classifies resumes into 14 IT & business roles using ML.",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


# ── Schemas ───────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

    model_config = {
        "json_schema_extra": {
            "examples": [{"text": "Experienced Python developer with 5 years in Django..."}]
        }
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """Returns server status and whether ML models are loaded."""
    loaded = is_models_loaded()
    return {
        "status": "ok" if loaded else "degraded",
        "models_loaded": loaded,
        "version": app.version,
        "environment": settings.APP_ENV,
    }


@app.post("/predict/text", tags=["Prediction"])
@limiter.limit(settings.RATE_LIMIT_PREDICT)
async def predict_from_text(request: Request, body: TextInput):
    """Predict job role from raw resume text."""
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty.")

    if len(body.text.split()) < 20:
        raise HTTPException(status_code=422, detail="Resume text is too short (minimum 20 words).")

    start = time.perf_counter()
    result = predict(body.text)
    dna = get_resume_dna(body.text, result["all_probs"], result["label"])
    elapsed = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        f"Text prediction | label={result['label']} "
        f"confidence={result['confidence']:.2%} | {elapsed}ms"
    )

    return {
        **result,
        "word_count": len(body.text.split()),
        "processing_time_ms": elapsed,
        "resume_dna": dna,
    }


@app.post("/predict/file", tags=["Prediction"])
@limiter.limit(settings.RATE_LIMIT_PREDICT)
async def predict_from_file(request: Request, file: UploadFile = File(...)):
    """Predict job role from an uploaded PDF or DOCX file."""
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Only PDF and DOCX are accepted.",
        )

    contents = await file.read()

    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 5 MB limit.")

    start = time.perf_counter()
    text = extract_text(contents, file.content_type)

    if not text or len(text.split()) < 20:
        raise HTTPException(
            status_code=422,
            detail="Could not extract enough text from the file (minimum 20 words).",
        )

    result = predict(text)
    dna = get_resume_dna(text, result["all_probs"], result["label"])
    elapsed = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        f"File prediction | file={file.filename} label={result['label']} "
        f"confidence={result['confidence']:.2%} | {elapsed}ms"
    )

    return {
        **result,
        "word_count": len(text.split()),
        "processing_time_ms": elapsed,
        "resume_dna": dna,
    }