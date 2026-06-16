import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.exceptions import HTTPException as StarletteHTTPException

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_BACKEND_DIR)
sys.path.insert(0, _ROOT_DIR)  # enables: from src.config import ...
sys.path.insert(0, _BACKEND_DIR)  # enables: from parser import ...

from parser import extract_text

from src.career_dna import get_resume_dna
from src.config import settings
from src.logger import get_logger
from src.predict_v2 import _load_assets, is_models_loaded, predict

logger = get_logger("resume_screener")

# ── RFC 7807 status title map ──────────────────────────────────────────────────
HTTP_STATUS_TITLES = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    413: "Payload Too Large",
    415: "Unsupported Media Type",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    500: "Internal Server Error",
}

# ── Rate limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup", extra={"event": "server_ready", "mode": "lazy_load"})
    yield
    logger.info("shutdown", extra={"event": "server_stopped"})


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

MAX_FILE_BYTES = 5 * 1024 * 1024
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
MIN_WORDS = 20
MAX_BATCH = 10


# ── HTTP request logger middleware ────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "http_request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": elapsed,
            "client_ip": request.client.host if request.client else "unknown",
        },
    )
    return response


# ── RFC 7807 error handlers ───────────────────────────────────────────────────
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.warning(
        "http_error",
        extra={"path": request.url.path, "status": exc.status_code, "detail": exc.detail},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": f"https://resume-screener.com/errors/{exc.status_code}",
            "title": HTTP_STATUS_TITLES.get(exc.status_code, "Error"),
            "status": exc.status_code,
            "detail": exc.detail,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    sanitized_errors = []
    for err in errors:
        if isinstance(err, dict):
            new_err = dict(err)
            if "ctx" in new_err and isinstance(new_err["ctx"], dict):
                new_ctx = dict(new_err["ctx"])
                for k, v in new_ctx.items():
                    if isinstance(v, Exception):
                        new_ctx[k] = str(v)
                new_err["ctx"] = new_ctx
            sanitized_errors.append(new_err)
        else:
            sanitized_errors.append(err)

    logger.warning(
        "validation_error",
        extra={"path": request.url.path, "errors": str(sanitized_errors)},
    )
    return JSONResponse(
        status_code=422,
        content={
            "type": "https://resume-screener.com/errors/422",
            "title": "Validation Error",
            "status": 422,
            "detail": sanitized_errors,
        },
    )


# ── Schemas ───────────────────────────────────────────────────────────────────
class TextInput(BaseModel):
    text: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"text": "Experienced Python developer with 5 years in Django and FastAPI..."}
            ]
        }
    }


class BatchInput(BaseModel):
    texts: list[str]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"texts": ["Python developer with Django...", "Data scientist with TensorFlow..."]}
            ]
        }
    }

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v):
        if not v:
            raise ValueError("texts list cannot be empty.")
        if len(v) > MAX_BATCH:
            raise ValueError(f"Maximum {MAX_BATCH} resumes per batch request.")
        return v


# ── Helpers ───────────────────────────────────────────────────────────────────
def _base_response(result: dict) -> dict:
    """Attach model_version to every prediction response."""
    return {**result, "model_version": settings.MODEL_VERSION}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health_check():
    """Returns server status and whether ML models are in memory."""
    loaded = is_models_loaded()
    return {
        "status": "ok" if loaded else "degraded",
        "models_loaded": loaded,
        "version": app.version,
        "model_version": settings.MODEL_VERSION,
        "environment": settings.APP_ENV,
    }


@app.post("/predict/text", tags=["Prediction"])
@limiter.limit(settings.RATE_LIMIT_PREDICT)
async def predict_from_text(request: Request, body: TextInput):
    """Predict job role from raw resume text."""
    if not body.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty.")
    if len(body.text.split()) < MIN_WORDS:
        raise HTTPException(
            status_code=422,
            detail=f"Resume text is too short (minimum {MIN_WORDS} words).",
        )

    start = time.perf_counter()
    result = predict(body.text)
    dna = get_resume_dna(body.text, result["all_probs"], result["label"])
    elapsed = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        "prediction",
        extra={
            "source": "text",
            "label": result["label"],
            "confidence": round(result["confidence"], 4),
            "duration_ms": elapsed,
        },
    )

    return {
        **_base_response(result),
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

    if not text or len(text.split()) < MIN_WORDS:
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract enough text (minimum {MIN_WORDS} words).",
        )

    result = predict(text)
    dna = get_resume_dna(text, result["all_probs"], result["label"])
    elapsed = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        "prediction",
        extra={
            "source": "file",
            "file_name": file.filename,
            "label": result["label"],
            "confidence": round(result["confidence"], 4),
            "duration_ms": elapsed,
        },
    )

    return {
        **_base_response(result),
        "word_count": len(text.split()),
        "processing_time_ms": elapsed,
        "resume_dna": dna,
    }


@app.post("/predict/batch", tags=["Prediction"])
@limiter.limit(settings.RATE_LIMIT_PREDICT)
async def predict_batch(request: Request, body: BatchInput):
    """
    Predict job roles for up to 10 resumes in one request.
    Each item in results either contains a prediction or an error field
    if that individual text was too short or empty.
    """
    start = time.perf_counter()
    results = []

    for idx, text in enumerate(body.texts):
        if not text.strip() or len(text.split()) < MIN_WORDS:
            results.append(
                {
                    "index": idx,
                    "error": f"Text too short (minimum {MIN_WORDS} words) or empty.",
                    "label": None,
                }
            )
            continue

        prediction = predict(text)
        results.append(
            {
                "index": idx,
                "label": prediction["label"],
                "confidence": prediction["confidence"],
                "top3": prediction["top3"],
            }
        )

    elapsed = round((time.perf_counter() - start) * 1000, 2)

    logger.info(
        "batch_prediction",
        extra={
            "total": len(body.texts),
            "errors": sum(1 for r in results if r.get("error")),
            "duration_ms": elapsed,
        },
    )

    return {
        "results": results,
        "total": len(results),
        "model_version": settings.MODEL_VERSION,
        "processing_time_ms": elapsed,
    }
