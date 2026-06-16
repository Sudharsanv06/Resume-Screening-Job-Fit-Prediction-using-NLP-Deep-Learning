import os

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


class Settings:
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_HOST: str = os.getenv("APP_HOST", "127.0.0.1")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

    CLASSIFIER_PATH: str = os.getenv("CLASSIFIER_PATH", "models/classifier_v2.pkl")
    LABEL_ENCODER_PATH: str = os.getenv("LABEL_ENCODER_PATH", "models/label_encoder_v2.pkl")
    SENTENCE_ENCODER_PATH: str = os.getenv("SENTENCE_ENCODER_PATH", "models/sentence_encoder")

    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,https://susanhuggingface7-resume-screener-api.hf.space").split(",")

    RATE_LIMIT_PREDICT: str = os.getenv("RATE_LIMIT_PREDICT", "10/minute")

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "v2")


settings = Settings()
