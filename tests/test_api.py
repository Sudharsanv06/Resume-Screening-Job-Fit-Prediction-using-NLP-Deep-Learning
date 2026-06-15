"""Integration tests for all FastAPI endpoints."""
import pytest

LONG_RESUME = (
    "Senior Python Developer with 6 years of professional experience. "
    "Expert in Django REST Framework, Flask, FastAPI, Celery, PostgreSQL, "
    "Redis, Docker, Kubernetes, AWS Lambda, S3, EC2, and CI/CD pipelines. "
    "Designed and shipped microservices handling 2M daily requests. "
    "Led cross-functional teams using agile methodologies and TDD practices."
)

SHORT_TEXT = "Python developer."   # fewer than 20 words — should be rejected


class TestHealthEndpoint:

    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_status_is_ok(self, client):
        assert client.get("/health").json()["status"] == "ok"

    def test_models_loaded_is_true(self, client):
        assert client.get("/health").json()["models_loaded"] is True

    def test_has_version_field(self, client):
        assert "version" in client.get("/health").json()

    def test_has_environment_field(self, client):
        assert "environment" in client.get("/health").json()


class TestPredictText:

    # ── Happy path ────────────────────────────────────────────────────────────
    def test_valid_text_returns_200(self, client):
        assert client.post("/predict/text", json={"text": LONG_RESUME}).status_code == 200

    def test_response_has_label(self, client):
        data = client.post("/predict/text", json={"text": LONG_RESUME}).json()
        assert "label" in data
        assert isinstance(data["label"], str)
        assert len(data["label"]) > 0

    def test_response_confidence_between_0_and_1(self, client):
        data = client.post("/predict/text", json={"text": LONG_RESUME}).json()
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_response_has_exactly_3_top_predictions(self, client):
        data = client.post("/predict/text", json={"text": LONG_RESUME}).json()
        assert "top3" in data
        assert len(data["top3"]) == 3

    def test_top3_items_have_label_and_score(self, client):
        data = client.post("/predict/text", json={"text": LONG_RESUME}).json()
        for item in data["top3"]:
            assert "label" in item
            assert "score" in item
            assert 0.0 <= item["score"] <= 1.0

    def test_response_has_positive_word_count(self, client):
        data = client.post("/predict/text", json={"text": LONG_RESUME}).json()
        assert "word_count" in data
        assert data["word_count"] > 0

    def test_response_has_processing_time(self, client):
        data = client.post("/predict/text", json={"text": LONG_RESUME}).json()
        assert "processing_time_ms" in data

    def test_response_has_resume_dna(self, client):
        data = client.post("/predict/text", json={"text": LONG_RESUME}).json()
        assert "resume_dna" in data
        assert isinstance(data["resume_dna"], dict)

    def test_resume_dna_has_cluster_scores(self, client):
        data = client.post("/predict/text", json={"text": LONG_RESUME}).json()
        assert "cluster_scores" in data["resume_dna"]

    # ── Rejection cases ───────────────────────────────────────────────────────
    def test_empty_text_returns_422(self, client):
        assert client.post("/predict/text", json={"text": ""}).status_code == 422

    def test_whitespace_only_returns_422(self, client):
        assert client.post("/predict/text", json={"text": "     "}).status_code == 422

    def test_too_short_text_returns_422(self, client):
        assert client.post("/predict/text", json={"text": SHORT_TEXT}).status_code == 422

    def test_missing_text_field_returns_422(self, client):
        assert client.post("/predict/text", json={}).status_code == 422

    def test_wrong_field_name_returns_422(self, client):
        assert client.post("/predict/text", json={"resume": LONG_RESUME}).status_code == 422


class TestPredictFile:

    # ── Happy path ────────────────────────────────────────────────────────────
    def test_pdf_upload_returns_200(self, client, pdf_bytes):
        response = client.post(
            "/predict/file",
            files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 200

    def test_docx_upload_returns_200(self, client, docx_bytes):
        mime     = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        response = client.post(
            "/predict/file",
            files={"file": ("resume.docx", docx_bytes, mime)},
        )
        assert response.status_code == 200

    def test_pdf_response_has_label(self, client, pdf_bytes):
        data = client.post(
            "/predict/file",
            files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        ).json()
        assert "label" in data

    def test_pdf_response_has_word_count(self, client, pdf_bytes):
        data = client.post(
            "/predict/file",
            files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        ).json()
        assert "word_count" in data
        assert data["word_count"] > 0

    def test_pdf_response_has_resume_dna(self, client, pdf_bytes):
        data = client.post(
            "/predict/file",
            files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        ).json()
        assert "resume_dna" in data

    # ── Rejection cases ───────────────────────────────────────────────────────
    def test_plain_text_file_returns_415(self, client):
        response = client.post(
            "/predict/file",
            files={"file": ("resume.txt", b"Some resume text here.", "text/plain")},
        )
        assert response.status_code == 415

    def test_image_file_returns_415(self, client):
        response = client.post(
            "/predict/file",
            files={"file": ("photo.png", b"\x89PNG\r\n", "image/png")},
        )
        assert response.status_code == 415

    def test_file_over_5mb_returns_413(self, client):
        big = b"x" * (5 * 1024 * 1024 + 1)  # 5 MB + 1 byte
        response = client.post(
            "/predict/file",
            files={"file": ("big.pdf", big, "application/pdf")},
        )
        assert response.status_code == 413

    def test_no_file_attached_returns_422(self, client):
        assert client.post("/predict/file").status_code == 422
