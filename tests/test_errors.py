"""Tests verifying all error responses follow RFC 7807 Problem Details format."""
import pytest

LONG_RESUME = (
    "Senior Python developer with Django REST Framework, FastAPI, PostgreSQL, "
    "Redis, Docker, Kubernetes, AWS, GitHub Actions, microservices, and pytest. "
    "Led teams of 5 engineers delivering production APIs serving 2M daily requests."
)


class TestRFC7807Format:
    """Every error response must have: type, title, status, detail."""

    def _assert_rfc7807(self, data: dict, expected_status: int):
        assert "type"   in data, "Missing 'type' field"
        assert "title"  in data, "Missing 'title' field"
        assert "status" in data, "Missing 'status' field"
        assert "detail" in data, "Missing 'detail' field"
        assert data["status"] == expected_status

    # ── 415 Unsupported Media Type ────────────────────────────────────────────
    def test_415_has_rfc7807_type(self, client):
        response = client.post(
            "/predict/file",
            files={"file": ("resume.txt", b"some text content here", "text/plain")},
        )
        assert response.status_code == 415
        self._assert_rfc7807(response.json(), 415)

    def test_415_type_url_contains_status_code(self, client):
        response = client.post(
            "/predict/file",
            files={"file": ("resume.txt", b"some text content here", "text/plain")},
        )
        assert "415" in response.json()["type"]

    def test_415_title_is_unsupported_media_type(self, client):
        response = client.post(
            "/predict/file",
            files={"file": ("resume.txt", b"some text content here", "text/plain")},
        )
        assert "Unsupported" in response.json()["title"]

    # ── 413 Payload Too Large ─────────────────────────────────────────────────
    def test_413_has_rfc7807_format(self, client):
        big      = b"x" * (5 * 1024 * 1024 + 1)
        response = client.post(
            "/predict/file",
            files={"file": ("big.pdf", big, "application/pdf")},
        )
        assert response.status_code == 413
        self._assert_rfc7807(response.json(), 413)

    def test_413_detail_mentions_limit(self, client):
        big      = b"x" * (5 * 1024 * 1024 + 1)
        response = client.post(
            "/predict/file",
            files={"file": ("big.pdf", big, "application/pdf")},
        )
        assert "5" in response.json()["detail"]

    # ── 422 from empty text ───────────────────────────────────────────────────
    def test_422_empty_text_has_rfc7807_format(self, client):
        response = client.post("/predict/text", json={"text": ""})
        assert response.status_code == 422
        data     = response.json()
        assert "type"   in data
        assert "status" in data
        assert data["status"] == 422

    def test_422_too_short_text_has_detail(self, client):
        response = client.post("/predict/text", json={"text": "Too short."})
        data     = response.json()
        assert "detail" in data
        assert data["detail"]  # must not be empty

    # ── 422 from missing Pydantic field ───────────────────────────────────────
    def test_422_pydantic_error_has_rfc7807_fields(self, client):
        response = client.post("/predict/text", json={})
        assert response.status_code == 422
        data     = response.json()
        assert "type"   in data
        assert "title"  in data
        assert "status" in data
        assert "detail" in data

    def test_422_pydantic_error_detail_is_list(self, client):
        # Pydantic field errors come back as a list of error objects
        response = client.post("/predict/text", json={})
        data     = response.json()
        assert isinstance(data["detail"], list)

    # ── 422 from batch validation ─────────────────────────────────────────────
    def test_422_batch_too_many_has_rfc7807_format(self, client):
        texts    = ["Python developer experience." * 5] * 11
        response = client.post("/predict/batch", json={"texts": texts})
        assert response.status_code == 422
        self._assert_rfc7807(response.json(), 422)

    def test_422_batch_empty_list_has_detail(self, client):
        response = client.post("/predict/batch", json={"texts": []})
        data     = response.json()
        assert "detail" in data


class TestModelVersionInResponses:
    """model_version must appear in all prediction responses."""

    def test_text_response_has_model_version(self, client):
        data = client.post("/predict/text", json={"text": LONG_RESUME}).json()
        assert "model_version" in data

    def test_text_model_version_is_string(self, client):
        data = client.post("/predict/text", json={"text": LONG_RESUME}).json()
        assert isinstance(data["model_version"], str)

    def test_file_response_has_model_version(self, client, pdf_bytes):
        data = client.post(
            "/predict/file",
            files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        ).json()
        assert "model_version" in data

    def test_batch_response_has_model_version(self, client):
        data = client.post("/predict/batch", json={"texts": [LONG_RESUME]}).json()
        assert "model_version" in data

    def test_health_response_has_model_version(self, client):
        data = client.get("/health").json()
        assert "model_version" in data
