"""Tests for the POST /predict/batch endpoint."""
from unittest.mock import patch

import pytest

RESUME_A = (
    "Senior Python developer with 6 years experience in Django REST Framework, "
    "Flask, FastAPI, PostgreSQL, Redis, Docker, Kubernetes, AWS, GitHub Actions CI/CD, "
    "microservices architecture, and test-driven development using pytest."
)
RESUME_B = (
    "Data Scientist with 4 years applying machine learning and deep learning. "
    "Expert in Python, TensorFlow, PyTorch, scikit-learn, Pandas, NumPy, SQL, "
    "Jupyter, MLflow, and deploying models to AWS SageMaker production environments."
)
SHORT = "Python developer."


class TestBatchHappyPath:

    def test_single_text_returns_200(self, client):
        response = client.post("/predict/batch", json={"texts": [RESUME_A]})
        assert response.status_code == 200

    def test_two_texts_returns_200(self, client):
        response = client.post("/predict/batch", json={"texts": [RESUME_A, RESUME_B]})
        assert response.status_code == 200

    def test_response_has_results_key(self, client):
        data = client.post("/predict/batch", json={"texts": [RESUME_A]}).json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_results_length_matches_input(self, client):
        data = client.post("/predict/batch", json={"texts": [RESUME_A, RESUME_B]}).json()
        assert data["total"] == 2
        assert len(data["results"]) == 2

    def test_each_result_has_index(self, client):
        data = client.post("/predict/batch", json={"texts": [RESUME_A, RESUME_B]}).json()
        for i, item in enumerate(data["results"]):
            assert item["index"] == i

    def test_each_result_has_label(self, client):
        data = client.post("/predict/batch", json={"texts": [RESUME_A]}).json()
        assert data["results"][0]["label"] is not None

    def test_each_result_has_confidence(self, client):
        data = client.post("/predict/batch", json={"texts": [RESUME_A]}).json()
        assert "confidence" in data["results"][0]
        assert 0.0 <= data["results"][0]["confidence"] <= 1.0

    def test_each_result_has_top3(self, client):
        data = client.post("/predict/batch", json={"texts": [RESUME_A]}).json()
        top3 = data["results"][0]["top3"]
        assert isinstance(top3, list)
        assert len(top3) == 3

    def test_response_has_total_field(self, client):
        data = client.post("/predict/batch", json={"texts": [RESUME_A, RESUME_B]}).json()
        assert "total" in data
        assert data["total"] == 2

    def test_response_has_model_version(self, client):
        data = client.post("/predict/batch", json={"texts": [RESUME_A]}).json()
        assert "model_version" in data
        assert isinstance(data["model_version"], str)

    def test_response_has_processing_time(self, client):
        data = client.post("/predict/batch", json={"texts": [RESUME_A]}).json()
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] >= 0

    def test_max_10_texts_returns_200(self, client):
        texts    = [RESUME_A] * 10
        response = client.post("/predict/batch", json={"texts": texts})
        assert response.status_code == 200


class TestBatchShortTextHandling:

    def test_short_text_does_not_crash_batch(self, client):
        data = client.post(
            "/predict/batch", json={"texts": [RESUME_A, SHORT]}
        ).json()
        assert data["total"] == 2

    def test_short_text_result_has_error_field(self, client):
        data = client.post(
            "/predict/batch", json={"texts": [RESUME_A, SHORT]}
        ).json()
        short_result = data["results"][1]
        assert "error" in short_result

    def test_short_text_result_has_null_label(self, client):
        data = client.post(
            "/predict/batch", json={"texts": [SHORT]}
        ).json()
        assert data["results"][0]["label"] is None

    def test_valid_entries_still_predict_alongside_short(self, client):
        data = client.post(
            "/predict/batch", json={"texts": [RESUME_A, SHORT]}
        ).json()
        valid_result = data["results"][0]
        assert valid_result["label"] is not None
        assert "error" not in valid_result

    def test_all_short_texts_still_returns_200(self, client):
        response = client.post("/predict/batch", json={"texts": [SHORT, SHORT]})
        assert response.status_code == 200


class TestBatchRejectionCases:

    def test_empty_texts_list_returns_422(self, client):
        response = client.post("/predict/batch", json={"texts": []})
        assert response.status_code == 422

    def test_11_texts_returns_422(self, client):
        texts    = [RESUME_A] * 11
        response = client.post("/predict/batch", json={"texts": texts})
        assert response.status_code == 422

    def test_missing_texts_field_returns_422(self, client):
        response = client.post("/predict/batch", json={})
        assert response.status_code == 422

    def test_texts_not_a_list_returns_422(self, client):
        response = client.post("/predict/batch", json={"texts": RESUME_A})
        assert response.status_code == 422
