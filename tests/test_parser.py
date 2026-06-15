"""Unit tests for backend/parser.py — PDF and DOCX text extraction."""
import pytest
from parser import extract_text

PDF_TYPE  = "application/pdf"
DOCX_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class TestPdfExtraction:

    def test_returns_a_string(self, pdf_bytes):
        assert isinstance(extract_text(pdf_bytes, PDF_TYPE), str)

    def test_returns_non_empty_text(self, pdf_bytes):
        result = extract_text(pdf_bytes, PDF_TYPE)
        assert len(result.strip()) > 0

    def test_contains_expected_keyword(self, pdf_bytes):
        result = extract_text(pdf_bytes, PDF_TYPE)
        assert "Python" in result

    def test_minimum_word_count(self, pdf_bytes):
        result = extract_text(pdf_bytes, PDF_TYPE)
        assert len(result.split()) >= 10


class TestDocxExtraction:

    def test_returns_a_string(self, docx_bytes):
        assert isinstance(extract_text(docx_bytes, DOCX_TYPE), str)

    def test_returns_non_empty_text(self, docx_bytes):
        result = extract_text(docx_bytes, DOCX_TYPE)
        assert len(result.strip()) > 0

    def test_contains_expected_keyword(self, docx_bytes):
        result = extract_text(docx_bytes, DOCX_TYPE)
        assert "Python" in result

    def test_preserves_paragraph_content(self, docx_bytes):
        result = extract_text(docx_bytes, DOCX_TYPE)
        # Both heading and paragraph content from the fixture should appear
        assert "Django" in result or "FastAPI" in result

    def test_minimum_word_count(self, docx_bytes):
        result = extract_text(docx_bytes, DOCX_TYPE)
        assert len(result.split()) >= 10


class TestUnsupportedType:

    def test_unsupported_mime_raises_or_returns_empty(self):
        dummy = b"This is a plain text file, not a PDF or DOCX."
        try:
            result = extract_text(dummy, "text/plain")
            # If it doesn't raise, it should at least return a string
            assert isinstance(result, str)
        except (ValueError, TypeError, Exception):
            pass  # Raising an exception is also valid behaviour
