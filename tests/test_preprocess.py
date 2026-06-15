"""Unit tests for src/preprocess.py — text cleaning operations."""
import pytest
from src.preprocess import clean_text   # adjust name if yours differs


class TestCleanText:

    def test_strips_html_tags(self):
        raw    = "<p>Experienced <b>Python</b> developer with <i>Django</i>.</p>"
        result = clean_text(raw)
        assert "<p>"  not in result
        assert "<b>"  not in result
        assert "<i>"  not in result

    def test_keeps_meaningful_words_after_html_strip(self):
        raw    = "<p>Experienced <b>Python</b> developer with <i>Django</i>.</p>"
        result = clean_text(raw)
        assert "Python" in result
        assert "Django" in result

    def test_strips_html_entities(self):
        raw    = "5 years &amp; strong experience in Java &lt;frameworks&gt;."
        result = clean_text(raw)
        assert "&amp;" not in result
        assert "&lt;"  not in result
        assert "&gt;"  not in result

    def test_collapses_extra_spaces(self):
        raw    = "Python   developer   with    many   spaces."
        result = clean_text(raw)
        assert "  " not in result

    def test_collapses_excess_newlines(self):
        raw    = "Skills:\n\n\nPython\n\n\nDjango"
        result = clean_text(raw)
        assert "\n\n\n" not in result

    def test_handles_empty_string(self):
        result = clean_text("")
        assert isinstance(result, str)
        assert result.strip() == ""

    def test_plain_text_passes_through(self):
        raw    = "Python developer with Docker and AWS experience."
        result = clean_text(raw)
        assert "Python" in result
        assert "Docker" in result
        assert "AWS"    in result

    def test_returns_string_type(self):
        assert isinstance(clean_text("<p>Hello world</p>"), str)

    def test_does_not_return_none(self):
        assert clean_text("Some resume text.") is not None

    def test_handles_only_whitespace(self):
        result = clean_text("    \n\n\t   ")
        assert isinstance(result, str)

    def test_handles_nested_html(self):
        raw    = "<div><ul><li><strong>Python</strong></li></ul></div>"
        result = clean_text(raw)
        assert "<" not in result
        assert "Python" in result

    def test_preserves_numbers(self):
        raw    = "<p>10 years experience with 3 programming languages.</p>"
        result = clean_text(raw)
        assert "10" in result
        assert "3"  in result

    def test_remove_stopwords(self):
        raw = "The Python and Django developer candidate."
        result = clean_text(raw, remove_stopwords=True)
        tokens = [t.lower() for t in result.split()]
        assert "the" not in tokens
        assert "and" not in tokens
        assert "Python" in result
        assert "Django" in result



def test_clean_resumes():
    from src.preprocess import clean_resumes
    resumes = [
        "<p>Python developer</p>",
        "Django backend engineer &amp; coder"
    ]
    results = clean_resumes(resumes, remove_stopwords=True)
    assert len(results) == 2
    assert "Python" in results[0]
    assert "Django" in results[1]
    assert "backend" in results[1]
    assert "&amp;" not in results[1]


def test_get_text_stats():
    from src.preprocess import get_text_stats
    text = "Python developer Django"
    stats = get_text_stats(text)
    assert stats["word_count"] == 3
    assert stats["char_length"] == len(text)
    assert stats["unique_words"] == 3

