"""
Text Preprocessing Module for Resume Screening
Day 2: Text cleaning and preprocessing utilities
"""

import html
import re


def clean_text(text, remove_stopwords=False):
    """
    Clean resume text for NLP processing.

    Args:
        text (str): Raw resume text
        remove_stopwords (bool): Whether to remove stopwords (optional)

    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""

    # Unescape HTML entities (e.g. &amp; -> &, &lt; -> <)
    text = html.unescape(text)

    # Remove HTML tags (e.g. <p>...</p>)
    text = re.sub(r"<[^>]*>", " ", text)

    # Remove remaining/raw HTML entity sequences if they exist
    text = text.replace("&amp;", "").replace("&lt;", "").replace("&gt;", "")

    # Replace newlines/tabs with space
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing spaces
    text = text.strip()

    # Optional: Remove stopwords
    if remove_stopwords:
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "is",
            "was",
            "are",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "should",
            "can",
            "could",
            "may",
            "might",
            "must",
            "that",
            "this",
            "these",
            "those",
        }
        words = text.split()
        text = " ".join([word for word in words if word.lower() not in stopwords])

    return text


def clean_resumes(resumes, remove_stopwords=False):
    """
    Clean a list of resume texts.

    Args:
        resumes (list): List of resume texts
        remove_stopwords (bool): Whether to remove stopwords

    Returns:
        list: List of cleaned resume texts
    """
    return [clean_text(resume, remove_stopwords) for resume in resumes]


def get_text_stats(text):
    """
    Get statistics about text.

    Args:
        text (str): Input text

    Returns:
        dict: Dictionary with text statistics
    """
    words = text.split()
    return {"char_length": len(text), "word_count": len(words), "unique_words": len(set(words))}


if __name__ == "__main__":
    # Test the cleaning function
    sample_text = (
        "Experienced Python developer with 5+ years in ML/DL! Expert in TensorFlow & PyTorch."
    )
    cleaned = clean_text(sample_text)
    print("Original:", sample_text)
    print("Cleaned:", cleaned)
    print("\nStats:", get_text_stats(cleaned))
