"""
predict_v2.py
=============
Inference module for Phase 2 of the Resume Screening project.
Loads the sentence encoder, Logistic Regression classifier, and label encoder to predict
the best job fit for any input resume text with confidence scores.
"""

import os

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

# Resolve paths dynamically relative to the file location to prevent path failures
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

ENCODER_PATH = os.path.join(PROJECT_ROOT, "models", "sentence_encoder")
CLASSIFIER_PATH = os.path.join(PROJECT_ROOT, "models", "classifier_v2.pkl")
LABEL_ENCODER_PATH = os.path.join(PROJECT_ROOT, "models", "label_encoder_v2.pkl")

# Global variables for caching loaded models
_encoder = None
_classifier = None
_label_encoder = None


def _load_assets():
    """Load model preprocessors and classifiers into global memory if not already cached."""
    global _encoder, _classifier, _label_encoder

    if _encoder is None:
        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(
                f"Model directory not found at {ENCODER_PATH}. Please run train_model.py first."
            )
        print("Loading sentence transformer encoder...")
        _encoder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=ENCODER_PATH)

    if _classifier is None:
        if not os.path.exists(CLASSIFIER_PATH):
            raise FileNotFoundError(
                f"Classifier file not found at {CLASSIFIER_PATH}. Please run train_model.py first."
            )
        print("Loading classifier...")
        _classifier = joblib.load(CLASSIFIER_PATH)

    if _label_encoder is None:
        if not os.path.exists(LABEL_ENCODER_PATH):
            raise FileNotFoundError(
                f"Label encoder file not found at {LABEL_ENCODER_PATH}. Please run train_model.py first."
            )
        print("Loading label encoder...")
        _label_encoder = joblib.load(LABEL_ENCODER_PATH)


def is_models_loaded() -> bool:
    """Returns True if all three ML assets are loaded into memory."""
    return all([_encoder is not None, _classifier is not None, _label_encoder is not None])


def predict(text: str) -> dict:
    """
    Predict the job fit category for a given resume text.

    Args:
        text (str): Raw resume text

    Returns:
        dict: Containing predicted label, confidence score, and top-3 predictions.
              Format:
              {
                "label": str,
                "confidence": float,
                "top3": [{"label": str, "score": float}]
              }
    """
    if not text or not text.strip():
        raise ValueError("Input resume text cannot be empty.")

    # Ensure assets are loaded
    _load_assets()

    # 1. Generate text embedding
    embedding = _encoder.encode([text])[0]
    embedding = np.expand_dims(embedding, axis=0)  # Reshape for sklearn predict_proba (1, 384)

    # 2. Predict probabilities
    probs = _classifier.predict_proba(embedding)[0]

    # 3. Get top class index and confidence
    top_class_idx = np.argmax(probs)
    confidence = float(probs[top_class_idx])
    predicted_label = _label_encoder.classes_[top_class_idx]

    # 4. Get top 3 predicted classes
    top_3_indices = np.argsort(probs)[-3:][::-1]
    top3_list = [
        {"label": str(_label_encoder.classes_[idx]), "score": float(probs[idx])}
        for idx in top_3_indices
    ]

    # 5. Get all prediction probabilities (needed for Resume DNA)
    class_names = _label_encoder.classes_
    all_probs = {str(class_names[idx]): float(probs[idx]) for idx in range(len(class_names))}

    return {
        "label": predicted_label,
        "confidence": confidence,
        "top3": top3_list,
        "all_probs": all_probs,
    }


if __name__ == "__main__":
    print("==================================================")
    print("SMOKE TEST FOR PREDICT_V2")
    print("==================================================")

    # Sample resumes representing different job profiles
    test_resumes = [
        # Data Scientist Test
        """
        SUMMARY
        Highly motivated Data Scientist with 4+ years of industry experience. Passionate about machine
        learning, statistical modeling, and deep learning configurations.
        SKILLS
        Python, SQL, PyTorch, TensorFlow, Scikit-Learn, Pandas, NumPy, Tableau, A/B testing
        EXPERIENCE
        Machine Learning Engineer | TechCorp Solutions (2021 - Present)
        - Developed customer churn prediction models using Random Forests, decreasing attrition by 15%.
        - Designed and deployed natural language processing algorithms to categorize user support tickets.
        EDUCATION
        Bachelor of Science in Computer Science
        State University of Technology (2018)
        """,
        # Frontend Developer Test
        """
        SUMMARY
        Creative Frontend Engineer with 5 years experience designing responsive user experiences.
        Strong focus on clean layouts, component modularity, and client performance.
        SKILLS
        React, Redux, HTML5, CSS3, JavaScript, TypeScript, Webpack, Vite, Tailwind CSS, Git
        EXPERIENCE
        React Web Developer | WebFlow Digital (2020 - Present)
        - Rebuilt company customer portal using React and Vite, reducing initial load times by 40%.
        - Developed reusable stateful UI component libraries shared across three engineering teams.
        EDUCATION
        MS in Software Engineering
        Metropolitan University (2019)
        """,
        # Security Analyst Test
        """
        SUMMARY
        Certified Information Security Specialist with 3+ years experience identifying and patching network
        vulnerabilities. Strong understanding of compliance frameworks and risk mitigation systems.
        SKILLS
        Penetration Testing, Kali Linux, Metasploit, Wireshark, SIEM logging, Firewalls, Cryptography
        EXPERIENCE
        Cybersecurity Analyst | SecureNet Solutions (2022 - Present)
        - Conducted threat scanning audits and system penetration tests across corporate environments.
        - Deployed network firewalls and configured real-time corporate SIEM dashboard alerts.
        EDUCATION
        Bachelor of Science in Information Systems
        National Science University (2021)
        """,
    ]

    try:
        # Load models
        _load_assets()

        for i, resume in enumerate(test_resumes, 1):
            print(f"\nAnalyzing Sample Resume #{i}:")
            print("-" * 50)
            res = predict(resume)
            print(f"Predicted Role: {res['label']}")
            print(f"Confidence: {res['confidence'] * 100:.2f}%")
            print("Top 3 Candidates:")
            for rank, item in enumerate(res["top3"], 1):
                print(f"  {rank}. {item['label']:25s} - {item['score'] * 100:.2f}%")
            print("-" * 50)

    except Exception as e:
        print(f"Error executing smoke test: {e}")
