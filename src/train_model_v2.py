"""
train_model_v2.py
=================
Retrains the ResumeAI classification pipeline on the new high-quality dataset.
Generates SentenceTransformer embeddings, splits stratified 70/15/15, tunes
Logistic Regression hyperparameter C using GridSearchCV, and serializes classifier and encoders.
"""

from pathlib import Path

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

# Resolve paths relative to file location
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "resumes_final.csv"
ENCODER_PATH = PROJECT_ROOT / "models" / "sentence_encoder"
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "classifier_v2.pkl"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder_v2.pkl"
METRICS_PATH = PROJECT_ROOT / "results" / "metrics_v3.txt"


def train_pipeline() -> None:
    """
    Executes the model training, hyperparameter tuning, evaluation, and serialization pipeline.
    """
    # 1. Load dataset
    print(f"Loading merged dataset from: {DATASET_PATH}")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Final dataset not found at {DATASET_PATH}. Please run merge_datasets.py first."
        )

    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    X_raw = df["resume_text"].values
    y_raw = df["category"].values

    # 2. Encode targets
    print("Encoding target labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)

    # 3. Stratified Split (70/15/15)
    print("Splitting dataset into 70% train / 15% val / 15% test...")
    # First split off 15% for test
    X_train_val_raw, X_test_raw, y_train_val, y_test = train_test_split(
        X_raw, y_encoded, test_size=0.15, stratify=y_encoded, random_state=42
    )
    # Then split 15% validation from the 85% train_val (15/85 = ~17.65%)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_val_raw, y_train_val, test_size=(15.0 / 85.0), stratify=y_train_val, random_state=42
    )

    print("Dataset Split Sizes:")
    print(f"  - Train Size:      {len(X_train_raw)} samples")
    print(f"  - Validation Size: {len(X_val_raw)} samples")
    print(f"  - Testing Size:    {len(X_test_raw)} samples")

    # 4. SentenceTransformer Encoding
    if ENCODER_PATH.exists():
        print(f"Loading SentenceTransformer from local path: {ENCODER_PATH}")
        encoder = SentenceTransformer(str(ENCODER_PATH))
    else:
        print("Loading SentenceTransformer model 'all-MiniLM-L6-v2' from Hub...")
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        # Save local copy if it didn't exist
        ENCODER_PATH.mkdir(parents=True, exist_ok=True)
        encoder.save(str(ENCODER_PATH))

    print("\nEncoding training text embeddings...")
    X_train = encoder.encode(X_train_raw, show_progress_bar=True)

    print("Encoding validation text embeddings...")
    X_val = encoder.encode(X_val_raw, show_progress_bar=False)

    print("Encoding testing text embeddings...")
    X_test = encoder.encode(X_test_raw, show_progress_bar=False)

    # 5. Grid Search for Logistic Regression
    print("\nRunning GridSearchCV on Logistic Regression (tuning C parameter)...")
    base_clf = LogisticRegression(max_iter=1000, random_state=42)
    param_grid = {"C": [0.5, 1.0, 5.0, 10.0]}

    grid_search = GridSearchCV(
        estimator=base_clf, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_clf = grid_search.best_estimator_
    best_C = float(grid_search.best_params_["C"])
    print(f"Best parameter found: C = {best_C}")

    # 6. Evaluation
    val_preds = best_clf.predict(X_val)
    test_preds = best_clf.predict(X_test)

    val_acc = accuracy_score(y_val, val_preds)
    test_acc = accuracy_score(y_test, test_preds)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Best C Hyperparameter: {best_C}")
    print(f"Validation Accuracy:   {val_acc * 100:.2f}%")
    print(f"Test Accuracy:         {test_acc * 100:.2f}%")

    class_names = [str(c) for c in label_encoder.classes_]
    report_str = classification_report(y_test, test_preds, target_names=class_names)
    print("\nTest Classification Report:")
    print("-" * 50)
    print(report_str)
    print("=" * 50)

    # 7. Serialize Assets
    print(f"\nSaving classifier to: {CLASSIFIER_PATH}")
    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_clf, CLASSIFIER_PATH)

    print(f"Saving label encoder to: {LABEL_ENCODER_PATH}")
    LABEL_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    # Save Metrics File
    metrics_content = (
        "MODEL RETRAINING & PERFORMANCE METRICS - V3\n"
        "===========================================\n"
        "Base Model: SentenceTransformer (all-MiniLM-L6-v2)\n"
        f"Classifier: GridSearchCV tuned LogisticRegression(max_iter=1000, random_state=42)\n"
        f"Best Hyperparameter (C): {best_C}\n\n"
        "Dataset Split Info:\n"
        f"  - Total clean samples: {len(df)}\n"
        f"  - Training samples: {len(X_train_raw)}\n"
        f"  - Validation samples: {len(X_val_raw)}\n"
        f"  - Testing samples: {len(X_test_raw)}\n\n"
        "Performance Summary:\n"
        f"  - Validation Accuracy: {val_acc * 100:.2f}%\n"
        f"  - Test Accuracy:       {test_acc * 100:.2f}%\n\n"
        "Test Classification Report:\n"
        "---------------------------\n"
        f"{report_str}"
    )

    print(f"Saving metrics report to: {METRICS_PATH}")
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(metrics_content)

    print("\nModel retraining pipeline successfully complete!")


if __name__ == "__main__":
    train_pipeline()
