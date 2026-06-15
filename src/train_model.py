"""
train_model.py
==============
Model training script for Phase 2.
Loads dataset/resumes_v2.csv, encodes text using sentence-transformers (all-MiniLM-L6-v2),
trains a Logistic Regression classifier on top, and serializes the assets.
Saves metrics to results/metrics_v2.txt.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer

def train_pipeline():
    # Define file paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    csv_path = os.path.join(project_root, "dataset", "resumes_v2.csv")
    encoder_save_path = os.path.join(project_root, "models", "sentence_encoder")
    classifier_save_path = os.path.join(project_root, "models", "classifier.pkl")
    label_encoder_save_path = os.path.join(project_root, "models", "label_encoder_v2.pkl")
    metrics_save_path = os.path.join(project_root, "results", "metrics_v2.txt")
    
    # 1. Load the dataset
    print(f"Loading dataset from {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run Phase 1 first.")
    
    df = pd.read_csv(csv_path)
    X_raw = df["resume_text"].values
    y_raw = df["category"].values
    
    # 2. Encode targets
    print("Encoding target labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    
    # 3. Stratified Train/Val/Test Split (70% Train, 15% Val, 15% Test)
    print("Splitting dataset (70% train / 15% val / 15% test)...")
    # First split off 15% for test
    X_train_val, X_test_raw, y_train_val, y_test = train_test_split(
        X_raw, y_encoded, test_size=0.15, stratify=y_encoded, random_state=42
    )
    # Then split 15% validation from the 85% train_val (15/85 = ~17.65%)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=(15.0/85.0), stratify=y_train_val, random_state=42
    )
    
    print(f"Dataset Split Sizes:")
    print(f"   - Train: {len(X_train_raw)} samples")
    print(f"   - Val:   {len(X_val_raw)} samples")
    print(f"   - Test:  {len(X_test_raw)} samples")
    
    # 4. Generate Sentence Embeddings
    print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Encoding training data text embeddings...")
    X_train = encoder.encode(X_train_raw, show_progress_bar=True)
    
    print("Encoding validation data text embeddings...")
    X_val = encoder.encode(X_val_raw, show_progress_bar=False)
    
    print("Encoding testing data text embeddings...")
    X_test = encoder.encode(X_test_raw, show_progress_bar=False)
    
    # 5. Train Logistic Regression Classifier
    print("Training Logistic Regression classifier (max_iter=1000)...")
    classifier = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    classifier.fit(X_train, y_train)
    
    # 6. Evaluate Model
    val_preds = classifier.predict(X_val)
    test_preds = classifier.predict(X_test)
    
    val_acc = accuracy_score(y_val, val_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc * 100:.2f}%)")
    print(f"Test Accuracy:       {test_acc:.4f} ({test_acc * 100:.2f}%)")
    
    # Classification Report
    class_names = label_encoder.classes_
    report_str = classification_report(y_test, test_preds, target_names=class_names)
    print("\nTest Classification Report:")
    print(report_str)
    
    # 7. Save Assets
    print(f"Saving sentence encoder to {encoder_save_path}...")
    os.makedirs(os.path.dirname(encoder_save_path), exist_ok=True)
    encoder.save(encoder_save_path)
    
    print(f"Saving classifier to {classifier_save_path}...")
    joblib.dump(classifier, classifier_save_path)
    
    print(f"Saving label encoder to {label_encoder_save_path}...")
    joblib.dump(label_encoder, label_encoder_save_path)
    
    # Save metrics report
    print(f"Saving metrics report to {metrics_save_path}...")
    os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)
    
    metrics_content = (
        "MODEL PERFORMANCE METRICS - V2\n"
        "====================================\n"
        f"Base Model: SentenceTransformer (all-MiniLM-L6-v2)\n"
        f"Classifier: LogisticRegression(max_iter=1000, C=1.0)\n\n"
        "Dataset Info:\n"
        f"- Total samples: {len(df)}\n"
        f"- Training samples: {len(X_train_raw)}\n"
        f"- Validation samples: {len(X_val_raw)}\n"
        f"- Testing samples: {len(X_test_raw)}\n\n"
        "Performance:\n"
        f"- Validation Accuracy: {val_acc * 100:.2f}%\n"
        f"- Test Accuracy: {test_acc * 100:.2f}%\n\n"
        "Test Classification Report:\n"
        "---------------------------\n"
        f"{report_str}"
    )
    
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        f.write(metrics_content)
        
    print("Model training pipeline complete!")

if __name__ == "__main__":
    train_pipeline()
