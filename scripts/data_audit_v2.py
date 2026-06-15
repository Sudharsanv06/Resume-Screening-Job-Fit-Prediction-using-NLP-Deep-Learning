"""
data_audit_v2.py
================
Audits the final merged dataset (resumes_final.csv).
Verifies sample sizes across all 14 target classes and computes word count statistics.
"""

from pathlib import Path
import pandas as pd

# Dynamic path resolution
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "resumes_final.csv"

def audit_dataset() -> None:
    """
    Loads dataset/resumes_final.csv and runs audit metrics.
    """
    print(f"Loading final dataset from: {DATASET_PATH}...")
    if not DATASET_PATH.exists():
        print(f"FAIL: Merged dataset not found at {DATASET_PATH}. Please run merge_datasets.py first.")
        return
        
    df = pd.read_csv(DATASET_PATH, encoding="utf-8")
    
    total_rows = len(df)
    print("\n" + "=" * 50)
    print("DATA AUDIT REPORT - V2")
    print("=" * 50)
    print(f"Total Rows: {total_rows}")
    
    # Calculate word count stats
    df["word_count"] = df["resume_text"].apply(lambda x: len(str(x).split()))
    min_words = int(df["word_count"].min())
    max_words = int(df["word_count"].max())
    avg_words = float(df["word_count"].mean())
    
    print(f"Word Count Statistics:")
    print(f"  - Minimum Word Count: {min_words}")
    print(f"  - Average Word Count: {avg_words:.2f}")
    print(f"  - Maximum Word Count: {max_words}")
    
    target_categories = [
        "Backend Developer", "Business Analyst", "Cloud Architect",
        "Data Analyst", "Data Engineer", "Data Scientist",
        "DevOps Engineer", "Frontend Developer", "Java Developer",
        "Mobile Developer", "Python Developer", "QA Engineer",
        "Security Analyst", "Web Developer"
    ]
    
    # Per-category counts
    print("\nRows per Category:")
    print("-" * 50)
    counts = df["category"].value_counts()
    
    low_sample_categories = []
    for cat in target_categories:
        count = int(counts.get(cat, 0))
        status = "OK"
        if count < 40:
            status = "LOW"
            low_sample_categories.append((cat, count))
        print(f"  {cat:<25}: {count:<5} | Status: {status}")
        
    print("-" * 50)
    
    # Audit PASS/FAIL status
    if len(low_sample_categories) == 0:
        print("\nAUDIT RESULT: PASS")
        print("All 14 categories have at least 40 samples.")
    else:
        print("\nAUDIT RESULT: FAIL")
        print("The following categories have fewer than 40 samples:")
        for cat, count in low_sample_categories:
            print(f"  - {cat}: {count} samples")
            
    print("=" * 50 + "\n")

if __name__ == "__main__":
    audit_dataset()
