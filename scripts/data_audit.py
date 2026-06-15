"""
data_audit.py
=============
Data audit script for Phase 1 of the Resume Screening project.
Loads dataset/resumes_v2.csv, validates counts and balances, calculates
text length metrics, and saves stats to results/preprocessing_info_v2.txt.
"""

import os
import pandas as pd

def audit_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    csv_path = os.path.join(project_root, "dataset", "resumes_v2.csv")
    results_dir = os.path.join(project_root, "results")
    stats_file_path = os.path.join(results_dir, "preprocessing_info_v2.txt")
    
    if not os.path.exists(csv_path):
        print(f"Error: Dataset file not found at {csv_path}. Please run generate_resumes.py first.")
        return
        
    print(f"Auditing dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    total_count = len(df)
    categories = df["category"].unique()
    num_classes = len(categories)
    
    # Calculate word counts
    df["word_count"] = df["resume_text"].apply(lambda x: len(str(x).split()))
    min_words = int(df["word_count"].min())
    max_words = int(df["word_count"].max())
    avg_words = float(df["word_count"].mean())
    
    # Calculate per-category stats
    category_counts = df["category"].value_counts()
    
    # Header format
    output_lines = []
    output_lines.append("DATA AUDIT & PREPROCESSING INFORMATION - V2")
    output_lines.append("===========================================")
    output_lines.append(f"Total Samples: {total_count}")
    output_lines.append(f"Number of Categories: {num_classes}")
    output_lines.append(f"Mean Word Count: {avg_words:.2f}")
    output_lines.append(f"Min Word Count: {min_words}")
    output_lines.append(f"Max Word Count: {max_words}")
    output_lines.append("\nCategory Distribution:")
    output_lines.append("----------------------")
    
    print("\n" + "="*50)
    print("DATASET AUDIT REPORT")
    print("="*50)
    print(f"Total Resumes: {total_count}")
    print(f"Unique Categories: {num_classes}")
    print(f"Word Count - Min: {min_words} | Max: {max_words} | Avg: {avg_words:.2f}")
    print("\nCategory Counts & Status:")
    print("-" * 50)
    
    flagged_categories = []
    for cat, count in category_counts.items():
        status = "OK"
        if count < 35:
            status = "LOW SAMPLE COUNT (< 35)"
            flagged_categories.append((cat, count))
        print(f"{cat:<25}: {count:<4} | {status}")
        output_lines.append(f"{cat:<25}: {count:<4} | {status}")
        
    if flagged_categories:
        print("\nWARNING: Some categories have fewer than 35 samples!")
        for cat, count in flagged_categories:
            print(f"   - {cat}: {count} samples")
        output_lines.append("\nWARNING: Low sample counts observed in some categories.")
    else:
        print("\nDataset verification passed: All categories have at least 35 samples.")
        output_lines.append("\nDataset verification passed: All categories are balanced.")
        
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Save statistics info
    print(f"\nSaving audit report to {stats_file_path}...")
    with open(stats_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print("Audit complete!")

if __name__ == "__main__":
    audit_dataset()
