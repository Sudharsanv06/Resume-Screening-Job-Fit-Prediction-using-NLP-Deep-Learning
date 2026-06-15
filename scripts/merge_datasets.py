"""
merge_datasets.py
=================
Combines, cleans, maps, and deduplicates real-world and synthetic resume datasets.
Generates dataset/resumes_final.csv with a balanced, clean set of 3,000+ real resumes
mapped to 14 standard IT and business categories.
"""

import re
import warnings
from pathlib import Path
import pandas as pd

# Define paths dynamically relative to the file location
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

def clean_text(text: str) -> str:
    """
    Cleans raw resume text by stripping HTML tags and normalizing whitespace.
    
    Args:
        text (str): The raw resume text.
        
    Returns:
        str: The cleaned and normalized text.
    """
    if not isinstance(text, str):
        return ""
    # 1. Strip HTML tags using regex
    text = re.sub(r'<[^>]+>', ' ', text)
    # 2. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def merge_and_clean_pipelines() -> None:
    """
    Executes the full merge, cleaning, balancing, and synthetic backup pipeline.
    """
    # 1. Paths
    dataset1_path = DATASET_DIR / "UpdatedResumeDataSet.csv"
    dataset2_path = DATASET_DIR / "Resume.csv"
    synthetic_path = DATASET_DIR / "resumes_v2.csv"
    output_path = DATASET_DIR / "resumes_final.csv"
    
    # Category mapping definition (supporting both Title Case and Uppercase for Resume.csv)
    category_mapping = {
        "Java Developer": "Java Developer",
        "Python Developer": "Python Developer",
        "DevOps Engineer": "DevOps Engineer",
        "Testing": "QA Engineer",
        "Automation Testing": "QA Engineer",
        "Web Designing": "Web Developer",
        "Data Science": "Data Scientist",
        "Network Security Engineer": "Security Analyst",
        "DotNet Developer": "Backend Developer",
        "Blockchain": "Backend Developer",
        "Database": "Data Engineer",
        "Hadoop": "Data Engineer",
        "ETL Developer": "Data Engineer",
        "Business Analyst": "Business Analyst",
        "Information-Technology": "Backend Developer",
        "Business-Development": "Business Analyst",
        "Finance": "Business Analyst",
        "HR": "Business Analyst",
        "SAP Developer": "Backend Developer",
        "Cloud": "Cloud Architect",
        
        # Uppercase mappings for Resume.csv
        "JAVA DEVELOPER": "Java Developer",
        "PYTHON DEVELOPER": "Python Developer",
        "DEVOPS ENGINEER": "DevOps Engineer",
        "TESTING": "QA Engineer",
        "AUTOMATION TESTING": "QA Engineer",
        "WEB DESIGNING": "Web Developer",
        "DATA SCIENCE": "Data Scientist",
        "NETWORK SECURITY ENGINEER": "Security Analyst",
        "DOTNET DEVELOPER": "Backend Developer",
        "BLOCKCHAIN": "Backend Developer",
        "DATABASE": "Data Engineer",
        "HADOOP": "Data Engineer",
        "ETL DEVELOPER": "Data Engineer",
        "BUSINESS ANALYST": "Business Analyst",
        "INFORMATION-TECHNOLOGY": "Backend Developer",
        "BUSINESS-DEVELOPMENT": "Business Analyst",
        "FINANCE": "Business Analyst",
        "SAP DEVELOPER": "Backend Developer",
        "CLOUD": "Cloud Architect"
    }
    
    target_categories = [
        "Backend Developer", "Business Analyst", "Cloud Architect",
        "Data Analyst", "Data Engineer", "Data Scientist",
        "DevOps Engineer", "Frontend Developer", "Java Developer",
        "Mobile Developer", "Python Developer", "QA Engineer",
        "Security Analyst", "Web Developer"
    ]
    
    dataframes = []
    
    # 2. Load Dataset 1
    print(f"Loading primary dataset from: {dataset1_path}")
    if not dataset1_path.exists():
        raise FileNotFoundError(f"Primary dataset not found at {dataset1_path}.")
    
    df1 = pd.read_csv(dataset1_path)
    print(f"Dataset 1 loaded: {len(df1)} rows.")
    
    # Apply mapping
    df1["mapped_category"] = df1["Category"].map(category_mapping)
    df1 = df1.dropna(subset=["mapped_category"])
    
    # Keep only text and mapped category, rename
    df1 = df1.rename(columns={"Resume": "resume_text", "mapped_category": "category"})
    df1 = df1[["resume_text", "category"]]
    dataframes.append(df1)
    
    # 3. Load Dataset 2 (optional check)
    print(f"Checking for secondary dataset at: {dataset2_path}")
    if dataset2_path.exists():
        df2 = pd.read_csv(dataset2_path)
        print(f"Dataset 2 loaded: {len(df2)} rows.")
        
        # Apply mapping
        df2["mapped_category"] = df2["Category"].map(category_mapping)
        df2 = df2.dropna(subset=["mapped_category"])
        
        # Keep only text and mapped category, rename
        df2 = df2.rename(columns={"Resume_str": "resume_text", "mapped_category": "category"})
        df2 = df2[["resume_text", "category"]]
        dataframes.append(df2)
    else:
        warnings.warn(f"Secondary dataset not found at {dataset2_path}. Continuing with Dataset 1 only.")
    
    # 4. Concatenate and clean
    df_merged = pd.concat(dataframes, ignore_index=True)
    print(f"Total merged raw rows: {len(df_merged)}")
    
    # Apply text cleaning
    print("Cleaning resume texts (HTML stripping and whitespace normalization)...")
    df_merged["resume_text"] = df_merged["resume_text"].apply(clean_text)
    
    # 5. Drop rows with fewer than 50 words
    print("Filtering short resumes (< 50 words)...")
    df_merged["word_count"] = df_merged["resume_text"].apply(lambda x: len(str(x).split()))
    df_merged = df_merged[df_merged["word_count"] >= 50]
    df_merged = df_merged.drop(columns=["word_count"])
    print(f"Rows after word count filtering: {len(df_merged)}")
    
    # 6. Drop exact duplicate resume_text rows
    print("Removing exact duplicate resumes...")
    df_merged = df_merged.drop_duplicates(subset=["resume_text"])
    print(f"Rows after deduplication: {len(df_merged)}")
    
    # 7. For each category, keep max 300 rows (cap dominant categories)
    print("Capping categories to a maximum of 300 rows...")
    df_merged = df_merged.groupby("category", as_index=False).apply(
        lambda x: x.head(300)
    ).reset_index(drop=True)
    
    # Make sure columns are clean
    df_merged = df_merged[["resume_text", "category"]]
    print(f"Rows after capping categories: {len(df_merged)}")
    
    # 8. Load synthetic dataset to back up low-sample categories
    print(f"Loading synthetic backup dataset from: {synthetic_path}")
    if not synthetic_path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found at {synthetic_path}.")
    
    df_synth = pd.read_csv(synthetic_path)
    df_synth["resume_text"] = df_synth["resume_text"].apply(clean_text)
    
    # Check counts and fill
    final_rows = []
    print("\nAuditing and filling category counts to minimum 40 samples:")
    print("-" * 60)
    
    for category in target_categories:
        cat_df = df_merged[df_merged["category"] == category]
        count = len(cat_df)
        
        if count < 40:
            deficit = 40 - count
            print(f"Category '{category}': {count} samples. Adding {deficit} synthetic samples.")
            # Get synthetic samples for this category
            synth_cat_df = df_synth[df_synth["category"] == category]
            # Take only the deficit amount
            synth_fill = synth_cat_df.head(deficit)[["resume_text", "category"]]
            # Concat
            cat_df = pd.concat([cat_df, synth_fill], ignore_index=True)
        else:
            print(f"Category '{category}': {count} samples. (Satisfied)")
            
        final_rows.append(cat_df)
        
    df_final = pd.concat(final_rows, ignore_index=True)
    
    # Double check formatting and drop any remaining duplicates just in case
    df_final = df_final.drop_duplicates(subset=["resume_text"])
    
    # 9. Save final dataset
    print(f"\nSaving final merged and cleaned dataset to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, index=False, encoding="utf-8")
    
    # 10. Print final stats
    print("\nFinal Category Counts:")
    print("=" * 40)
    final_counts = df_final["category"].value_counts()
    for cat in target_categories:
        print(f"{cat:<25}: {final_counts.get(cat, 0)}")
    print("=" * 40)
    print(f"Total Rows: {len(df_final)}")
    print("Dataset merge complete!\n")

if __name__ == "__main__":
    merge_and_clean_pipelines()
