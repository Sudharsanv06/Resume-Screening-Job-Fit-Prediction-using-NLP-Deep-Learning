"""
career_dna.py
=============
Provides career analytics, skill matching, and path mapping logic.
Calculates cluster scores, detects present and missing skills from resumes,
and profiles alternative career paths.
"""

from typing import Dict, List, Any

ROLE_CLUSTERS: Dict[str, List[str]] = {
    "Data Science":      ["Data Scientist", "Data Analyst", "Data Engineer"],
    "Backend & Python":  ["Backend Developer", "Python Developer", "Java Developer"],
    "DevOps & Cloud":   ["DevOps Engineer", "Cloud Architect"],
    "Frontend & Mobile": ["Frontend Developer", "Mobile Developer", "Web Developer"],
    "Security":         ["Security Analyst"],
    "Business":         ["Business Analyst", "QA Engineer"]
}

SKILL_KEYWORDS: Dict[str, List[str]] = {
    "Data Scientist":    ["python", "pytorch", "tensorflow", "sklearn", "pandas", "numpy",
                          "jupyter", "matplotlib", "seaborn", "model", "training",
                          "classification", "regression", "neural", "mlflow"],
    "Data Analyst":      ["sql", "excel", "tableau", "powerbi", "pandas", "numpy",
                          "visualization", "dashboard", "report", "pivot", "statistics",
                          "analysis", "query", "database", "insights"],
    "Data Engineer":     ["spark", "hadoop", "kafka", "airflow", "etl", "pipeline",
                          "sql", "aws", "azure", "databricks", "bigquery", "scala",
                          "dbt", "warehouse", "ingestion"],
    "Backend Developer": ["api", "rest", "microservices", "docker", "sql", "nodejs",
                          "django", "flask", "fastapi", "postgresql", "redis",
                          "authentication", "server", "endpoint", "mvc"],
    "Python Developer":  ["python", "django", "flask", "fastapi", "pip", "venv",
                          "pytest", "sqlalchemy", "celery", "redis", "asyncio",
                          "pydantic", "requests", "boto3", "scripting"],
    "Java Developer":    ["java", "spring", "springboot", "maven", "gradle", "junit",
                          "hibernate", "microservices", "jvm", "kafka", "tomcat",
                          "rest", "api", "docker", "kubernetes"],
    "DevOps Engineer":   ["docker", "kubernetes", "ci/cd", "jenkins", "terraform",
                          "ansible", "aws", "azure", "gcp", "linux", "bash",
                          "monitoring", "prometheus", "grafana", "pipeline"],
    "Cloud Architect":   ["aws", "azure", "gcp", "terraform", "kubernetes", "serverless",
                          "lambda", "s3", "vpc", "iam", "cloudformation",
                          "microservices", "devops", "security", "cost"],
    "Frontend Developer": ["react", "vue", "angular", "javascript", "typescript", "css",
                          "html", "webpack", "tailwind", "redux", "rest",
                          "responsive", "api", "testing", "accessibility"],
    "Mobile Developer":  ["android", "ios", "swift", "kotlin", "flutter", "react native",
                          "firebase", "xcode", "gradle", "api", "sqlite",
                          "push notifications", "ui", "ux", "deployment"],
    "Web Developer":     ["html", "css", "javascript", "react", "php", "wordpress",
                          "mysql", "bootstrap", "jquery", "rest", "api",
                          "responsive", "deployment", "git", "seo"],
    "QA Engineer":       ["testing", "selenium", "pytest", "junit", "automation",
                          "manual", "testcases", "regression", "performance",
                          "jira", "bug", "api testing", "postman", "cypress", "ci/cd"],
    "Security Analyst":  ["cybersecurity", "penetration testing", "siem", "firewall",
                          "vulnerability", "encryption", "network", "compliance",
                          "incident", "forensics", "nmap", "metasploit", "python", "risk", "audit"],
    "Business Analyst":  ["requirements", "stakeholder", "agile", "scrum", "process",
                          "analysis", "documentation", "jira", "sql", "excel",
                          "workflow", "uat", "business case", "reporting", "wireframe"]
}

def get_cluster_scores(all_probs: Dict[str, float]) -> Dict[str, float]:
    """
    Computes average prediction scores for 6 predefined role clusters.
    
    Args:
        all_probs (Dict[str, float]): Category probabilities mapping.
        
    Returns:
        Dict[str, float]: Score (0.0 to 100.0) for each cluster.
    """
    cluster_scores = {}
    for cluster, roles in ROLE_CLUSTERS.items():
        total = sum(all_probs.get(role, 0.0) for role in roles)
        avg = total / len(roles)
        cluster_scores[cluster] = float(round(avg * 100, 2))
    return cluster_scores

def get_skill_gaps(resume_text: str, predicted_role: str) -> Dict[str, Any]:
    """
    Identifies present and missing skills based on keyword presence in the resume text.
    
    Args:
        resume_text (str): Lowercase or raw text of the resume.
        predicted_role (str): The predicted classification label.
        
    Returns:
        Dict[str, Any]: Contains role name, present skills, missing skills, and fit percentage.
    """
    resume_text_lower = resume_text.lower()
    keywords = SKILL_KEYWORDS.get(predicted_role, [])
    
    present = []
    missing = []
    
    for kw in keywords:
        # Simple substring check is highly robust for standard resume matches
        if kw in resume_text_lower:
            present.append(kw)
        else:
            missing.append(kw)
            
    denominator = len(keywords) if len(keywords) > 0 else 15
    fit_pct = int(round((len(present) / denominator) * 100))
    
    return {
        "role": predicted_role,
        "present": present,
        "missing": missing,
        "fit_pct": fit_pct
    }

def get_alternative_paths(all_probs: Dict[str, float], predicted_role: str, resume_text: str = "") -> List[Dict[str, Any]]:
    """
    Identifies top 3 alternative paths from other categories, with skill gap counts.
    
    Args:
        all_probs (Dict[str, float]): Category probabilities mapping.
        predicted_role (str): The predicted primary role.
        resume_text (str): Optional raw resume text to calculate accurate gaps.
        
    Returns:
        List[Dict[str, Any]]: Sorted list of alternative paths.
    """
    # Sort roles by probability score descending
    sorted_probs = sorted(all_probs.items(), key=lambda item: item[1], reverse=True)
    
    alternative_paths = []
    for role, score in sorted_probs:
        if role == predicted_role:
            continue
        if len(alternative_paths) >= 3:
            break
            
        # Compute gap count: total keywords - keywords found in resume
        keywords = SKILL_KEYWORDS.get(role, [])
        if resume_text:
            resume_text_lower = resume_text.lower()
            present_count = sum(1 for kw in keywords if kw in resume_text_lower)
            gap_count = len(keywords) - present_count
        else:
            gap_count = 15 # default if no text provided
            
        alternative_paths.append({
            "role": role,
            "score": float(score),
            "gap_count": int(gap_count)
        })
        
    return alternative_paths

def get_resume_dna(resume_text: str, all_probs: Dict[str, float], predicted_role: str) -> Dict[str, Any]:
    """
    Assembles the complete Resume DNA payload.
    
    Args:
        resume_text (str): Raw resume text.
        all_probs (Dict[str, float]): Prediction probability mapping.
        predicted_role (str): Primary predicted class.
        
    Returns:
        Dict[str, Any]: Complete career radar, skill gap, and alt path scores.
    """
    return {
        "cluster_scores": get_cluster_scores(all_probs),
        "skill_gap": get_skill_gaps(resume_text, predicted_role),
        "alternative_paths": get_alternative_paths(all_probs, predicted_role, resume_text)
    }
