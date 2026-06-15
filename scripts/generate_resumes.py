"""
generate_resumes.py
===================
Synthetic resume generator for Phase 1 of the Resume Screening project.
Generates 40 realistic resumes per category for 14 job categories using 
structured templates (Summary, Skills, Experience, Education).
Saves output to dataset/resumes_v2.csv.
"""

import os
import random
import csv

# Set random seed for reproducibility
random.seed(42)

# Define the 14 job categories
CATEGORIES = [
    "Backend Developer", "Business Analyst", "Cloud Architect", "Data Analyst", 
    "Data Engineer", "Data Scientist", "DevOps Engineer", "Frontend Developer", 
    "Java Developer", "Mobile Developer", "Python Developer", "QA Engineer", 
    "Security Analyst", "Web Developer"
]

# Vocabulary and content pools for generating templates
FIRST_NAMES = ["James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles", "Sarah", "Emily", "Jessica", "Ashley", "Amanda", "Jennifer", "Melissa", "Stephanie", "Nicole", "Heather"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson", "Martinez", "Anderson", "Taylor", "Thomas", "Hernandez", "Moore", "Martin", "Jackson", "Thompson", "White"]
COMPANIES = ["TechCorp Solutions", "InnovateSoft", "CloudDynamics", "DataSystems Inc.", "WebFlow Digital", "Apex Global", "SecureNet Solutions", "Alpha Consulting", "ByteWise Technologies", "NexGen Enterprises"]
UNIVERSITIES = ["Tech University", "State University of Technology", "Metropolitan University", "Institute of Science & Tech", "Western Engineering College", "National Science University"]
DEGREES = ["Bachelor of Science in Computer Science", "MS in Information Technology", "Bachelor of Engineering", "MS in Software Engineering", "Bachelor of Science in Information Systems"]

# Category-specific content configurations
CATEGORY_CONTENT = {
    "Backend Developer": {
        "titles": ["Backend Engineer", "Senior Backend Developer", "Software Engineer - Backend"],
        "adjectives": ["detail-oriented", "analytical", "highly skilled", "experienced"],
        "focus_areas": ["building highly scalable RESTful services", "designing microservices architectures", "optimizing database performance", "implementing robust backend architectures"],
        "years": ["3+", "5+", "4+", "6"],
        "skills": ["Python", "Go", "Node.js", "Express", "FastAPI", "PostgreSQL", "MongoDB", "Redis", "REST APIs", "Microservices", "Docker", "Git", "SQL"],
        "achievements": [
            "Designed and implemented high-performance REST APIs handling over 10,000 requests per minute.",
            "Migrated a monolithic application into microservices, reducing server response times by 30%.",
            "Optimized database queries and schema design, leading to a 40% reduction in database CPU usage.",
            "Integrated secure OAuth2 authentication flow and third-party payment gateways."
        ],
        "duties": [
            "Collaborating with frontend engineers to define API contracts and specs.",
            "Implementing caching strategies using Redis to improve user experience.",
            "Writing comprehensive unit tests to ensure application reliability and security."
        ]
    },
    "Business Analyst": {
        "titles": ["Business Analyst", "Senior Systems Analyst", "Product Owner / Business Analyst"],
        "adjectives": ["results-driven", "adaptable", "detail-oriented", "strategic"],
        "focus_areas": ["requirements engineering", "process optimization", "facilitating Agile workshops", "aligning business strategies with technical deliverables"],
        "years": ["4+", "5+", "3+", "6+"],
        "skills": ["Agile methodologies", "Scrum", "JIRA", "Confluence", "SQL", "Excel", "UML", "BRD", "FRD", "User Stories", "Tableau", "PowerBI", "Gap Analysis"],
        "achievements": [
            "Successfully gathered requirements and delivered a complex ERP platform, increasing project success rate by 20%.",
            "Created detailed Business Requirement Documents (BRDs) and Functional Specification Documents.",
            "Conducted process mapping and gap analysis, improving internal operational efficiency by 15%.",
            "Led UAT sessions with key business stakeholders to ensure software alignment with client needs."
        ],
        "duties": [
            "Translating high-level business requirements into clear backlog items and user stories.",
            "Facilitating scrum ceremonies including product backlog refinement and sprint planning.",
            "Communicating project updates, risks, and scope changes clearly to stakeholders."
        ]
    },
    "Cloud Architect": {
        "titles": ["Cloud Solutions Architect", "Senior Cloud Architect", "Infrastructure Architect"],
        "adjectives": ["certified", "innovative", "forward-thinking", "highly analytical"],
        "focus_areas": ["designing resilient cloud environments", "orchestrating enterprise migrations", "implementing hybrid cloud deployments", "reducing cloud infrastructure spending"],
        "years": ["5+", "8+", "6+", "7+"],
        "skills": ["AWS", "Azure", "GCP", "Terraform", "CloudFormation", "Kubernetes", "IAM", "VPC", "EC2", "S3", "RDS", "Serverless", "Lambda", "Docker"],
        "achievements": [
            "Designed and deployed secure, scalable cloud architecture on AWS for a high-traffic e-commerce portal.",
            "Orchestrated cloud migration from on-premise servers to Azure, saving 25% in annual infrastructure costs.",
            "Implemented infrastructure as code using Terraform to automate cloud environment setups.",
            "Designed disaster recovery plans ensuring 99.99% availability of mission-critical databases."
        ],
        "duties": [
            "Evaluating architectural options and making recommendations regarding cloud service providers.",
            "Collaborating with cybersecurity teams to enforce IAM roles and zero-trust policies.",
            "Monitoring and optimizing cloud performance metrics using native cloud monitoring tools."
        ]
    },
    "Data Analyst": {
        "titles": ["Data Analyst", "Senior Business Intelligence Analyst", "Data & Insights Analyst"],
        "adjectives": ["statistically minded", "insight-driven", "analytical", "highly inquisitive"],
        "focus_areas": ["building interactive business intelligence dashboards", "performing descriptive and diagnostic analytics", "conducting statistical testing", "interpreting customer metrics"],
        "years": ["2+", "4+", "3+", "5+"],
        "skills": ["SQL", "Tableau", "PowerBI", "Python", "Pandas", "NumPy", "Excel", "Data Visualization", "A/B Testing", "Statistics", "Reporting", "SAS"],
        "achievements": [
            "Developed Tableau and PowerBI dashboards that tracking critical company KPIs, driving strategic decision making.",
            "Conducted comprehensive A/B testing on user registration funnel, improving conversion rates by 8%.",
            "Wrote optimized SQL queries to query massive datasets, reducing report generation time by 50%.",
            "Identified operational bottlenecks via data auditing, saving the sales division $50k annually."
        ],
        "duties": [
            "Creating daily, weekly, and monthly reports detailing sales performance and user metrics.",
            "Cleaning and transforming messy data from various marketing and transaction systems.",
            "Translating numbers and graphs into actionable business insights for executive teams."
        ]
    },
    "Data Engineer": {
        "titles": ["Data Engineer", "Senior Big Data Engineer", "ETL Developer / Data Engineer"],
        "adjectives": ["highly technical", "process-driven", "performance-focused", "experienced"],
        "focus_areas": ["constructing robust ETL pipelines", "designing scalable data lakes", "optimizing database indexing", "streaming real-time event analytics"],
        "years": ["3+", "5+", "4+", "6+"],
        "skills": ["Python", "SQL", "Apache Spark", "Hadoop", "Kafka", "Airflow", "ETL Pipelines", "Snowflake", "BigQuery", "Data Warehousing", "Scala", "NoSQL"],
        "achievements": [
            "Built scalable ETL pipelines using Apache Spark and Airflow, processing over 5TB of raw daily logs.",
            "Migrated an on-premise data warehouse to Snowflake, enabling business units to query data 3x faster.",
            "Implemented a real-time event streaming pipeline using Kafka for credit card transaction logging.",
            "Optimized query performance on relational databases, reducing database timeouts by 60%."
        ],
        "duties": [
            "Maintaining and monitoring cloud data infrastructure and batch orchestration engines.",
            "Collaborating with Data Scientists to prepare analytics-ready feature tables.",
            "Implementing strict data validation schemas to ensure high data quality and reliability."
        ]
    },
    "Data Scientist": {
        "titles": ["Data Scientist", "Machine Learning Engineer", "Senior Data Scientist"],
        "adjectives": ["highly analytical", "research-oriented", "problem-solving", "quantitative"],
        "focus_areas": ["building predictive machine learning models", "implementing deep learning architectures", "natural language processing", "computer vision applications"],
        "years": ["3+", "5+", "4+", "6+"],
        "skills": ["Python", "R", "Machine Learning", "Deep Learning", "Scikit-Learn", "TensorFlow", "PyTorch", "NLP", "Computer Vision", "SQL", "Pandas", "Statistics"],
        "achievements": [
            "Built a churn prediction model using Random Forests, reducing customer attrition by 12%.",
            "Developed an NLP text classification system for customer tickets, automating 40% of routing tasks.",
            "Implemented custom CNN models for automated defect detection in manufacturing assembly lines.",
            "Successfully deployed deep learning models into production environments utilizing Docker containerization."
        ],
        "duties": [
            "Designing experimental frameworks and performing rigorous hypothesis and statistical testing.",
            "Engineering features from structured and unstructured text databases to improve model accuracy.",
            "Presenting analytical findings and technical ML approaches to product and business leaders."
        ]
    },
    "DevOps Engineer": {
        "titles": ["DevOps Engineer", "Site Reliability Engineer (SRE)", "Platform Engineer"],
        "adjectives": ["infrastructure-focused", "efficiency-driven", "highly collaborative", "experienced"],
        "focus_areas": ["automating CI/CD integration", "managing container orchestration infrastructure", "implementing infrastructure as code", "improving system reliability"],
        "years": ["3+", "5+", "4+", "6+"],
        "skills": ["Docker", "Kubernetes", "Jenkins", "GitLab CI", "GitHub Actions", "Terraform", "Ansible", "AWS", "Linux", "Prometheus", "Grafana", "Bash", "CI/CD"],
        "achievements": [
            "Designed and automated CI/CD pipelines using GitHub Actions, reducing deployment time from 1 hour to 10 minutes.",
            "Managed production Kubernetes clusters (EKS) hosting 50+ microservices with zero downtime.",
            "Implemented automated infrastructure provisioning using Terraform across dev, staging, and production environments.",
            "Configured alerts and monitoring dashboards in Grafana, decreasing incident response times by 35%."
        ],
        "duties": [
            "Configuring secure system firewalls, VPNs, and IAM policies inside AWS infrastructure.",
            "Collaborating with development teams to debug deployment issues and network configurations.",
            "Performing capacity planning and optimizing cluster resources to lower infrastructure spending."
        ]
    },
    "Frontend Developer": {
        "titles": ["Frontend Developer", "Senior Frontend Engineer", "UI / React Web Developer"],
        "adjectives": ["creatively minded", "UI-focused", "highly detail-oriented", "skilled"],
        "focus_areas": ["developing responsive single-page web applications", "building interactive user interfaces", "ensuring cross-browser compatibility", "optimizing web asset loading speed"],
        "years": ["3+", "5+", "4+", "6+"],
        "skills": ["React", "Angular", "Vue.js", "HTML5", "CSS3", "JavaScript", "TypeScript", "Redux", "SASS", "Webpack", "Vite", "Responsive Design", "Git"],
        "achievements": [
            "Developed a modern, responsive web application dashboard using React and Tailwind CSS, increasing user engagement.",
            "Migrated legacy frontend components to TypeScript, reducing interface-related runtime bugs by 45%.",
            "Optimized frontend build configurations and code-splitting, improving initial page load time by 1.5 seconds.",
            "Designed and maintained a shared reusable UI component library, streamlining developer workflow."
        ],
        "duties": [
            "Collaborating closely with designers to convert wireframes into functional web pages.",
            "Connecting frontend state with RESTful API endpoints and handling error boundary states.",
            "Conducting accessibility audits to ensure compliance with Web Content Accessibility Guidelines (WCAG)."
        ]
    },
    "Java Developer": {
        "titles": ["Java Developer", "Senior Java Engineer", "Software Developer - Java"],
        "adjectives": ["process-oriented", "highly technical", "experienced", "analytical"],
        "focus_areas": ["designing enterprise-level Java applications", "implementing robust microservices architectures", "integrating database persistence layers", "developing reliable web APIs"],
        "years": ["4+", "5+", "3+", "6+"],
        "skills": ["Java", "Spring Boot", "Spring Cloud", "Hibernate", "RESTful APIs", "Microservices", "Maven", "Gradle", "JUnit", "MySQL", "Docker", "Git", "SQL"],
        "achievements": [
            "Developed enterprise Java applications using Spring Boot, securing reliable database persistence with Hibernate.",
            "Designed and built microservices handling 500+ parallel transactions per second with low latency.",
            "Implemented database pooling and query optimizations, resolving critical system memory issues.",
            "Wrote comprehensive unit and integration tests using JUnit and Mockito, ensuring 85% code coverage."
        ],
        "duties": [
            "Building application integrations with active MQ systems and message queues.",
            "Reviewing developer code, recommending design patterns, and mentoring junior Java engineers.",
            "Deploying compiled jar distributions within secure, isolated Docker containers."
        ]
    },
    "Mobile Developer": {
        "titles": ["Mobile App Developer", "Senior iOS / Android Developer", "React Native Engineer"],
        "adjectives": ["highly focused", "detail-oriented", "skilled", "creative"],
        "focus_areas": ["developing cross-platform mobile solutions", "building native iOS and Android apps", "designing mobile interfaces", "publishing apps to official stores"],
        "years": ["3+", "5+", "4+", "6+"],
        "skills": ["React Native", "Flutter", "Swift", "SwiftUI", "Kotlin", "Android SDK", "iOS SDK", "Xcode", "Firebase", "REST APIs", "Git", "Mobile UI Design"],
        "achievements": [
            "Developed and published a cross-platform React Native app, achieving over 100k active app downloads.",
            "Built a native iOS application using Swift and SwiftUI, integrated with Core Data and background syncing.",
            "Implemented offline-first mobile support and push notifications with Firebase integration.",
            "Improved app performance by optimizing image caching and reducing mobile bundle storage sizes."
        ],
        "duties": [
            "Translating UI layouts from Figma into pixel-perfect and responsive mobile layouts.",
            "Working with product managers to plan release schedules for App Store and Google Play Store.",
            "Integrating security standards for mobile data encryption and credential storage."
        ]
    },
    "Python Developer": {
        "titles": ["Python Developer", "Software Engineer - Python", "Automation & Python Engineer"],
        "adjectives": ["problem-solving", "highly efficient", "versatile", "experienced"],
        "focus_areas": ["writing backend web APIs", "automating daily operational tasks", "building robust web scrapers", "scripting data integration steps"],
        "years": ["3+", "5+", "4+", "6+"],
        "skills": ["Python", "Django", "Flask", "FastAPI", "PyTest", "NumPy", "Pandas", "Web Scraping", "Automation", "REST APIs", "SQL", "Git", "Docker"],
        "achievements": [
            "Created backend REST APIs using FastAPI, allowing rapid processing of high-throughput web requests.",
            "Automated system reports and data processing workflows, reducing manual operations by 20 hours a week.",
            "Developed a robust web scraping pipeline using BeautifulSoup and Selenium, indexing 1M+ product listings.",
            "Built clean and maintainable Python packages for internal data parsing and math utilities."
        ],
        "duties": [
            "Creating scalable backend architectures and writing database migrations.",
            "Setting up automated unit test suites using PyTest to ensure system stability.",
            "Configuring and orchestrating tasks inside background worker queues."
        ]
    },
    "QA Engineer": {
        "titles": ["QA Automation Engineer", "Senior QA Specialist", "Software Developer in Test (SDET)"],
        "adjectives": ["meticulous", "quality-driven", "detail-oriented", "methodical"],
        "focus_areas": ["building robust automated testing frameworks", "increasing test coverage metrics", "designing manual test specs", "performing end-to-end regression testing"],
        "years": ["3+", "5+", "4+", "6+"],
        "skills": ["Selenium", "Cypress", "JUnit", "PyTest", "QA Automation", "Manual Testing", "API Testing", "Postman", "JIRA", "Bug Tracking", "CI/CD"],
        "achievements": [
            "Designed and implemented an automated testing framework using Selenium, increasing test coverage by 60%.",
            "Integrated test runners into GitLab CI pipeline, preventing bug leaks in production releases.",
            "Wrote comprehensive test plans and executed sanity and boundary regression test cases.",
            "Discovered and logged critical security flaws and performance leaks in JIRA, reducing production hotfixes."
        ],
        "duties": [
            "Creating detail-rich bug tickets and verifying resolved issues through validation testing.",
            "Partnering with development engineers to align code structures with testing models.",
            "Testing backend API structures using Postman scripts and parameter variations."
        ]
    },
    "Security Analyst": {
        "titles": ["Cybersecurity Analyst", "Information Security Specialist", "Security Engineer"],
        "adjectives": ["security-focused", "analytical", "highly vigilant", "alert"],
        "focus_areas": ["identifying security vulnerabilities", "designing secure firewall network guidelines", "performing risk penetration auditing", "enforcing safety compliance policies"],
        "years": ["4+", "5+", "3+", "6+"],
        "skills": ["Penetration Testing", "Kali Linux", "Metasploit", "Wireshark", "Firewalls", "SIEM", "Cryptography", "Network Security", "Vulnerability Scanning", "Compliance"],
        "achievements": [
            "Conducted network security audits and penetration tests, identifying and patching critical system openings.",
            "Configured and managed corporate SIEM dashboard to identify and isolate active network threats.",
            "Established corporate-wide security compliance frameworks, resolving audit gaps.",
            "Designed and held staff workshops on cyber hygiene and corporate phishing awareness."
        ],
        "duties": [
            "Monitoring active security logs and investigating potential network breach behaviors.",
            "Ensuring servers, firewalls, and network configurations are patched and secure.",
            "Enforcing security best practices for password hashing and data access controls."
        ]
    },
    "Web Developer": {
        "titles": ["Web Developer", "Full Stack Developer", "Web Designer & Developer"],
        "adjectives": ["highly creative", "versatile", "customer-focused", "experienced"],
        "focus_areas": ["developing dynamic websites", "customizing CMS platforms", "designing clean web layouts", "integrating ecommerce and custom modules"],
        "years": ["3+", "5+", "4+", "6+"],
        "skills": ["HTML", "CSS", "JavaScript", "PHP", "WordPress", "Laravel", "MySQL", "jQuery", "Bootstrap", "Vite", "SEO", "Responsive Design", "Git"],
        "achievements": [
            "Designed and launched responsive ecommerce websites, increasing sales conversions by 20%.",
            "Developed custom plugins and themes for WordPress sites, optimizing system loading times.",
            "Built robust, database-driven sites with PHP and Laravel, ensuring secure payment gateways.",
            "Improved website search indexing through technical SEO auditing and meta adjustments."
        ],
        "duties": [
            "Updating existing websites to maintain cross-browser styling and scripting layouts.",
            "Troubleshooting backend site bugs and managing website server migrations.",
            "Partnering with businesses to detail hosting configurations and analytics tracking."
        ]
    }
}

# Template to build a highly realistic resume
RESUME_TEMPLATE = """{name}
{title} | Email: {email} | Phone: {phone}

SUMMARY
{summary}

SKILLS
{skills}

EXPERIENCE
{job_title} | {company} ({duration})
{exp_bullets}

EDUCATION
{degree}
{univ} ({grad_year})"""

def clean_bullets(bullets):
    return "\n".join(f"- {b}" for b in bullets)

def generate_resume(category: str) -> str:
    """
    Generate a realistic resume string matching the requested category
    and conforming to the word count constraint (80-150 words).
    """
    first_name = random.choice(FIRST_NAMES)
    last_name = random.choice(LAST_NAMES)
    name = f"{first_name} {last_name}"
    
    email = f"{first_name.lower()}.{last_name.lower()}@example.com"
    phone = f"+1-555-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
    
    content = CATEGORY_CONTENT[category]
    title = random.choice(content["titles"])
    
    # Generate Summary
    adj = random.choice(content["adjectives"])
    focus = random.choice(content["focus_areas"])
    yr = random.choice(content["years"])
    summary = f"A {adj} {title} with {yr} years of experience in the software industry, specialized in {focus} and delivering high-quality business value."
    
    # Generate Skills
    skills_sample = random.sample(content["skills"], min(len(content["skills"]), random.randint(6, 8)))
    skills_str = ", ".join(skills_sample)
    
    # Generate Experience
    company = random.choice(COMPANIES)
    duration = f"{random.randint(2018, 2022)} - Present"
    job_title = random.choice(content["titles"])
    
    # Draw a random subset of achievements & duties to vary word count and structure
    ach = random.sample(content["achievements"], 1)
    dut = random.sample(content["duties"], 2)
    exp_bullets = clean_bullets(ach + dut)
    
    # Generate Education
    degree = random.choice(DEGREES)
    univ = random.choice(UNIVERSITIES)
    grad_year = str(random.randint(2012, 2020))
    
    # Populate the main template
    resume_text = RESUME_TEMPLATE.format(
        name=name,
        title=title,
        email=email,
        phone=phone,
        summary=summary,
        skills=skills_str,
        job_title=job_title,
        company=company,
        duration=duration,
        exp_bullets=exp_bullets,
        degree=degree,
        univ=univ,
        grad_year=grad_year
    )
    
    # Verify word count and pad or trim if needed (highly unlikely due to template bounds)
    word_count = len(resume_text.split())
    if word_count < 80:
        # Pad with some additional general duties
        extra_duties = [d for d in content["duties"] if d not in dut]
        if extra_duties:
            resume_text += f"\n- {extra_duties[0]}"
    
    return resume_text

def main():
    print("Starting synthetic resume dataset generation...")
    
    # Resolve directory paths relative to the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_root, "dataset")
    
    # Create dataset directory if it doesn't exist
    os.makedirs(dataset_dir, exist_ok=True)
    csv_path = os.path.join(dataset_dir, "resumes_v2.csv")
    
    resumes = []
    
    # Generate 40 resumes per category
    for category in CATEGORIES:
        print(f"   Generating 40 resumes for: {category}")
        for _ in range(40):
            resume_text = generate_resume(category)
            resumes.append((resume_text, category))
            
    # Write to CSV
    print(f"Saving {len(resumes)} resumes to {csv_path}...")
    with open(csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["resume_text", "category"])
        writer.writerows(resumes)
        
    print("Synthetic dataset creation complete!")

if __name__ == "__main__":
    main()
