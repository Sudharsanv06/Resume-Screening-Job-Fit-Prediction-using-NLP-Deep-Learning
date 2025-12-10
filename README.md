# Resume Screening & Job Fit Prediction (NLP + Deep Learning)

## 🎯 Project Overview

This project implements a **Deep Learning NLP model** to automatically screen resumes and predict job fit by classifying them into relevant job categories. The system uses LSTM/Bi-LSTM neural networks to analyze resume text and match candidates to appropriate job roles.

## 🔍 Problem Statement

Manual resume screening is time-consuming and prone to human bias. This project automates the resume screening process using Natural Language Processing and Deep Learning to:

- Extract meaningful features from resume text
- Classify resumes into predefined job categories
- Predict the best job fit for candidates
- Speed up the recruitment process with AI-powered automation

## 🛠️ Technologies Used

### Programming Language
- **Python** 3.8+

### Deep Learning Framework
- **TensorFlow** / **Keras**

### NLP Tools
- Tokenizer (Text preprocessing)
- Padding (Sequence normalization)
- Word Embeddings (Feature representation)

### Model Architecture
- **LSTM** (Long Short-Term Memory)
- **Bi-LSTM** (Bidirectional LSTM)

### Development Environment
- **IDE**: VS Code
- **Version Control**: Git + GitHub

### Data Analysis & Visualization
- pandas
- numpy
- matplotlib
- seaborn

## 📊 Dataset Information

### Dataset Structure
- **File**: `dataset/resumes.csv`
- **Columns**:
  - `Resume`: Resume text (job descriptions, skills, experience)
  - `Category`: Job role classification (target variable)

### Job Categories
The dataset includes resumes for various job roles such as:
- Data Scientist
- Java Developer
- Web Developer
- DevOps Engineer
- Mobile Developer
- Backend Developer
- Frontend Developer
- Data Analyst
- Python Developer
- Cloud Architect
- QA Engineer
- Data Engineer
- Security Analyst
- Business Analyst
- And more...

### Dataset Statistics
- **Total Resumes**: 30+ samples (expandable)
- **Job Categories**: 13+ unique roles
- **Data Format**: CSV (Comma-Separated Values)

## 📁 Project Structure

```
project-root/
│
├── dataset/
│   └── resumes.csv              # Resume dataset
│
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory Data Analysis
│
├── src/
│   └── preprocess.py            # Data preprocessing utilities
│
├── models/
│   └── (trained models will be saved here)
│
├── results/
│   └── (training results, metrics, plots)
│
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore file
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

### Clone the Repository
```bash
git clone https://github.com/Sudharsanv06/Resume-Screening-Job-Fit-Prediction-using-NLP-Deep-Learning.git
cd Resume-Screening-Job-Fit-Prediction-using-NLP-Deep-Learning
```

## 📅 Development Timeline

### ✅ Day 1: Project Setup + Dataset + EDA
- [x] Project structure created
- [x] Dataset added (`resumes.csv`)
- [x] Initial exploratory data analysis
- [x] README documentation

### 🔜 Day 2: Data Preprocessing + Feature Engineering
- Text cleaning and normalization
- Tokenization and padding
- Label encoding
- Train-test split

### 🔜 Day 3: Model Building + Training
- LSTM/Bi-LSTM model architecture
- Model compilation and training
- Hyperparameter tuning
- Model evaluation

### 🔜 Day 4: Testing + Deployment + Documentation
- Model testing on new resumes
- Performance metrics
- Viva preparation
- Final documentation

## 🎓 Project Goals

By the end of this project, you will have:

✅ A trained Deep Learning NLP model (LSTM/Bi-LSTM)  
✅ Resume classification system for multiple job roles  
✅ Job-fit prediction functionality  
✅ Complete documentation (viva-ready + resume-ready)  
✅ Clean Git commit history  

## 👨‍💻 Author

**Sudharsan V**

GitHub: [@Sudharsanv06](https://github.com/Sudharsanv06)

## 📝 License

This project is open source and available for educational purposes.

---

**Status**: 🟢 Day 1 Complete - Ready for Day 2!
