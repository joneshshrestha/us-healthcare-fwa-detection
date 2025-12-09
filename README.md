# US Healthcare Fraud, Waste, and Abuse (FWA) Detection

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A machine learning solution for detecting fraudulent healthcare providers using Medicare-inspired claims data. This project applies advanced data science techniques to identify Fraud, Waste, and Abuse (FWA) patterns, achieving **95.46% ROC-AUC** and **82.18% recall** with a Random Forest classifier.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

Healthcare fraud costs the US healthcare system billions of dollars annually, with estimates ranging from 3-10% of total healthcare spending. This project tackles this critical problem by applying the **Knowledge Discovery in Databases (KDD) process** and advanced machine learning techniques to identify healthcare providers engaging in fraudulent activities.

### Business Problem

Healthcare Fraud, Waste, and Abuse (FWA) detection is challenging due to:
- Massive data volumes (millions of claims)
- Highly imbalanced datasets (fraud is rare)
- Evolving fraud patterns
- Need for interpretable models for regulatory compliance

### Solution Objectives

- Detect fraudulent healthcare providers from Medicare-inspired claims data
- Achieve ROC-AUC score of at least 80%
- Provide interpretable insights for audit prioritization
- Balance precision and recall for practical deployment

---

## Dataset

The synthetic Medicare-inspired dataset consists of four integrated data sources:

### Data Sources

1. **Beneficiary Data** (138,556 records, 25 features)
   - Patient demographics (age, gender, DOB, race, location)
   - 11 chronic conditions (Alzheimer's, heart failure, kidney disease, cancer, etc.)
   - Annual reimbursement and deductible amounts
   - Medicare Part A and Part B coverage information

2. **Inpatient Claims** (40,474 records, 30 features)
   - Claim dates, reimbursement and deductible amounts
   - Physician IDs (attending, operating, other)
   - Up to 10 diagnosis codes per claim
   - Up to 6 procedure codes per claim
   - Admission and discharge information

3. **Outpatient Claims** (517,737 records, 27 features)
   - Claim dates and financial amounts
   - Up to 10 diagnosis codes per claim
   - Up to 6 procedure codes per claim (mostly sparse)

4. **Provider Labels** (5,410 records, 2 features)
   - Provider ID
   - Fraud label (Yes/No)

### Dataset Characteristics

- **Total Scale:** 5,410 providers, 558,211 claims, 138,556 beneficiaries
- **Class Distribution:** Highly imbalanced (9.35% fraudulent, 90.65% non-fraudulent)
  - 506 fraudulent providers
  - 4,904 non-fraudulent providers
- **Data Quality:** 100% beneficiary-claim linkage, 100% provider-label match, no duplicate claims

---

## Methodology

### 1. Data Preprocessing

**Provider-Level Aggregation:** Claims and beneficiary data were aggregated to provider level, creating 47 engineered features:

- **Claim Volume Features (6):** Total claims, inpatient/outpatient counts, ratios, unique beneficiaries, claims per beneficiary
- **Financial Features (7):** Total/average reimbursement and deductible, standard deviations, ratios, max/min
- **Temporal Features (2):** Average claim duration, average admission duration
- **Medical Code Features (6):** Unique diagnosis/procedure codes, averages per claim, top code ratios
- **Physician Features (5):** Unique physician counts by type, diversity ratios
- **Demographic Features (21):** Aggregated beneficiary demographics, chronic condition statistics

**Feature Scaling:** StandardScaler applied to all features for model compatibility

**Train-Validation-Test Split:** Stratified 70-10-20 split maintaining 9.35% fraud rate across all sets

### 2. Exploratory Data Analysis

- Statistical analysis revealed right-skewed distributions with high-volume outliers
- Fraudulent providers showed substantially higher:
  - Total reimbursements (fraud mean: ~$584k vs. non-fraud: ~$53k)
  - Claim volumes and beneficiary counts
  - Diversity in diagnosis and procedure codes
- High feature correlations identified (e.g., TotalClaims with OutpatientClaims: 0.996)

### 3. Unsupervised Learning

**K-Means Clustering (k=3):**
- Identified distinct provider profiles
- High-volume cluster showed 60.97% fraud rate vs. 9.35% baseline
- PCA visualization revealed partial cluster separation

**Isolation Forest:**
- 10% contamination rate
- Detected anomalies had 32.35% fraud rate
- Captured 34.58% of fraud cases

### 4. Feature Selection

- Mutual Information scoring with percentile-based selection (tested 10-100%)
- Addressed multicollinearity from highly correlated features

### 5. Supervised Learning

**Models Evaluated:**

1. **Baseline: Logistic Regression with Class Weighting**
   - Fraud F1: 0.5562
   - ROC-AUC: 0.9587

2. **Logistic Regression + SMOTE**
   - Fraud F1: 0.5732
   - ROC-AUC: 0.9577
   - SMOTE provided slight improvement over class weighting

3. **Random Forest** (Best Model)
   - GridSearchCV hyperparameter tuning
   - Best params: n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1
   - Fraud F1: 0.6473 (threshold=0.50)
   - ROC-AUC: 0.9546

4. **XGBoost**
   - GridSearchCV hyperparameter tuning
   - Fraud F1: 0.6288
   - ROC-AUC: 0.9579

### 6. Model Optimization

- **Decision Threshold Optimization:** Validation set used to find optimal threshold (0.46) maximizing F1-score
- **10-Fold Cross-Validation:** Mean ROC-AUC of 0.9940 ± 0.0036 confirmed model stability

---

## Key Results

### Best Model: Random Forest (Optimized Threshold = 0.46)

| Metric | Score |
|--------|-------|
| **ROC-AUC** | **95.46%** |
| **Precision (Fraud)** | 53.90% |
| **Recall (Fraud)** | 82.18% |
| **F1-Score (Fraud)** | 65.10% |
| **PR-AUC** | 69.53% |

### Performance Comparison (Test Set)

| Model | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|-------|-----------|--------|----------|---------|--------|
| Logistic Regression (Class Weight) | 0.3966 | 0.9307 | 0.5562 | 0.9587 | 0.7520 |
| Logistic Regression + SMOTE | 0.4141 | 0.9307 | 0.5732 | 0.9577 | - |
| Random Forest (threshold=0.50) | 0.5571 | 0.7723 | 0.6473 | 0.9546 | 0.6953 |
| XGBoost (threshold=0.50) | 0.5625 | 0.7129 | 0.6288 | 0.9579 | 0.7209 |
| **Random Forest (threshold=0.46)** | **0.5390** | **0.8218** | **0.6510** | **0.9546** | **0.6953** |

### Confusion Matrix (Optimized Random Forest)

|  | Predicted Non-Fraud | Predicted Fraud |
|---|---------------------|-----------------|
| **Actual Non-Fraud** | 910 (TN) | 71 (FP) |
| **Actual Fraud** | 18 (FN) | 83 (TP) |

- **True Positives:** 83 fraud cases correctly identified
- **False Negatives:** 18 fraud cases missed (17.82% miss rate)
- **False Positives:** 71 non-fraud cases flagged (audit candidates)

### Top 5 Fraud Indicators (Random Forest Feature Importance)

1. **TotalReimbursement** (0.09)
2. **TotalDeductible** (0.09)
3. **UniqueProcedureCodes** (0.09)
4. **InpatientClaims** (~0.07)
5. **ClaimCount-related features** (~0.06)

Financial metrics and claim volumes dominate predictive power, consistent with EDA findings.

---

## Installation

### Prerequisites

- Python 3.11.11
- pip or conda package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/joneshshrestha/us-healthcare-fwa-detection.git
   cd us-healthcare-fwa-detection
   ```

2. **Create virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
   ```

---

## Usage

### Running the Analysis

1. **Place the dataset** in `./US Healthcare Dataset/` directory:
   - `Train_Beneficiarydata-1542865627584.csv`
   - `Train_Inpatientdata-1542865627584.csv`
   - `Train_Outpatientdata-1542865627584.csv`
   - `Train-1542865627584.csv`

2. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook fwa-detection.ipynb
   ```

3. **Execute cells sequentially** to reproduce:
   - Data loading and exploration
   - Feature engineering
   - Unsupervised learning (K-Means, Isolation Forest)
   - Supervised learning (Logistic Regression, Random Forest, XGBoost)
   - Model evaluation and optimization

### Key Notebook Sections

1. **Data Loading and Initial Exploration** - Load and examine datasets
2. **Data Preprocessing** - Feature engineering and aggregation
3. **Exploratory Data Analysis** - Statistical analysis and visualizations
4. **Unsupervised Learning** - Clustering and anomaly detection
5. **Feature Selection** - Mutual Information and correlation analysis
6. **Supervised Learning** - Model training and hyperparameter tuning
7. **Model Evaluation** - Performance metrics and threshold optimization
8. **Cross-Validation** - Stability assessment across folds

---

## Project Structure

```
us-healthcare-fwa-detection/
├── fwa-detection.ipynb          # Main analysis notebook
├── US Healthcare Dataset/       # Dataset directory (zip included in repo)
│   ├── Train_Beneficiarydata-1542865627584.csv
│   ├── Train_Inpatientdata-1542865627584.csv
│   ├── Train_Outpatientdata-1542865627584.csv
│   └── Train-1542865627584.csv
├── Final Report.pdf             # Comprehensive project report
└── README.md                    # This file
```

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.11.11 | Core language |
| NumPy | - | Numerical operations |
| Pandas | - | Data manipulation and aggregation |
| Matplotlib | - | Base visualizations |
| Seaborn | - | Statistical visualizations |
| Scikit-learn | - | ML algorithms, metrics, preprocessing |
| Imbalanced-learn | - | SMOTE oversampling |
| XGBoost | - | Gradient boosting classifier |

All libraries are open-source and standard for data science workflows.

---

## Conclusions

### Key Achievements

1. **Exceeded Performance Target:** 95.46% ROC-AUC (15.46 percentage points above 80% goal)
2. **High Recall:** 82.18% detection rate captures 4 out of 5 fraudulent providers
3. **Interpretable Features:** Financial metrics and claim volumes are primary fraud indicators
4. **Stable Model:** Cross-validation ROC-AUC of 99.40% ± 0.36% confirms robustness

### Practical Applications

- **Risk-Scoring Tool:** Deploy as automated provider risk assessment system
- **Audit Prioritization:** Focus investigative resources on high-risk providers (82% fraud capture rate)
- **Cost Savings:** Reduce manual investigation workload while maintaining high fraud detection
- **Decision Support:** Clear feature importance provides actionable insights for investigators

### Limitations

- **False Negatives:** 17.8% of fraud cases remain undetected
- **Synthetic Data:** Results based on Medicare-inspired data; real-world validation required
- **Tabular Features Only:** Does not incorporate temporal patterns or network relationships
- **Class Imbalance:** Even with SMOTE, minority class remains challenging

### Future Work

- Incorporate temporal analysis of claims patterns over time
- Build graph-based models over provider-beneficiary networks
- Validate on real Medicare claims data (subject to data access and privacy regulations)
- Implement continuous model updates as fraud patterns evolve
- Explore deep learning approaches for sequential claim analysis

---

## Data Source

This project uses a synthetic Medicare-inspired Healthcare Provider Fraud Detection dataset, which simulates real-world healthcare claims data while maintaining privacy and confidentiality.

---

## License

This project is available under the MIT License. See LICENSE file for details.

---

## Contact

**Jonesh Shrestha**  
- Website: [joneshshrestha.com](https://joneshshrestha.com)
- LinkedIn: [joneshshrestha](https://www.linkedin.com/in/joneshshrestha/)
- GitHub: [@joneshshrestha](https://github.com/joneshshrestha)