# US Healthcare Fraud, Waste, and Abuse (FWA) Detection

Provider-level fraud detection on Medicare-inspired claims data using a full KDD workflow: feature engineering, unsupervised risk signals, feature selection, supervised modeling, and decision-threshold optimization.

## Project goals

- Detect potentially fraudulent healthcare providers (Fraud, Waste, and Abuse).
- Achieve at least **0.80 ROC-AUC** on a held-out test set.

## Dataset

This project uses four linked tables (joined via beneficiary IDs and provider IDs):

1. **Beneficiary data**: demographics, chronic conditions, annual reimbursements, and coverage info  
2. **Inpatient claims**: dates, reimbursements/deductibles, diagnosis and procedure codes, physician IDs  
3. **Outpatient claims**: similar structure (procedure codes are sparse)  
4. **Provider labels**: provider ID and binary fraud label (Yes/No)

**Scale and class balance (provider-level):**

- **5,410 providers**
- **558,211 claims** total
  - 40,474 inpatient
  - 517,737 outpatient
- **138,556 beneficiaries**
- Highly imbalanced labels: **9.35% fraud rate** (506 fraud, 4,904 non-fraud)

**Data quality checks:**

- 100% linkage between claim beneficiary IDs and beneficiary table
- Provider IDs in labels match providers in claims
- No duplicate claim IDs (claims are unique)

## Methodology (KDD process)

### 1) Data preprocessing and feature engineering

Because labels are at the **provider** level, all claim-level and beneficiary-level information is aggregated into **47 provider-level features**, grouped as:

- **Claim volume features (6)**: total claims, inpatient/outpatient counts, inpatient-outpatient ratio, unique beneficiaries, claims per beneficiary  
- **Financial features (7)**: total/avg reimbursements and deductibles, standard deviations, ratios, max/min  
- **Temporal features (2)**: average claim duration, average admission duration  
- **Medical code features (6)**: unique diagnosis/procedure codes, avg codes per claim, top-code ratio  
- **Physician features (5)**: unique physician counts by role + diversity ratio  
- **Demographic features (21)**: aggregated beneficiary demographics, chronic-condition burden, and percentages

**Missing data handling** was treated as structured (for example, diagnosis and procedure code “missingness” often means “no additional codes”), and DOD is used only to compute the percent of deceased beneficiaries.

**Scaling:** Features are standardized with `StandardScaler`. Log scaling was tested for skewed financial features, but not used in final modeling.

### 2) Train-validation-test split

A stratified split keeps the fraud rate consistent across splits:

- Train: **70%** (3,787 providers)
- Validation: **10%** (541 providers) for threshold tuning
- Test: **20%** (1,082 providers)

### 3) Class imbalance strategy

Two approaches were compared:

- Class weighting
- SMOTE oversampling

SMOTE produced the best overall precision-recall balance for the final models.

### 4) Unsupervised learning (risk signals)

- **KMeans** clustering evaluated across **k = 2 to 10**
- **Isolation Forest** anomaly detection (10% contamination)
- **PCA** used for 2D visualization of clusters and anomalies

### 5) Feature selection

- **Mutual Information** scoring
- `SelectPercentile` tested across **10% to 100%**
- Final feature subset selected via cross-validation

### 6) Supervised learning and tuning

Models evaluated:

- Logistic Regression (baseline)
- Random Forest (with `GridSearchCV`)
- XGBoost (with `GridSearchCV`)

Decision threshold is tuned on the validation set to maximize fraud-class F1.

## Results

### Best model

**Random Forest + SMOTE + optimized decision threshold (0.46)**

- Test ROC-AUC: **0.9546**
- Fraud precision: **0.5390**
- Fraud recall: **0.8218**
- Fraud F1: **0.6510**

Confusion matrix at threshold 0.46 (fraud-positive class):

- True positives: 83
- False negatives: 18
- False positives: 71
- True negatives: 910

10-fold cross-validation (Random Forest) showed strong stability:

- Mean ROC-AUC: **0.9940 ± 0.0036**

### Model comparison (fraud class, held-out test set)

| Model | Setup | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | class_weight=balanced | 0.3966 | 0.9307 | 0.5562 | 0.9587 | 0.7520 |
| Logistic Regression + SMOTE | SMOTE on training data | 0.4141 | 0.9307 | 0.5732 | 0.9577 |  |
| Random Forest | SMOTE, threshold=0.50 | 0.5571 | 0.7723 | 0.6473 | 0.9546 | 0.6953 |
| XGBoost | SMOTE, threshold=0.50 | 0.5625 | 0.7129 | 0.6288 | 0.9579 | 0.7209 |
| Random Forest | SMOTE, threshold=0.46 | 0.5390 | 0.8218 | 0.6510 | 0.9546 | 0.6953 |

### Interpretation and key signals

- EDA showed fraud providers tend to have higher claim volumes, higher reimbursements, greater code diversity, and differences in average duration patterns.
- Random Forest feature importance is dominated by financial and volume-related features (for example, total reimbursement and total deductible), consistent with EDA.
- Unsupervised methods complement supervised modeling by surfacing high-risk clusters and anomalies for targeted auditing.

## Visualizations

### Exploratory data analysis

![Figure 5.1.1: Distribution of Total Reimbursement by Fraud Status](assets/fig_5_1_1_total_reimbursement.png)

![Figure 5.1.2: Distribution of Total Claims by Fraud Status](assets/fig_5_1_2_total_claims.png)

### Unsupervised learning

![Figure 5.2.1: PCA Projection of Providers Colored by KMeans Cluster (k=3)](assets/fig_5_2_1_pca_kmeans_k3.png)

![Figure 5.2.2: PCA Projection of Providers by Fraud Label](assets/fig_5_2_2_pca_fraud_label.png)

### Threshold tuning

> Note: the report labels both plots as “Figure 5.3.1”. Here they are split into (a) and (b) for clarity.

![Figure 5.3.1a: F1-Score vs Classification Threshold (Validation Set)](assets/fig_5_3_1a_f1_vs_threshold.png)

![Figure 5.3.1b: Precision-Recall Trade-off vs Classification Threshold (Validation Set)](assets/fig_5_3_1b_pr_tradeoff_vs_threshold.png)

### Supplementary figures

![Figure D1: Top 20 Features by Random Forest Importance](assets/fig_d1_rf_feature_importance_top20.png)

![Figure D2: Isolation Forest anomaly detection and anomaly score distribution](assets/fig_d2_isolation_forest_pca_and_scores.png)

![Figure D3: Correlation matrix of key provider features](assets/fig_d3_correlation_matrix.png)

![Figure D4: Histograms of key provider features (log-transformed)](assets/fig_d4_histograms_log_transformed.png)

![Figure D5: Model performance comparisons](assets/fig_d5_model_performance_comparisons.png)

## Tools and libraries

- Python 3.11.11
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn
- imbalanced-learn (SMOTE)
- XGBoost

## Reproducibility

1. Create and activate a Python environment.
2. Install dependencies (example):
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
````

3. Run the notebook:

   * `fwa-detection.ipynb`

## Limitations and future work

* This project is based on Medicare-inspired synthetic data and provider-level tabular features only.
* Future directions include richer temporal modeling and graph-based methods over provider-beneficiary networks.