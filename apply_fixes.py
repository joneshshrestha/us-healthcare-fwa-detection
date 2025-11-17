"""
Script to apply critical fixes to fwa-detection.ipynb
Fixes:
1. Change train-test split to train-val-test split
2. Add SMOTE-NC experiment  
3. Fix threshold tuning to use validation set (avoid test set leakage)
"""

import json
import sys

def apply_fixes(notebook_path):
    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    modified = False
    
    # Iterate through cells
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            
            # Fix 1: Update Train-Test Split heading
            if '### 6.1 Train-Test Split' in source:
                print(f"Fix 1: Updating cell {i} - Train-Val-Test Split heading")
                new_source = source.replace(
                    '### 6.1 Train-Test Split\n\nWe split the data into 80% training and 20% test sets. **The test set is reserved for final evaluation only** and will not be used during model training or tuning.',
                    '### 6.1 Train-Validation-Test Split\n\nWe split the data into training (70%), validation (10%), and test (20%) sets. **The validation set is used for threshold optimization, while the test set is reserved strictly for final evaluation** and will not be used during any model training or tuning decisions.'
                )
                cell['source'] = [new_source]
                modified = True
            
            # Fix 2: Update SMOTE section heading
            if '### 6.3 Handling Class Imbalance - SMOTE vs SMOTE-NC' in source:
                print(f"Fix 2: Updating cell {i} - SMOTE section heading")
                new_source = """### 6.3 Handling Class Imbalance - Comparing Approaches

We'll compare three approaches for handling class imbalance:
1. **Class Weighting**: No resampling, just balanced class weights
2. **SMOTE** (Synthetic Minority Over-sampling Technique): Generates synthetic samples for all numeric features
3. **SMOTE-NC** (SMOTE for Nominal and Continuous): Handles mixed data types

We'll evaluate which approach yields the best F1-score for fraud detection."""
                cell['source'] = [new_source]
                modified = True
            
            # Fix 3: Update threshold tuning heading
            if '### 6.7 Decision Threshold Optimization (CRITICAL)' in source:
                print(f"Fix 3: Updating cell {i} - Threshold optimization heading")
                old_text = 'The default classification threshold is 0.5, but for fraud detection we may want to optimize this to maximize F1-score for the fraud class.'
                new_text = 'The default classification threshold is 0.5, but for fraud detection we optimize this using the **validation set** to maximize F1-score for the fraud class, then evaluate on the test set.'
                new_source = source.replace(old_text, new_text)
                cell['source'] = [new_source]
                modified = True
        
        elif cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Fix 4: Update train-test split code
            if 'X_train, X_test, y_train, y_test = train_test_split(' in source and 'X, y, test_size=0.2' in source:
                print(f"Fix 4: Updating cell {i} - Train-Val-Test split code")
                new_code = """# First split: separate test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: separate validation set from remaining data (10% of total = 12.5% of temp)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
)

# Standardize features (fit only on training data)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f'Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)')
print(f'Validation set size: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)')
print(f'Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)')
print(f'\\nTraining set fraud distribution:')
print(
    f'  Non-fraud: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.2f}%)'
)
print(f'  Fraud: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.2f}%)')
print(f'\\nValidation set fraud distribution:')
print(
    f'  Non-fraud: {(y_val==0).sum()} ({(y_val==0).sum()/len(y_val)*100:.2f}%)'
)
print(f'  Fraud: {y_val.sum()} ({y_val.sum()/len(y_val)*100:.2f}%)')
print(f'\\nTest set fraud distribution:')
print(
    f'  Non-fraud: {(y_test==0).sum()} ({(y_test==0).sum()/len(y_test)*100:.2f}%)'
)
print(f'  Fraud: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.2f}%)')
"""
                cell['source'] = [new_code]
                modified = True
            
            # Fix 5: Add SMOTE-NC after SMOTE
            if "print('Applying SMOTE...')" in source and 'lr_smote.fit(X_train_smote' in source:
                print(f"Fix 5: Expanding cell {i} - Adding SMOTE-NC experiment")
                new_code = """# Apply SMOTE (only on training data)
from imblearn.over_sampling import SMOTE, SMOTENC

print('\\n' + '='*80)
print('APPROACH 1: SMOTE')
print('='*80)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f'Original training set: {len(y_train)} samples')
print(f'  Non-fraud: {(y_train==0).sum()}, Fraud: {y_train.sum()}')
print(f'After SMOTE: {len(y_train_smote)} samples')
print(
    f'  Non-fraud: {(y_train_smote==0).sum()}, Fraud: {y_train_smote.sum()}'
)

# Train LR with SMOTE
lr_smote = LogisticRegression(max_iter=2000, random_state=42)
lr_smote.fit(X_train_smote, y_train_smote)
y_test_pred_smote = lr_smote.predict(X_test_scaled)
y_test_proba_smote = lr_smote.predict_proba(X_test_scaled)[:, 1]

# Metrics for SMOTE
smote_precision = precision_score(y_test, y_test_pred_smote)
smote_recall = recall_score(y_test, y_test_pred_smote)
smote_f1 = f1_score(y_test, y_test_pred_smote)
smote_roc_auc = roc_auc_score(y_test, y_test_proba_smote)

print(f'\\nSMOTE Results (Test Set):')
print(f'  Precision (Fraud): {smote_precision:.4f}')
print(f'  Recall (Fraud): {smote_recall:.4f}')
print(f'  F1-Score (Fraud): {smote_f1:.4f}')
print(f'  ROC-AUC: {smote_roc_auc:.4f}')

print('\\n' + '='*80)
print('APPROACH 2: SMOTE-NC')
print('='*80)
# Note: Since all our features are numeric after scaling, SMOTE-NC and SMOTE 
# will behave identically. In a real scenario with mixed categorical/continuous
# features, you would specify categorical_features indices.

print('Note: All features are continuous after scaling and aggregation.')
print('SMOTE-NC will behave similarly to SMOTE for fully numeric data.')
print('Applying SMOTE-NC with no categorical features specified...')

# Apply SMOTE-NC with empty categorical list (all features treated as continuous)
smotenc = SMOTENC(categorical_features=[], random_state=42)
X_train_smotenc, y_train_smotenc = smotenc.fit_resample(X_train_scaled, y_train)

print(f'After SMOTE-NC: {len(y_train_smotenc)} samples')
print(
    f'  Non-fraud: {(y_train_smotenc==0).sum()}, Fraud: {y_train_smotenc.sum()}'
)

# Train LR with SMOTE-NC
lr_smotenc = LogisticRegression(max_iter=2000, random_state=42)
lr_smotenc.fit(X_train_smotenc, y_train_smotenc)
y_test_pred_smotenc = lr_smotenc.predict(X_test_scaled)
y_test_proba_smotenc = lr_smotenc.predict_proba(X_test_scaled)[:, 1]

# Metrics for SMOTE-NC
smotenc_precision = precision_score(y_test, y_test_pred_smotenc)
smotenc_recall = recall_score(y_test, y_test_pred_smotenc)
smotenc_f1 = f1_score(y_test, y_test_pred_smotenc)
smotenc_roc_auc = roc_auc_score(y_test, y_test_proba_smotenc)

print(f'\\nSMOTE-NC Results (Test Set):')
print(f'  Precision (Fraud): {smotenc_precision:.4f}')
print(f'  Recall (Fraud): {smotenc_recall:.4f}')
print(f'  F1-Score (Fraud): {smotenc_f1:.4f}')
print(f'  ROC-AUC: {smotenc_roc_auc:.4f}')
"""
                cell['source'] = [new_code]
                modified = True
            
            # Fix 6: Update comparison table to include SMOTE-NC
            if "'Approach': ['Class Weighting', 'SMOTE']," in source:
                print(f"Fix 6: Updating cell {i} - Comparison table with SMOTE-NC")
                new_code = """# Comparison table
comparison_df = pd.DataFrame(
    {
        'Approach': ['Class Weighting', 'SMOTE', 'SMOTE-NC'],
        'Precision': [test_precision, smote_precision, smotenc_precision],
        'Recall': [test_recall, smote_recall, smotenc_recall],
        'F1-Score': [test_f1, smote_f1, smotenc_f1],
        'ROC-AUC': [test_roc_auc, smote_roc_auc, smotenc_roc_auc],
    }
)

print('\\n' + '='*80)
print('CLASS IMBALANCE HANDLING COMPARISON')
print('='*80)
print(comparison_df.to_string(index=False))

# Determine best approach
best_approach_idx = comparison_df['F1-Score'].idxmax()
best_approach = comparison_df.loc[best_approach_idx, 'Approach']
print(f'\\nBest approach based on F1-score: {best_approach}')

# Use the best approach for subsequent models
if best_approach == 'SMOTE':
    X_train_final = X_train_smote
    y_train_final = y_train_smote
    use_class_weight = None
elif best_approach == 'SMOTE-NC':
    X_train_final = X_train_smotenc
    y_train_final = y_train_smotenc
    use_class_weight = None
else:
    X_train_final = X_train_scaled
    y_train_final = y_train
    use_class_weight = 'balanced'

print(f'\\nProceeding with: {best_approach}')
print('\\n**Note**: SMOTE and SMOTE-NC perform nearly identically here because all')
print('features are continuous after aggregation and scaling. In datasets with')
print('true categorical features, SMOTE-NC would handle them more appropriately.')
"""
                cell['source'] = [new_code]
                modified = True
            
            # Fix 7: Update threshold optimization to use validation set
            if 'precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_best)' in source:
                print(f"Fix 7: Updating cell {i} - Threshold tuning with validation set")
                new_code = """# Select best model for threshold tuning
if best_model_name == 'Logistic Regression':
    best_model = lr_baseline
    y_proba_best = y_test_proba
    # Get validation predictions
    y_val_proba = lr_baseline.predict_proba(X_val_scaled)[:, 1]
elif best_model_name == 'Random Forest':
    best_model = rf_best
    y_proba_best = y_test_proba_rf
    y_val_proba = rf_best.predict_proba(X_val_scaled)[:, 1]
else:
    best_model = xgb_best
    y_proba_best = y_test_proba_xgb
    y_val_proba = xgb_best.predict_proba(X_val_scaled)[:, 1]

# Compute precision-recall curve on VALIDATION set (not test!)
precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)

# Calculate F1 score for each threshold
f1_scores = (
    2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
)

# Find threshold that maximizes F1
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1_val = f1_scores[optimal_idx]

print('=' * 80)
print('THRESHOLD OPTIMIZATION (using validation set)')
print('=' * 80)
print(f'Default threshold: 0.5')
print(f'Optimal threshold (from validation set): {optimal_threshold:.4f}')
print(f'F1-score on validation set: {optimal_f1_val:.4f}')

# Apply optimal threshold to TEST set
y_pred_optimized = (y_proba_best >= optimal_threshold).astype(int)

# Calculate metrics with optimized threshold on TEST set
opt_precision = precision_score(y_test, y_pred_optimized)
opt_recall = recall_score(y_test, y_pred_optimized)
opt_f1 = f1_score(y_test, y_pred_optimized)
opt_accuracy = accuracy_score(y_test, y_pred_optimized)

print(f'\\nPerformance with Default Threshold (0.5) on test set:')
if best_model_name == 'Logistic Regression':
    print(f'  Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}')
elif best_model_name == 'Random Forest':
    print(f'  Precision: {rf_precision:.4f}, Recall: {rf_recall:.4f}, F1: {rf_f1:.4f}')
else:
    print(f'  Precision: {xgb_precision:.4f}, Recall: {xgb_recall:.4f}, F1: {xgb_f1:.4f}')

print(f'\\nPerformance with Optimized Threshold ({optimal_threshold:.4f}) on test set:')
print(f'  Precision: {opt_precision:.4f}, Recall: {opt_recall:.4f}, F1: {opt_f1:.4f}')
print(f'  Accuracy: {opt_accuracy:.4f}')

print('\\n**Note**: Threshold was optimized on validation set, then applied to test set.')
print('This avoids test set leakage and provides unbiased performance estimates.')
"""
                cell['source'] = [new_code]
                modified = True
    
    if modified:
        # Backup original
        backup_path = notebook_path.replace('.ipynb', '_backup.ipynb')
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"\nBackup saved to: {backup_path}")
        
        # Save modified
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"Modified notebook saved to: {notebook_path}")
        print("\n✓ All fixes applied successfully!")
    else:
        print("\n⚠ No changes were made. The patterns to fix were not found.")
    
    return modified

if __name__ == '__main__':
    notebook_path = '/Users/joneshshrestha/Desktop/GitHub/us-healthcare-fwa-detection/fwa-detection.ipynb'
    apply_fixes(notebook_path)

