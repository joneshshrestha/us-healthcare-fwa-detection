# Critical Fixes to Apply to fwa-detection.ipynb

## Issue 1: Train-Val-Test Split (Fix Test Set Leakage)

### Cell 67 - Update Markdown:
Change from:
```
### 6.1 Train-Test Split

We split the data into 80% training and 20% test sets. **The test set is reserved for final evaluation only** and will not be used during model training or tuning.
```

To:
```
### 6.1 Train-Validation-Test Split

We split the data into training (70%), validation (10%), and test (20%) sets. **The validation set is used for threshold optimization, while the test set is reserved strictly for final evaluation** and will not be used during any model training or tuning decisions.
```

### Cell 68 - Update Code:
The current train_test_split code needs to be replaced with a train-val-test split.

FIND this code and replace the entire cell with the corrected version in the next section.

## Issue 2: Add SMOTE-NC Experiment

### Cell 72 - Update Markdown:
Change section title and description to properly reflect all three approaches being tested.

### Cell 73 - Expand SMOTE code:
Add SMOTE-NC experiment after SMOTE.

### Cell 74 - Update comparison:
Add SMOTE-NC as third row in comparison table.

## Issue 3: Fix Threshold Tuning (Use Validation Set)

### In the threshold optimization cell:
Change `precision_recall_curve(y_test, y_proba_best)` 
To: `precision_recall_curve(y_val, y_proba_val)`

And update all related code to use validation set for threshold tuning, then apply to test set.

---

## Detailed Code Replacements:

See the Python notebook cells below for exact replacements.

