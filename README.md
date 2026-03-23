# Credit Card Fraud Detection

End-to-end machine learning pipeline to detect fraudulent credit card transactions, addressing severe class imbalance (0.17% fraud rate) using SMOTE.

**Dataset:** 284,807 transactions · 492 fraud · [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## Results

| Model | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest (no SMOTE) | ~0.77 | ~0.95 | ~0.85 | ~0.97 |
| Logistic Regression + SMOTE | ~0.89 | ~0.06 | ~0.12 | ~0.97 |
| **Random Forest + SMOTE** | **~0.84** | **~0.93** | **~0.88** | **~0.98** |

SMOTE improved fraud recall by ~35% over training on imbalanced data alone.

---

## Project Structure

```
credit-card-fraud-detection/
├── config.py                    ← all paths & hyperparameters (edit here)
├── run_all.py                   ← run full pipeline in one command
├── requirements.txt
├── .gitignore
│
├── dataset/                     ← put creditcard.csv here (gitignored)
│
├── src/
│   ├── 01_eda.py                ← class distribution, correlations, plots
│   ├── 02_preprocess.py         ← scale Amount, train/test split
│   ├── 03_smote_and_train.py    ← SMOTE + Logistic Regression + Random Forest
│   ├── 04_evaluate.py           ← confusion matrix, ROC, PR curves
│   ├── 05_threshold_tuning.py   ← precision/recall trade-off analysis
│   └── 06_predict_new.py        ← score new transactions
│
└── outputs/                     ← generated at runtime (gitignored)
    ├── plots/                   ← 6 PNG charts
    ├── models/                  ← trained .pkl files + scaler
    ├── results_summary.csv
    ├── threshold_analysis.csv
    └── feature_importance.csv
```

---

## Setup

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle and place it at:
# dataset/creditcard.csv
```

---

## Run

```bash
# Full pipeline (recommended)
python run_all.py

# Or step by step
python src/01_eda.py
python src/02_preprocess.py
python src/03_smote_and_train.py
python src/04_evaluate.py
python src/05_threshold_tuning.py
python src/06_predict_new.py
```

All paths and hyperparameters are configured in `config.py` — no hardcoded paths anywhere.

---

## Key Concepts

**Why not just use accuracy?**
A model that predicts "not fraud" for every transaction gets 99.83% accuracy but catches zero fraudsters. Recall and F1 are the correct metrics here.

**Why SMOTE on training set only?**
Applying SMOTE before splitting causes data leakage — synthetic samples from training distribution bleed into the test set, inflating evaluation scores. Always split first, SMOTE second.

**Threshold tuning**
`predict()` uses 0.5 by default. Lowering the threshold increases recall (catch more fraud) at the cost of more false positives. `05_threshold_tuning.py` shows the full trade-off curve.

---

## Tech Stack

Python · Scikit-learn · Imbalanced-learn · Pandas · Matplotlib · Seaborn