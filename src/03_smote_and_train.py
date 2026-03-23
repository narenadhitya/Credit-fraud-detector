"""
03_smote_and_train.py — SMOTE + Model Training

Applies SMOTE to the training set only, then trains:
  - Logistic Regression
  - Random Forest

Both models are saved to outputs/models/.

Run from the repo root:
    python src/03_smote_and_train.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from config import (
    X_TRAIN, Y_TRAIN, LR_PATH, RF_PATH, IMPORTANCE_CSV,
    SMOTE_STRATEGY, SMOTE_K,
    RF_N_ESTIMATORS, LR_MAX_ITER, RANDOM_STATE,
    make_dirs
)

make_dirs()


def apply_smote(X_train, y_train):
    """
    SMOTE (Synthetic Minority Oversampling Technique):
      1. Pick a real fraud sample
      2. Find its k=5 nearest fraud neighbours
      3. Create a new point on the line between them
    Result: balanced training set with no duplicate fraud rows.
    """
    print(f"\n{'─'*45}")
    print("APPLYING SMOTE")
    print(f"{'─'*45}")
    print(f"  Before — Legit: {(y_train==0).sum():>7,} | Fraud: {(y_train==1).sum():>5,}")

    t0 = time.time()
    sm = SMOTE(
        sampling_strategy=SMOTE_STRATEGY,
        k_neighbors=SMOTE_K,
        random_state=RANDOM_STATE
    )
    X_res, y_res = sm.fit_resample(X_train, y_train)
    elapsed = time.time() - t0

    synthetic = (y_res == 1).sum() - (y_train == 1).sum()
    print(f"  After  — Legit: {(y_res==0).sum():>7,} | Fraud: {(y_res==1).sum():>5,}  "
          f"(+{synthetic:,} synthetic samples)")
    print(f"  Done in {elapsed:.1f}s")
    return X_res, y_res


def train_logistic_regression(X_train, y_train):
    print(f"\n{'─'*45}")
    print("TRAINING: Logistic Regression")
    print(f"{'─'*45}")
    t0 = time.time()
    model = LogisticRegression(
        max_iter=LR_MAX_ITER,
        C=1.0,
        solver="lbfgs",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    joblib.dump(model, LR_PATH)
    print(f"  Saved → {LR_PATH.relative_to(LR_PATH.parents[2])}")
    return model


def train_random_forest(X_train, y_train):
    print(f"\n{'─'*45}")
    print("TRAINING: Random Forest")
    print(f"{'─'*45}")
    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")
    joblib.dump(model, RF_PATH)
    print(f"  Saved → {RF_PATH.relative_to(RF_PATH.parents[2])}")
    return model


def save_feature_importance(model, feature_names):
    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    df.to_csv(IMPORTANCE_CSV, index=False)
    print(f"\n  Feature importance saved → {IMPORTANCE_CSV.relative_to(IMPORTANCE_CSV.parents[2])}")

    print(f"\n  Top 10 features:")
    for _, row in df.head(10).iterrows():
        bar = "█" * int(row["importance"] * 300)
        print(f"    {row['feature']:<15}  {row['importance']:.4f}  {bar}")


def main():
    print("=" * 55)
    print("STEP 3 — SMOTE + MODEL TRAINING")
    print("=" * 55)

    X_train = pd.read_csv(X_TRAIN)
    y_train = pd.read_csv(Y_TRAIN).squeeze()

    print(f"\nTraining set: {len(X_train):,} samples | "
          f"Fraud: {y_train.sum()} ({y_train.mean()*100:.4f}%)")

    X_res, y_res = apply_smote(X_train, y_train)
    train_logistic_regression(X_res, y_res)
    rf = train_random_forest(X_res, y_res)
    save_feature_importance(rf, X_train.columns.tolist())

    print(f"\n{'='*55}")
    print("TRAINING COMPLETE")
    print(f"{'='*55}")
    print("  Models trained on SMOTE-balanced data.")
    print("  Evaluation will use the ORIGINAL imbalanced test set.")
    print(f"\nNext: python src/04_evaluate.py")


if __name__ == "__main__":
    main()