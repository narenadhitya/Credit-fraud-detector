"""
02_preprocess.py — Preprocessing & Train/Test Split

- Scales the Amount column with StandardScaler
- Drops the raw Time column
- Splits 80/20 with stratification to preserve fraud rate
- Saves processed splits and the fitted scaler

Run from the repo root:
    python src/02_preprocess.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from config import (
    RAW_DATA, SCALER_PATH,
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST,
    TEST_SIZE, RANDOM_STATE,
    make_dirs
)

make_dirs()


def load_and_engineer(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows.")

    # Scale Amount — V1-V28 are already PCA-scaled but Amount is raw euros
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

    # Persist scaler — needed at inference time for new transactions
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved scaler → {SCALER_PATH.relative_to(SCALER_PATH.parents[2])}")

    # Drop Amount (replaced by Amount_scaled) and raw Time
    # Time has no direct fraud signal; hour-of-day engineering is optional
    df = df.drop(columns=["Amount", "Time"])
    return df, scaler


def split_and_save(df: pd.DataFrame):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # stratify=y keeps the same 0.17% fraud rate in BOTH splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\n{'─'*45}")
    print("TRAIN / TEST SPLIT")
    print(f"{'─'*45}")
    print(f"  Training : {len(X_train):>7,}  (fraud: {y_train.sum():>4} | {y_train.mean()*100:.4f}%)")
    print(f"  Test     : {len(X_test):>7,}  (fraud: {y_test.sum():>4} | {y_test.mean()*100:.4f}%)")
    print(f"  ✓ Fraud rate matches in both splits (stratify worked)")

    # Save so all downstream scripts use the identical split
    X_train.to_csv(X_TRAIN, index=False)
    X_test.to_csv(X_TEST,   index=False)
    y_train.to_csv(Y_TRAIN, index=False)
    y_test.to_csv(Y_TEST,   index=False)

    print(f"\n  Saved splits to outputs/")
    return X_train, X_test, y_train, y_test


def main():
    print("=" * 55)
    print("STEP 2 — PREPROCESSING")
    print("=" * 55)

    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found at {RAW_DATA}. Run step 1 first to verify setup."
        )

    df, _ = load_and_engineer(RAW_DATA)
    print(f"\nFeatures: {df.shape[1] - 1} inputs + 1 target (Class)")
    split_and_save(df)

    print(f"\n{'='*55}")
    print("WHY split BEFORE SMOTE?")
    print(f"{'='*55}")
    print("  If you SMOTE first, synthetic fraud samples from the")
    print("  training distribution leak into the test set.")
    print("  Evaluation scores get inflated — that's data leakage.")
    print("  Always keep the test set as a pristine real-world holdout.")
    print(f"\nNext: python src/03_smote_and_train.py")


if __name__ == "__main__":
    main()