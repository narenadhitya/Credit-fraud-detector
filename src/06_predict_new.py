"""
06_predict_new.py — Score New Transactions

Demonstrates how the trained model would be used in production
to score an incoming transaction in real time.

Run from the repo root:
    python src/06_predict_new.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import joblib
from config import X_TEST, Y_TEST, RF_PATH, SCALER_PATH, DEFAULT_THRESHOLD


def score_transactions(X: pd.DataFrame, model, threshold: float) -> pd.DataFrame:
    """
    Returns a DataFrame with fraud_prob and prediction for each row.
    In production: X comes from your transactions database,
    with Amount pre-scaled using the saved scaler.
    """
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    return pd.DataFrame({
        "fraud_probability": probs.round(4),
        "prediction":        ["FRAUD" if p else "Legit" for p in preds],
    })


def main():
    print("=" * 55)
    print("STEP 6 — SCORING NEW TRANSACTIONS")
    print("=" * 55)

    rf_model = joblib.load(RF_PATH)
    scaler   = joblib.load(SCALER_PATH)
    threshold = DEFAULT_THRESHOLD

    print(f"\nModel:     Random Forest")
    print(f"Threshold: {threshold}  (edit DEFAULT_THRESHOLD in config.py to change)")

    # Pull a sample of real + fraud from the test set to simulate new arrivals
    X_test = pd.read_csv(X_TEST)
    y_test = pd.read_csv(Y_TEST).squeeze()

    sample = pd.concat([
        X_test[y_test == 0].head(4),
        X_test[y_test == 1].head(4),
    ]).reset_index(drop=True)
    true_labels = ["Legit"] * 4 + ["FRAUD"] * 4

    results = score_transactions(sample, rf_model, threshold)
    results["actual"] = true_labels
    results["correct"] = results["prediction"] == results["actual"]

    print(f"\n{'─'*60}")
    print(f"  {'TxID':>4}  {'Fraud Prob':>12}  {'Prediction':>12}  {'Actual':>8}  {'OK?':>5}")
    print("─" * 60)
    for i, row in results.iterrows():
        flag = "✓" if row["correct"] else "✗"
        print(f"  {i+1:>4}  {row['fraud_probability']:>12.4f}  "
              f"{row['prediction']:>12}  {row['actual']:>8}  {flag:>5}")

    acc = results["correct"].mean()
    print(f"\n  Sample accuracy: {acc*100:.0f}%  ({results['correct'].sum()}/8)")

    print(f"""
{'='*55}
HOW TO USE IN PRODUCTION
{'='*55}

  from config import RF_PATH, SCALER_PATH, DEFAULT_THRESHOLD
  import joblib, pandas as pd

  model  = joblib.load(RF_PATH)
  scaler = joblib.load(SCALER_PATH)

  def score(raw_transaction: dict) -> dict:
      df = pd.DataFrame([raw_transaction])
      df["Amount_scaled"] = scaler.transform(df[["Amount"]])
      df = df.drop(columns=["Amount", "Time"], errors="ignore")

      prob = model.predict_proba(df)[0, 1]
      return {{
          "fraud_probability": round(prob, 4),
          "decision":          "BLOCK" if prob >= DEFAULT_THRESHOLD else "APPROVE",
      }}

  # Runs in milliseconds — suitable for real-time scoring.
""")


if __name__ == "__main__":
    main()