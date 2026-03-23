"""
05_threshold_tuning.py — Threshold Tuning

The default predict() threshold is 0.5.
For fraud detection you often want to lower it to catch more
fraud (higher recall) at the cost of more false positives.

This script sweeps thresholds and shows the trade-off clearly.

Run from the repo root:
    python src/05_threshold_tuning.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
from config import (
    X_TEST, Y_TEST, RF_PATH,
    THRESHOLD_ANALYSIS_CSV, PLOT_THRESHOLD,
    DEFAULT_THRESHOLD, make_dirs
)

make_dirs()


def sweep_thresholds(y_test, y_prob, thresholds):
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_test == 1)).sum())
        fp = int(((y_pred == 1) & (y_test == 0)).sum())
        fn = int(((y_pred == 0) & (y_test == 1)).sum())
        tn = int(((y_pred == 0) & (y_test == 0)).sum())
        rows.append({
            "threshold": round(t, 2),
            "recall":    round(recall_score(y_test,    y_pred, zero_division=0), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test,        y_pred, zero_division=0), 4),
            "flagged":   int(y_pred.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return pd.DataFrame(rows)


def print_sweep(df: pd.DataFrame) -> None:
    print(f"\n{'Threshold':>10}  {'Recall':>8}  {'Precision':>10}  {'F1':>8}  "
          f"{'Flagged':>10}  {'Missed (FN)':>13}")
    print("─" * 68)
    for _, r in df.iterrows():
        marker = "  ◀ default" if r["threshold"] == DEFAULT_THRESHOLD else ""
        print(f"  {r['threshold']:>6.2f}    {r['recall']:>8.4f}  {r['precision']:>10.4f}  "
              f"{r['f1']:>8.4f}  {r['flagged']:>10,}  {r['fn']:>13,}{marker}")


def print_recommendations(df: pd.DataFrame) -> None:
    best_f1  = df.loc[df["f1"].idxmax()]
    high_rec = df[df["recall"] >= 0.90]
    best_rec = high_rec.iloc[-1] if len(high_rec) else df.loc[df["recall"].idxmax()]

    print(f"\n{'='*55}")
    print("RECOMMENDED THRESHOLDS")
    print(f"{'='*55}")
    print(f"\n  Best F1 (balanced):  threshold={best_f1['threshold']}")
    print(f"    Recall={best_f1['recall']}  Precision={best_f1['precision']}  "
          f"Flagged={int(best_f1['flagged']):,}")

    print(f"\n  High recall (≥0.90): threshold={best_rec['threshold']}")
    print(f"    Recall={best_rec['recall']}  Precision={best_rec['precision']}  "
          f"Flagged={int(best_rec['flagged']):,}")

    print(f"""
  Business context guide:
    Catch every fraud (losses are expensive) → lower threshold (~0.2-0.3)
    Minimise false alarms (customer experience) → raise threshold (~0.6-0.7)
    Balanced                                    → best_f1 threshold above
""")


def plot_threshold(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Threshold Tuning — Precision/Recall/F1 Trade-off",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(df["threshold"], df["recall"],    color="#E05C5C", lw=2, label="Recall")
    ax.plot(df["threshold"], df["precision"], color="#4A90D9", lw=2, label="Precision")
    ax.plot(df["threshold"], df["f1"],        color="#27AE60", lw=2, label="F1")
    ax.axvline(DEFAULT_THRESHOLD, color="gray", ls="--", lw=1, label=f"Default ({DEFAULT_THRESHOLD})")
    best_t = df.loc[df["f1"].idxmax(), "threshold"]
    ax.axvline(best_t, color="#27AE60", ls=":", lw=1.5, label=f"Best F1 ({best_t})")
    ax.set(xlabel="Threshold", ylabel="Score", xlim=(0.05, 0.95))
    ax.set_title("Metrics vs Threshold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(df["threshold"], df["flagged"], color="#E8A835", lw=2, label="Flagged txns")
    ax.set(xlabel="Threshold", ylabel="Flagged transactions", xlim=(0.05, 0.95))
    ax.set_title("False Alarm Volume vs Threshold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(df["threshold"], df["fn"], color="#E05C5C", lw=2, ls="--", label="Missed fraud (FN)")
    ax2.set_ylabel("Missed frauds", color="#E05C5C")
    ax2.tick_params(axis="y", labelcolor="#E05C5C")
    ax2.legend(loc="center right", fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOT_THRESHOLD, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {PLOT_THRESHOLD.relative_to(PLOT_THRESHOLD.parents[2])}")


def main():
    print("=" * 55)
    print("STEP 5 — THRESHOLD TUNING")
    print("=" * 55)

    X_test  = pd.read_csv(X_TEST)
    y_test  = pd.read_csv(Y_TEST).squeeze()
    rf_model = joblib.load(RF_PATH)

    y_prob = rf_model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.05, 0.96, 0.05)
    df = sweep_thresholds(y_test, y_prob, thresholds)

    print_sweep(df)
    print_recommendations(df)
    plot_threshold(df)

    df.to_csv(THRESHOLD_ANALYSIS_CSV, index=False)
    print(f"Saved → {THRESHOLD_ANALYSIS_CSV.relative_to(THRESHOLD_ANALYSIS_CSV.parents[2])}")
    print(f"\nNext: python src/06_predict_new.py")


if __name__ == "__main__":
    main()