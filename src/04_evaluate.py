"""
04_evaluate.py — Model Evaluation

Loads both trained models, evaluates on the untouched test set,
compares with/without SMOTE, and generates:
  - Confusion matrices
  - ROC + Precision-Recall curves
  - Feature importance bar chart
  - results_summary.csv

Run from the repo root:
    python src/04_evaluate.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    recall_score, precision_score, f1_score
)
from sklearn.ensemble import RandomForestClassifier
import joblib
from config import (
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST,
    LR_PATH, RF_PATH, IMPORTANCE_CSV,
    RESULTS_SUMMARY_CSV,
    PLOT_CONFUSION, PLOT_ROC_PR, PLOT_IMPORTANCE,
    RF_N_ESTIMATORS, RANDOM_STATE,
    make_dirs
)

make_dirs()


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    rec     = recall_score(y_test, y_pred)
    prec    = precision_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    avg_pr  = average_precision_score(y_test, y_prob)
    cm      = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"              Predicted Legit   Predicted Fraud")
    print(f"  Actual Legit    {tn:>7,}          {fp:>6,}")
    print(f"  Actual Fraud    {fn:>7,}          {tp:>6,}")
    print(f"\n  Precision  : {prec:.4f}  ({tp}/{tp+fp} flagged were real fraud)")
    print(f"  Recall     : {rec:.4f}  ({tp}/{tp+fn} actual frauds caught)")
    print(f"  F1 Score   : {f1:.4f}")
    print(f"  ROC-AUC    : {roc_auc:.4f}")
    if fn:
        print(f"  ⚠  Missed {fn} real fraud transactions (False Negatives)")

    return {
        "name": name, "recall": rec, "precision": prec, "f1": f1,
        "roc_auc": roc_auc, "avg_pr": avg_pr,
        "y_pred": y_pred, "y_prob": y_prob, "cm": cm,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def plot_confusion_matrices(results: list) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Confusion Matrices — Test Set", fontsize=13, fontweight="bold")
    cmaps = ["Reds", "Blues", "Greens"]

    for ax, res, cmap in zip(axes, results, cmaps):
        sns.heatmap(res["cm"], annot=True, fmt=",d", cmap=cmap, ax=ax,
                    xticklabels=["Legit", "Fraud"],
                    yticklabels=["Legit", "Fraud"],
                    cbar=False, linewidths=0.5)
        ax.set_title(res["name"], fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.text(0.5, -0.22,
                f"Recall={res['recall']:.3f}  Prec={res['precision']:.3f}  F1={res['f1']:.3f}",
                transform=ax.transAxes, ha="center", fontsize=9, color="#444")

    plt.tight_layout()
    plt.savefig(PLOT_CONFUSION, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {PLOT_CONFUSION.relative_to(PLOT_CONFUSION.parents[2])}")


def plot_roc_pr(results: list, y_test) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Performance Curves — Test Set", fontsize=13, fontweight="bold")
    palette = ["#E05C5C", "#4A90D9", "#27AE60"]

    ax = axes[0]
    for res, color in zip(results, palette):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{res['name']}  (AUC={res['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC=0.5)")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for res, color in zip(results, palette):
        p, r, _ = precision_recall_curve(y_test, res["y_prob"])
        ax.plot(r, p, color=color, linewidth=2,
                label=f"{res['name']}  (AP={res['avg_pr']:.4f})")
    ax.axhline(y_test.mean(), color="gray", linestyle="--", linewidth=0.8,
               label=f"Baseline (prevalence={y_test.mean():.4f})")
    ax.set(xlabel="Recall", ylabel="Precision")
    ax.set_title("Precision-Recall Curve\n(better metric for imbalanced data)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_ROC_PR, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {PLOT_ROC_PR.relative_to(PLOT_ROC_PR.parents[2])}")


def plot_feature_importance() -> None:
    df = pd.read_csv(IMPORTANCE_CSV).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df["feature"][::-1], df["importance"][::-1],
                   color="#4A90D9", edgecolor="white", linewidth=0.3)
    ax.set_title("Top 15 Features — Random Forest Importance", fontweight="bold")
    ax.set_xlabel("Importance Score")
    for bar, val in zip(bars, df["importance"][::-1]):
        ax.text(val + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOT_IMPORTANCE, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {PLOT_IMPORTANCE.relative_to(PLOT_IMPORTANCE.parents[2])}")


def main():
    print("=" * 55)
    print("STEP 4 — MODEL EVALUATION")
    print("=" * 55)

    X_train = pd.read_csv(X_TRAIN)
    X_test  = pd.read_csv(X_TEST)
    y_train = pd.read_csv(Y_TRAIN).squeeze()
    y_test  = pd.read_csv(Y_TEST).squeeze()

    lr_model = joblib.load(LR_PATH)
    rf_model = joblib.load(RF_PATH)

    print(f"\nTest set: {len(X_test):,} samples | Fraud: {y_test.sum()} ({y_test.mean()*100:.4f}%)")

    # Train a no-SMOTE baseline so we can quantify SMOTE's improvement
    print(f"\n{'─'*50}")
    print("Training no-SMOTE baseline (for comparison)...")
    rf_base = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS,
                                     random_state=RANDOM_STATE, n_jobs=-1)
    rf_base.fit(X_train, y_train)

    print(f"\n{'='*55}")
    print("RESULTS")
    print(f"{'='*55}")

    res_base = evaluate_model(rf_base,    X_test, y_test, "RF — No SMOTE")
    res_lr   = evaluate_model(lr_model,   X_test, y_test, "Logistic Regression + SMOTE")
    res_rf   = evaluate_model(rf_model,   X_test, y_test, "Random Forest + SMOTE")

    recall_gain = (res_rf["recall"] - res_base["recall"]) / res_base["recall"] * 100
    f1_gain     = (res_rf["f1"]     - res_base["f1"])     / res_base["f1"]     * 100

    print(f"\n{'='*55}")
    print("SMOTE IMPACT")
    print(f"{'='*55}")
    print(f"  Recall  : {res_base['recall']:.4f} → {res_rf['recall']:.4f}  (+{recall_gain:.1f}%)")
    print(f"  F1      : {res_base['f1']:.4f} → {res_rf['f1']:.4f}  (+{f1_gain:.1f}%)")
    print(f"  ROC-AUC : {res_base['roc_auc']:.4f} → {res_rf['roc_auc']:.4f}")

    results = [res_base, res_lr, res_rf]
    plot_confusion_matrices(results)
    plot_roc_pr(results, y_test)
    plot_feature_importance()

    summary = pd.DataFrame([{
        "Model":         r["name"],
        "Recall":        round(r["recall"],    4),
        "Precision":     round(r["precision"], 4),
        "F1":            round(r["f1"],        4),
        "ROC_AUC":       round(r["roc_auc"],   4),
        "Avg_Precision": round(r["avg_pr"],    4),
    } for r in results])
    summary.to_csv(RESULTS_SUMMARY_CSV, index=False)

    print(f"\n{'='*55}")
    print("FINAL SUMMARY")
    print(f"{'='*55}")
    print(summary.to_string(index=False))
    print(f"\nSaved → {RESULTS_SUMMARY_CSV.relative_to(RESULTS_SUMMARY_CSV.parents[2])}")
    print(f"\nNext: python src/05_threshold_tuning.py")


if __name__ == "__main__":
    main()