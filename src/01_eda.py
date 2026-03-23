"""
01_eda.py — Exploratory Data Analysis

Loads the raw dataset, prints key statistics, and saves
visualisations to outputs/plots/.

Run from the repo root:
    python src/01_eda.py
"""

import sys
from pathlib import Path

# Allow imports from the repo root (where config.py lives)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import (
    RAW_DATA, PLOT_EDA_OVERVIEW, PLOT_CORRELATIONS, make_dirs
)

make_dirs()


def load_data() -> pd.DataFrame:
    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"\nDataset not found at: {RAW_DATA}\n"
            f"Download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            f"and place creditcard.csv inside the dataset/ folder."
        )
    df = pd.read_csv(RAW_DATA)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns from {RAW_DATA.name}")
    return df


def print_summary(df: pd.DataFrame) -> None:
    counts     = df["Class"].value_counts()
    fraud_rate = counts[1] / len(df) * 100

    print(f"\n{'='*55}")
    print("DATASET SUMMARY")
    print(f"{'='*55}")
    print(f"  Shape       : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Missing vals: {df.isnull().sum().sum()}")
    print(f"\n  Legitimate  : {counts[0]:>8,}  ({100 - fraud_rate:.2f}%)")
    print(f"  Fraud       : {counts[1]:>8,}  ({fraud_rate:.4f}%)")
    print(f"  Imbalance   : {counts[0] // counts[1]}:1 ratio")

    print(f"\n  Amount stats by class:")
    print(df.groupby("Class")["Amount"].describe().round(2).to_string())

    time_hours = df["Time"].max() / 3600
    print(f"\n  Time span   : {time_hours:.1f} hours of transactions")


def plot_overview(df: pd.DataFrame) -> None:
    counts = df["Class"].value_counts()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Credit Card Fraud — EDA Overview", fontsize=13, fontweight="bold")

    # 1. Class distribution (log scale to make fraud visible)
    axes[0].bar(["Legitimate", "Fraud"], [counts[0], counts[1]],
                color=["#4A90D9", "#E05C5C"], edgecolor="white", linewidth=0.5)
    axes[0].set_yscale("log")
    axes[0].set_title("Class Distribution (log scale)")
    axes[0].set_ylabel("Count")
    for i, v in enumerate([counts[0], counts[1]]):
        axes[0].text(i, v * 1.3, f"{v:,}", ha="center", fontsize=10)

    # 2. Amount distribution by class
    axes[1].hist(df[df["Class"] == 0]["Amount"].clip(upper=500),
                 bins=60, alpha=0.6, color="#4A90D9", label="Legitimate", density=True)
    axes[1].hist(df[df["Class"] == 1]["Amount"].clip(upper=500),
                 bins=60, alpha=0.7, color="#E05C5C", label="Fraud",       density=True)
    axes[1].set_title("Amount Distribution (clipped at $500)")
    axes[1].set_xlabel("Amount ($)")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    # 3. Fraud rate by hour
    df = df.copy()
    df["Hour"] = (df["Time"] / 3600).astype(int)
    fraud_by_hour = df[df["Class"] == 1].groupby("Hour").size()
    legit_by_hour = df[df["Class"] == 0].groupby("Hour").size()

    ax_l = axes[2]
    ax_l.plot(legit_by_hour.index, legit_by_hour.values,
              color="#4A90D9", linewidth=1.5, label="Legitimate")
    ax_l.set_ylabel("Legitimate txns", color="#4A90D9")
    ax_l.tick_params(axis="y", labelcolor="#4A90D9")

    ax_r = ax_l.twinx()
    ax_r.plot(fraud_by_hour.index, fraud_by_hour.values,
              color="#E05C5C", linewidth=2, label="Fraud")
    ax_r.set_ylabel("Fraud txns", color="#E05C5C")
    ax_r.tick_params(axis="y", labelcolor="#E05C5C")

    ax_l.set_title("Transactions by Hour")
    ax_l.set_xlabel("Hour")
    lines = ax_l.get_lines() + ax_r.get_lines()
    ax_l.legend(lines, [l.get_label() for l in lines], fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOT_EDA_OVERVIEW, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {PLOT_EDA_OVERVIEW.relative_to(PLOT_EDA_OVERVIEW.parents[2])}")


def plot_correlations(df: pd.DataFrame) -> None:
    correlations = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False)
    top = correlations.head(12)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(top.index[::-1], top.values[::-1],
                   color=["#E05C5C" if v > 0.2 else "#4A90D9" for v in top.values[::-1]])
    ax.set_title("Top 12 Features — Absolute Correlation with Fraud Label", fontweight="bold")
    ax.set_xlabel("Absolute Correlation")
    ax.axvline(0.2, color="gray", linestyle="--", linewidth=0.8, label="0.2 threshold")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, top.values[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOT_CORRELATIONS, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {PLOT_CORRELATIONS.relative_to(PLOT_CORRELATIONS.parents[2])}")


def main():
    print("=" * 55)
    print("STEP 1 — EXPLORATORY DATA ANALYSIS")
    print("=" * 55)
    df = load_data()
    print_summary(df)
    plot_overview(df)
    plot_correlations(df)
    print(f"\nNext: python src/02_preprocess.py")


if __name__ == "__main__":
    main()