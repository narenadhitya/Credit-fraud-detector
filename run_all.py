"""
run_all.py — Run the complete pipeline end to end.

Run from the repo root:
    python run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("src/01_eda.py",              "EDA & Visualisation"),
    ("src/02_preprocess.py",       "Preprocessing & Split"),
    ("src/03_smote_and_train.py",  "SMOTE + Model Training"),
    ("src/04_evaluate.py",         "Evaluation & Plots"),
    ("src/05_threshold_tuning.py", "Threshold Tuning"),
    ("src/06_predict_new.py",      "Score New Transactions"),
]


def check_dataset():
    data_path = ROOT / "dataset" / "creditcard.csv"
    if not data_path.exists():
        print(f"\n✗ Dataset not found at: {data_path}")
        print("  Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("  Then place creditcard.csv inside the dataset/ folder.\n")
        sys.exit(1)
    size_mb = data_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Dataset found ({size_mb:.0f} MB)")


def check_dependencies():
    missing = []
    for pkg in ["pandas", "numpy", "sklearn", "imblearn", "matplotlib", "seaborn", "joblib"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n✗ Missing packages: {missing}")
        print("  Run: pip install -r requirements.txt\n")
        sys.exit(1)
    print("  ✓ All dependencies installed")


def main():
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION — FULL PIPELINE")
    print("=" * 60)

    print("\nPre-flight checks...")
    check_dataset()
    check_dependencies()

    total_start = time.time()

    for i, (script, desc) in enumerate(STEPS, 1):
        script_path = ROOT / script

        if not script_path.exists():
            print(f"\n✗ Script not found: {script_path}")
            print("  Make sure all src/ files are present.")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"[{i}/{len(STEPS)}] {desc}")
        print("=" * 60)

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT),      # always run from repo root
        )

        if result.returncode != 0:
            print(f"\n✗ Step {i} ({script}) failed with exit code {result.returncode}.")
            print("  Fix the error above, then re-run.")
            sys.exit(1)

        elapsed = time.time() - t0
        print(f"\n  ✓ Step {i} completed in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE  ({total_elapsed:.0f}s total)")
    print(f"{'='*60}")
    print("\nOutputs generated:")
    print(f"  {ROOT / 'outputs' / 'plots':<50} ← 6 PNG charts")
    print(f"  {ROOT / 'outputs' / 'models':<50} ← trained .pkl files")
    print(f"  {ROOT / 'outputs' / 'results_summary.csv'}")
    print(f"  {ROOT / 'outputs' / 'threshold_analysis.csv'}")
    print(f"  {ROOT / 'outputs' / 'feature_importance.csv'}")


if __name__ == "__main__":
    main()