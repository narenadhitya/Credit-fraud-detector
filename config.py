"""
config.py — single source of truth for all paths and settings.

Every script imports from here, so nothing is ever hardcoded.
Paths are always resolved relative to THIS file's location,
so the project works regardless of where you clone it.
"""

from pathlib import Path

# ── Root of the repo (wherever config.py lives) ──────────────
ROOT = Path(__file__).resolve().parent

# ── Directories ──────────────────────────────────────────────
DATA_DIR    = ROOT / "dataset"
SRC_DIR     = ROOT / "src"
OUTPUT_DIR  = ROOT / "outputs"
PLOTS_DIR   = OUTPUT_DIR / "plots"
MODELS_DIR  = OUTPUT_DIR / "models"

# ── Input file ───────────────────────────────────────────────
RAW_DATA    = DATA_DIR / "creditcard.csv"

# ── Processed splits (written by 02_preprocess.py) ───────────
X_TRAIN     = OUTPUT_DIR / "X_train.csv"
X_TEST      = OUTPUT_DIR / "X_test.csv"
Y_TRAIN     = OUTPUT_DIR / "y_train.csv"
Y_TEST      = OUTPUT_DIR / "y_test.csv"

# ── Saved models (written by 03_smote_and_train.py) ──────────
SCALER_PATH = MODELS_DIR / "amount_scaler.pkl"
LR_PATH     = MODELS_DIR / "logistic_regression.pkl"
RF_PATH     = MODELS_DIR / "random_forest.pkl"

# ── Result CSVs ──────────────────────────────────────────────
IMPORTANCE_CSV        = OUTPUT_DIR / "feature_importance.csv"
RESULTS_SUMMARY_CSV   = OUTPUT_DIR / "results_summary.csv"
THRESHOLD_ANALYSIS_CSV= OUTPUT_DIR / "threshold_analysis.csv"

# ── Plot filenames ────────────────────────────────────────────
PLOT_EDA_OVERVIEW     = PLOTS_DIR / "01_eda_overview.png"
PLOT_CORRELATIONS     = PLOTS_DIR / "02_feature_correlations.png"
PLOT_CONFUSION        = PLOTS_DIR / "03_confusion_matrices.png"
PLOT_ROC_PR           = PLOTS_DIR / "04_roc_pr_curves.png"
PLOT_IMPORTANCE       = PLOTS_DIR / "05_feature_importance.png"
PLOT_THRESHOLD        = PLOTS_DIR / "06_threshold_tuning.png"

# ── Model hyperparameters ─────────────────────────────────────
RANDOM_STATE     = 42
TEST_SIZE        = 0.2
SMOTE_STRATEGY   = 1.0   # 1.0 = balance classes fully (50/50)
SMOTE_K          = 5
RF_N_ESTIMATORS  = 100
LR_MAX_ITER      = 1000

# Default decision threshold (tune in 05_threshold_tuning.py)
DEFAULT_THRESHOLD = 0.5


def make_dirs():
    """Create all output directories if they don't exist."""
    for d in [DATA_DIR, OUTPUT_DIR, PLOTS_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)