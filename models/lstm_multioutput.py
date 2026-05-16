# =============================================================================
# lstm_multioutput.py
# Direct multi-output LSTM model for time-series forecasting.
# Adaptable to any dataset through the CONFIG dictionary.
#
# KEY DIFFERENCE compared to seq2seq_*.py:
#   - Uses a standard Keras Sequential architecture (no encoder-decoder).
#   - Predicts all future steps in a single forward pass.
#   - Uses model.fit() with EarlyStopping instead of a manual GradientTape loop.
#   - split_sequences returns only (X, y): no decoder input is required.
# =============================================================================

# ──────────────────────────────────────────────────────────────────────────────
#  0. CENTRAL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
# Main configuration dictionary used across the entire pipeline.
# Most experiment changes can be done directly here.
CONFIG = {
    # ── Paths ────────────────────────────────────────────────────────────────
    "data_path":    "/data/plus_z_1/ring_1.csv",
    "results_dir":  "/results",
    "results_file": "lstm_multi.txt",

    # ── Data filtering ───────────────────────────────────────────────────────
    "xtal_id":      30600,           # Crystal ID used for training and evaluation
    "target_var":   "calibration",   # Target column to predict
    "time_col":     "laser_datetime",# Column used for temporal ordering
    "calib_min":    0.7,             # Minimum accepted target value
    "calib_max":    1.0,             # Maximum accepted target value
    "train_year":   [2016],          # List of years used for training
    "test_year":    2017,            # Year used for external evaluation

    # ── Columns to retain after loading ────────────────────────────────────
    "keep_cols": ["xtal_id", "calibration", "int_deliv_inv_ub",
                  "laser_datetime", "time"],

    # ── Reproducibility ─────────────────────────────────────────────────────
    "seed": 1234,

    # =========================================================================
    #  DEFAULT CONFIGURATION (run_single)
    # =========================================================================
    "default": {
        # ── Predictor variables ─────────────────────────────────────────────
        "variables":      ["delta_lumi"],   # var6
        # n_steps == n_outputs == stride for non-overlapping windows
        "n_steps":        96,
        "n_outputs":      96,
        "stride":         96,

        # ── Architecture ──────────────────────────────────────────────────────
        "rnn_units":      [1024, 1024],  # Number of units per LSTM layer
        "dropout":        0.2,           # Dropout applied after each LSTM layer
        "internal_norm":  False,         # Apply BatchNormalization after each LSTM layer

        # ── Training ─────────────────────────────────────────────────────
        "epochs":         1,
        # batch_size: None → dynamic (32 if n_steps >= 48, 128 if n_steps < 48)
        "batch_size":     None,

        # ── Optimizer ───────────────────────────────────────────────────────
        # Options: "adam" | "adamw" | "sgd" | "rmsprop" | "adagrad"
        # None → use "adam" (library default value)
        "optimizer":      None,
        # Optimizer learning rate
        "learning_rate":  1e-3,

        # ── Loss function ────────────────────────────────────────────────
        # Options: "mse" | "mae" | "huber" | "logcosh"
        # None → use "mse"
        # mse     : strongly penalizes large errors; standard choice for regression.
        # mae     : more robust to outliers; converges more slowly.
        # huber   : hybrid mse/mae loss; mse for small errors, mae for large ones.
        # logcosh : similar to Huber but differentiable over the entire domain.
        "loss":           None,

        # ── Early stopping ────────────────────────────────────────────────────
        "patience":       50,   # Number of epochs without improvement before stopping
        # Training history metric to monitor: "val_loss" or "loss"
        "monitor":        "val_loss",
        # Minimum improvement required to reset the patience counter.
        # None → 0.0 (any improvement counts, with no minimum threshold)
        # Example: 1e-4 prevents stopping due to insignificant improvements
        "min_delta":      None,

        # ── Data normalization ────────────────────────────────────────────
        # Options: "MinMax" | "Standard" | "PowerTransformer" | "None"
        "norm_method":    "MinMax",

        # ── Train/validation split ────────────────────────────────────────────
        "val_split":      0.1,   # Fraction of data used for validation (0–1)

        # ── Shuffle windows before training ───────────────────────
        # True shuffles windows in the tf.data.Dataset (breaks temporal order)
        "shuffle":        False,

        # ── Hardware ──────────────────────────────────────────────────────────
        # True  → force CPU usage (reproducible, slower)
        # False → use GPU if available
        "use_cpu":       False,

        # ── Evaluation metrics computed on the TEST set ───────────────────────
        # Metrics are computed on the original scale after training.
        # Available: "mape" | "smape" | "mae" | "rmse" | "maxae" | "r2"
        # - mape  : Mean Absolute Percentage Error (×100). Main project metric.
        # - smape : Symmetric MAPE; more robust when the target approaches 0.
        # - mae   : Mean Absolute Error; expressed in the same units as the target.
        # - rmse  : Root Mean Squared Error; penalizes large errors.
        # - maxae : Maximum absolute error; indicates the worst individual case.
        # - r2    : Coefficient of determination (1=perfect, 0=mean baseline, <0=worse).
        "test_metrics":  ["mape", "smape", "mae", "rmse", "maxae", "r2"],

        # ── Saving ──────────────────────────────────────────────────────────
        # True → save model weights and scalers after training
        "save_weights":   True,
    },
    # =========================================================================
    # FIXED GRID PARAMETERS (not combined, applied to all runs)
    # =========================================================================
    "grid_fixed": {
        # ── Training epochs ───────────────────────────────────────────
        "epochs": 1,
        # ── Forecast horizons to evaluate ─────────────────────────────────────
        "horizons": [1, 12, 24, 36, 48, 60, 72, 84, 96],
        # ── Evaluation metrics (fixed for the entire grid) ─────────────────
        "test_metrics":  ["mape","mae","r2"],
        # ── Reference MAPE values per horizon (baseline to outperform) ───────
        "reference_metrics": {
            "mape": {
                1: 0.09, 12: 0.327, 24: 0.397, 36: 0.456,
                48: 0.529, 60: 0.578, 72: 0.63, 84: 0.674, 96: 0.723,},
            "mae":{1: 1, 12: 1, 24: 1, 36: 1,
            48: 1, 60: 1, 72: 1, 84: 1, 96: 1,},
            "r2":{1: 0.1, 12: 0.1, 24: 0.1, 36: 0.1,
            48: 0.1, 60: 0.1, 72: 0.1, 84: 0.1, 96: 0.1,},
        },
        # batch_size: None → dynamic (32 if n_steps >= 48, 128 if n_steps < 48)
        "batch_size":     None,
        # ── Fixed training parameters ─────────────────────────────────
        "val_split":      0.1,
        "patience":       50,
        # ── Hardware ──────────────────────────────────────────────────────────
        "use_cpu":       False,
        "save_weights":   False, # Do not enable when testing many configurations
    },

    # =========================================================================
    # GRID CONFIGURATION (parameters combined through Cartesian product)
    # =========================================================================
    "grid": {
        # ── Architectures to explore ──────────────────────────────────────────
        "rnn_units": [
            [1024, 1024],
        ],
        # ── Variable sets ────────────────────────────────────────────
        "variables": [
            ["delta_lumi"],
        ],
        # ── Normalizers ────────────────────────────────────────────────────
        "norm_method": [
            "MinMax",
        ],
        "internal_norm": [
            False,
        ],
        # ── Dropout after each LSTM layer ───────────────────────────────────
        "dropout": [
            0.2,
        ],
        # ── Shuffling of windows ───────────────────────────────────────────────
        "shuffle": [
            False,
        ],
        # ── Optimizer ───────────────────────────────────────────────────────
        # None → uses the default value defined in the code ("adam")
        "optimizer": [
            None,
        ],
        # ── Learning rate ───────────────────────────────────────────────
        "learning_rate": [
            1e-3,
        ],
        # ── Loss function ────────────────────────────────────────────────
        # None → uses the default value defined in the code ("mse")
        "loss": [
            None,
        ],
        # ── Metric to monitor for early stopping ────────────────────────────
        "monitor": [
            "val_loss",
        ],
        # ── Minimum improvement threshold for early stopping ──────────────────
        # None → 0.0 (no threshold)
        "min_delta": [
            None,
        ],
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import gc
import random
import traceback
from collections import defaultdict
from itertools import product

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping

# ──────────────────────────────────────────────────────────────────────────────
# 2. REPRODUCIBILITY AND ENVIRONMENT SETUP
# ──────────────────────────────────────────────────────────────────────────────

def _set_env_vars(seed: int) -> None:
    """Set environment variables for reproducibility (deterministic behavior)."""
    os.environ.update({
        "PYTHONHASHSEED":         str(seed),
        "TF_DETERMINISTIC_OPS":   "1",
        "TF_CUDNN_DETERMINISTIC": "1",
        "OMP_NUM_THREADS":        "1",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
    })


def reset_environment(seed: int = 1234, use_cpu: bool = True) -> None:
    """Clear the Keras session and reset all seeds for reproducibility."""
    tf.keras.backend.clear_session()
    gc.collect()

    if use_cpu:
        # Disable GPU by hiding all CUDA devices
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    _set_env_vars(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    print(f"Environment reset — seed set for reproducibility: {seed}")


# ── Initialization upon import ────────────────────────────────────────────────
# Seeds are set at module load time so that any object created before the first
# explicit reset_environment() call is also deterministic.
_seed = CONFIG["seed"]
_set_env_vars(_seed)
random.seed(_seed)
np.random.seed(_seed)
tf.random.set_seed(_seed)
tf.keras.utils.set_random_seed(_seed)
tf.config.experimental.enable_op_determinism()

# ── GPU memory growth ─────────────────────────────────────────────────────────
# Enable memory growth to prevent TensorFlow from reserving all VRAM at startup.
# This is critical for long sessions or when the GPU is shared with other processes:
# without this option, TF allocates all available memory from the first import,
# which can cause OOM errors if another process also uses the GPU.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[GPU] Memory growth enabled on {len(gpus)} GPU(s)")


# ──────────────────────────────────────────────────────────────────────────────
# 3. DATA LOADING AND PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def load_data(cfg: dict = CONFIG) -> pd.DataFrame:
    """
    Load the CSV file, filter by crystal ID and calibration range,
    parse datetime columns, and sort chronologically.

    Args:
        cfg: Global configuration dictionary (see CONFIG).

    Returns:
        Filtered and sorted DataFrame ready for feature engineering.
    """
    df = pd.read_csv(cfg["data_path"])
    df = df[df["xtal_id"] == cfg["xtal_id"]][cfg["keep_cols"]].copy()
    df = df[(df[cfg["target_var"]] >= cfg["calib_min"]) &
            (df[cfg["target_var"]] <= cfg["calib_max"])]

    # Detect date columns dynamically (avoids hardcoding column names)
    date_cols = [c for c in ["time", "laser_datetime"] if c in df.columns]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S")

    return df.sort_values(cfg["time_col"]).reset_index(drop=True)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features for a single crystal:
      dint_dtime, delta_lumi, delta_t, days_no_int.

    Operates on a copy of the input DataFrame and returns the result,
    leaving the original unmodified.

    Args:
        df: DataFrame for a single crystal, sorted by time.

    Returns:
        DataFrame with additional engineered feature columns.
    """
    i = df.copy().reset_index(drop=True)

    dt_sec = i["time"].diff().dt.total_seconds()
    d_lum = i["int_deliv_inv_ub"].diff()

    i["dint_dtime"]   = d_lum / dt_sec          # Luminosity rate of change
    i["delta_lumi"]   = d_lum                   # Absolute luminosity increment
    i["delta_t"]      = i["laser_datetime"].diff().dt.total_seconds()  # Time between laser shots
    i["days_no_int"] = i["time"].diff().dt.days  # Days since last interaction

    # ── Edge case and outlier corrections ────────────────────────────────────
    # First row has no previous value; copy the rate from the next row
    i.loc[0, "dint_dtime"] = i.loc[1, "dint_dtime"]
    # Reset rate to 0 on days with no interaction (no luminosity delivered)
    i.loc[i["days_no_int"] > 0,  "dint_dtime"]              = 0
    # Zero-fill the first row for difference-based features
    i.loc[0, ["days_no_int", "delta_t", "delta_lumi"]]      = 0
    # Suppress spurious deltas after long gaps (> 30 days without interaction)
    i.loc[i["days_no_int"] > 30, ["delta_lumi", "delta_t"]] = 0
    # Fill any remaining NaN/inf values in the rate column
    i["dint_dtime"] = i["dint_dtime"].bfill().ffill()
    i.loc[~np.isfinite(i["dint_dtime"]), "dint_dtime"]       = 0

    return i


def prepare_splits(cfg: dict = CONFIG):
    """
    Load data, apply feature engineering per year, and return
    (df_train, df_test) as a tuple.

    Feature engineering is applied year-by-year so that diff()-based
    computations reset at each period boundary, preserving the intended
    behavior of the original codebase.

    Args:
        cfg: Global configuration dictionary (see CONFIG).

    Returns:
        Tuple (df_train, df_test) with engineered features.

    Raises:
        ValueError: If either split is empty after filtering.
    """
    df       = load_data(cfg)
    time_col = cfg["time_col"]
    yr       = df[time_col].dt.year

    # train_year is a list → use .isin() to support multiple training years
    train_years = cfg["train_year"] if isinstance(cfg["train_year"], list) else [cfg["train_year"]]
    df_train = feature_engineering(
        df[yr.isin(train_years)].copy().reset_index(drop=True)
    )
    df_test  = feature_engineering(
        df[yr == cfg["test_year"]].copy().reset_index(drop=True)
    )

    if len(df_train) == 0:
        raise ValueError(
            f"df_train is empty. Check that 'train_year'={cfg['train_year']} "
            f"exists in CSV '{cfg['data_path']}' for crystal "
            f"xtal_id={cfg['xtal_id']} and that the filters "
            f"calib_min/calib_max ({cfg['calib_min']}–{cfg['calib_max']}) "
            f"do not discard all rows."
        )
    if len(df_test) == 0:
        raise ValueError(
            f"df_test is empty. Check that 'test_year'={cfg['test_year']} "
            f"exists in CSV '{cfg['data_path']}' for crystal "
            f"xtal_id={cfg['xtal_id']} and that the filters "
            f"calib_min/calib_max ({cfg['calib_min']}–{cfg['calib_max']}) "
            f"do not discard all rows."
        )

    return df_train, df_test


# ──────────────────────────────────────────────────────────────────────────────
# 4. VARIABLE GROUPS (adaptable)
# ──────────────────────────────────────────────────────────────────────────────
# Pre-defined input feature sets for convenience. Pass any of these lists
# as the `var` argument to train_model() or evaluate_model().
VARIABLE_SETS = {
    "var1": ["int_deliv_inv_ub", "dint_dtime"],
    "var2": ["int_deliv_inv_ub", "dint_dtime", "delta_lumi"],
    "var3": ["int_deliv_inv_ub", "dint_dtime", "delta_t"],
    "var4": ["int_deliv_inv_ub"],
    "var5": ["dint_dtime"],
    "var6": ["delta_lumi"],
}


# ──────────────────────────────────────────────────────────────────────────────
# 5. WINDOWING (direct multi-output — no decoder)
# ──────────────────────────────────────────────────────────────────────────────

def split_sequences(
    sequences: np.ndarray,
    n_steps: int,
    n_outputs: int = 1,
    stride: int = 1,
):
    """
    Generate sliding windows for direct multi-output prediction.

    Expected column layout in `sequences`:
        [var1, var2, ..., calibration, target]
        ─ target: last column (index n_cols - 1)

    Args:
        sequences : 2-D array of shape (T, n_cols).
        n_steps   : number of past time steps used as input (look-back window).
        n_outputs : number of future steps to predict per window.
        stride    : step size between consecutive windows.
                    Set stride == n_steps for non-overlapping windows.

    Returns:
        X  (N, n_steps, n_features)  ← all columns except target
        y  (N, n_outputs)            ← future target values
    """
    assert sequences.shape[1] >= 2, \
        "At least 2 columns are required (features + target)."

    X_list, y_list = [], []
    T     = sequences.shape[0]
    y_col = sequences.shape[1] - 1   # last column = target

    for i in range(0, T - n_steps - n_outputs + 1, stride):
        end_ix  = i + n_steps
        out_end = end_ix + n_outputs
        X_list.append(sequences[i:end_ix, :-1])          # (n_steps, n_features)
        y_list.append(sequences[end_ix:out_end, y_col])  # (n_outputs,)

    return np.array(X_list), np.array(y_list)


# ──────────────────────────────────────────────────────────────────────────────
# 6. MULTI-OUTPUT MODEL
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    input_shape: tuple,
    n_outputs: int = 1,
    rnn_units: list = None,
    dropout: float = 0.2,
    use_batchnorm: bool = False,
    optimizer: str = None,
    learning_rate: float = 1e-3,
    loss: str = None,
) -> Sequential:
    """
    Build a multi-output Sequential LSTM model.

    Architecture:
        Input → [LSTM → (BatchNorm) → Dropout] × L → Dense(n_outputs)

    Args:
        input_shape   : (n_steps, n_features) tuple.
        n_outputs     : number of future time steps to predict simultaneously.
        rnn_units     : list of integers, one per LSTM layer (e.g. [256, 128]).
                        Defaults to [128, 32, 32] if None.
        dropout       : dropout rate applied after each LSTM layer (0–1).
        use_batchnorm : if True, inserts BatchNormalization after each LSTM.
        optimizer     : optimizer name (see _build_optimizer). None → "adam".
        learning_rate : learning rate passed to the optimizer.
        loss          : loss function name (see _build_loss). None → "mse".

    Returns:
        Compiled Keras Sequential model.
    """
    if rnn_units is None:
        rnn_units = [128, 32, 32]

    model = Sequential()
    model.add(Input(shape=input_shape))

    for i, units in enumerate(rnn_units):
        # Only the last LSTM layer should NOT return sequences
        return_sequences = i < len(rnn_units) - 1
        model.add(LSTM(units, return_sequences=return_sequences))
        if use_batchnorm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout))

    # Single Dense layer produces all forecast steps at once
    model.add(Dense(n_outputs))
    model.compile(
        optimizer=_build_optimizer(optimizer, learning_rate),
        loss=_build_loss(loss),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    return model


# ──────────────────────────────────────────────────────────────────────────────
# 7. NORMALIZATION HELPER
# ──────────────────────────────────────────────────────────────────────────────

def _make_scaler(method: str):
    """
    Return a scaler instance for the requested normalization method.

    Args:
        method : "MinMax" | "Standard" | "PowerTransformer" | "None"

    Returns:
        Fitted-ready scaler instance, or None if method is "None".

    Raises:
        ValueError: If an unrecognized method name is provided.
    """
    options = {
        "MinMax":           MinMaxScaler,
        "Standard":         StandardScaler,
        "PowerTransformer": PowerTransformer,
        "None":             None,
    }
    if method not in options:
        raise ValueError(f"Invalid scaler: '{method}'. Options: {list(options)}")
    cls = options[method]
    return cls() if cls is not None else None


# ──────────────────────────────────────────────────────────────────────────────
# 7b. COMPILATION AND METRICS HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _build_optimizer(name: str, learning_rate: float):
    """
    Instantiate a Keras optimizer from its string name.

    Args:
        name          : "adam" | "adamw" | "sgd" | "rmsprop" | "adagrad"
                        None   → defaults to "adam"
        learning_rate : learning rate to apply to the optimizer.

    Returns:
        A tf.keras.optimizers instance ready to pass to model.compile().

    Raises:
        ValueError: If an unrecognized optimizer name is provided.
    """
    name = (name or "adam").lower()
    options = {
        "adam":    lambda: tf.keras.optimizers.Adam(learning_rate=learning_rate),
        "adamw":   lambda: tf.keras.optimizers.AdamW(learning_rate=learning_rate),
        "sgd":     lambda: tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
        "rmsprop": lambda: tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        "adagrad": lambda: tf.keras.optimizers.Adagrad(learning_rate=learning_rate),
    }
    if name not in options:
        raise ValueError(f"Invalid optimizer: '{name}'. Options: {list(options)}")
    return options[name]()


def _build_loss(name: str) -> str:
    """
    Validate and normalize the loss function name for Keras.

    A string is returned instead of a loss instance because model.compile()
    accepts both forms and strings are more readable in the Keras training log.

    Args:
        name : "mse" | "mae" | "huber" | "logcosh"
               None  → defaults to "mse"

    Returns:
        Normalized loss name string accepted directly by model.compile().

    Raises:
        ValueError: If an unrecognized loss name is provided.
    """
    name = (name or "mse").lower()
    options = {"mse", "mae", "huber", "logcosh"}
    if name not in options:
        raise ValueError(f"Invalid loss: '{name}'. Options: {list(options)}")
    return name


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      metrics: list) -> dict:
    """
    Compute the requested evaluation metrics on the original (unscaled) values.

    Args:
        y_true   : ground-truth values (1-D array).
        y_pred   : model predictions (1-D array).
        metrics  : list of metric names to compute.
                   Available: "mape" | "smape" | "mae" | "rmse" | "maxae" | "r2"

    Returns:
        Dictionary {metric_name: value}.

    Interpretation guide for calibration series (range ~0.7–1.0):
        mape  : mean percentage error; directly comparable across horizons.
        smape : more stable than mape when true values approach zero.
        mae   : error in calibration units; easy to interpret physically.
        rmse  : like mae but penalizes error peaks; useful for detecting failures.
        maxae : worst-case point error; relevant for reconstruction guarantees.
        r2    : fraction of variance explained; 1=perfect, 0=mean baseline, <0=worse.
    """
    from sklearn.metrics import (
        mean_absolute_percentage_error,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    results = {}
    for m in metrics:
        name = m.lower()
        if name == "mape":
            results["mape"]  = 100 * mean_absolute_percentage_error(y_true, y_pred)
        elif name == "smape":
            # Symmetric MAPE: avoids asymmetry issues when predictions are close to 0
            denom = (np.abs(y_true) + np.abs(y_pred)) / 2
            smape = np.mean(np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)) * 100
            results["smape"] = smape
        elif name == "mae":
            results["mae"]   = mean_absolute_error(y_true, y_pred)
        elif name == "rmse":
            results["rmse"]  = np.sqrt(mean_squared_error(y_true, y_pred))
        elif name == "maxae":
            results["maxae"] = float(np.max(np.abs(y_true - y_pred)))
        elif name == "r2":
            results["r2"]    = r2_score(y_true, y_pred)
        else:
            print(f"[WARNING] Unknown metric ignored: '{m}'")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 8. TRAINING
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    df_train:        pd.DataFrame,
    var:             list,
    target_var:      str   = "calibration",
    n_steps:         int   = 10,
    n_outputs:       int   = 1,
    stride:          int   = 1,
    rnn_units:       list  = None,
    dropout:         float = 0.2,
    epochs:          int   = 100,
    batch_size:      int   = 32,
    optimizer:       str   = None,        # None → "adam"; see _build_optimizer()
    learning_rate:   float = 1e-3,
    loss:            str   = None,        # None → "mse";  see _build_loss()
    ext_norm_method: str   = "MinMax",
    internal_norm:   bool  = False,
    val_split_ratio: float = 0.1,
    shuffle:         bool  = False,       # True shuffles the tf.data.Dataset each epoch
    patience:        int   = 50,
    monitor:         str   = "val_loss",  # "val_loss" | "loss"
    min_delta:       float = None,        # None → 0.0 (no minimum improvement threshold)
    save_weights:    bool  = False,
    results_dir:     str   = None,
    xtal_id:         int   = None,
    time_col:        str   = "laser_datetime",
):
    """
    Train the multi-output LSTM model on df_train using a tf.data.Dataset pipeline.

    Using tf.data.Dataset (instead of passing NumPy arrays directly to model.fit)
    enables automatic prefetching and a more efficient GPU pipeline, matching the
    behavior of the seq2seq models in this suite.

    If save_weights=True, the following files are written:
        <results_dir>/<xtal_id>_multi_<n_steps>_<n_outputs>_<norm>/model_<n_steps>steps.weights.h5
        <results_dir>/<xtal_id>_multi_<n_steps>_<n_outputs>_<norm>/scaler_X_<n_steps>steps.pkl
        <results_dir>/<xtal_id>_multi_<n_steps>_<n_outputs>_<norm>/scaler_y_<n_steps>steps.pkl
        <results_dir>/<xtal_id>_multi_<n_steps>_<n_outputs>_<norm>/n_vars.txt

    Args:
        df_train        : Training DataFrame with feature-engineered columns.
        var             : List of predictor column names.
        target_var      : Name of the target column to predict.
        n_steps         : Look-back window size (number of past time steps as input).
        n_outputs       : Number of future steps to predict simultaneously.
        stride          : Step size between consecutive windows.
        rnn_units       : List of units per LSTM layer. Defaults to [128, 32, 32].
        dropout         : Dropout rate after each LSTM layer.
        epochs          : Maximum number of training epochs.
        batch_size      : Mini-batch size for training.
        optimizer       : Optimizer name. None → "adam".
        learning_rate   : Optimizer learning rate.
        loss            : Loss function name. None → "mse".
        ext_norm_method : External (input) normalization method.
        internal_norm   : If True, adds BatchNormalization after each LSTM layer.
        val_split_ratio : Fraction of training data used for validation.
        shuffle         : If True, shuffles the training dataset each epoch.
        patience        : Early stopping patience (epochs without improvement).
        monitor         : Metric to monitor for early stopping.
        min_delta       : Minimum improvement to reset patience counter.
        save_weights    : If True, saves model weights and scalers to disk.
        results_dir     : Directory for saving weights (required if save_weights=True).
        xtal_id         : Crystal identifier used in file naming.
        time_col        : Column name used for temporal sorting.

    Returns:
        Tuple (model, history, scaler_X, scaler_y).
    """
    if rnn_units is None:
        rnn_units = [128, 32, 32]

    df_train = df_train.sort_values(time_col).copy()

    # ── Feature and target preparation ───────────────────────────────────────
    data = df_train[var + [target_var]].copy()
    data["target"] = data[target_var].copy()

    # ── Scaling ───────────────────────────────────────────────────────────────
    # scaler_X scales both predictor features and the calibration column (as context).
    # scaler_y scales only the target column (used for inverse transform at evaluation).
    scaler_X = _make_scaler(ext_norm_method)
    scaler_y = _make_scaler(ext_norm_method)

    if scaler_X is not None:
        data[var + [target_var]] = scaler_X.fit_transform(data[var + [target_var]])
    if scaler_y is not None:
        data["target"] = scaler_y.fit_transform(data[["target"]])

    # ── Sliding window generation ─────────────────────────────────────────────
    X_full, y_full = split_sequences(data.values, n_steps, n_outputs, stride)

    split    = int(len(X_full) * (1 - val_split_ratio))
    X_train, X_val = X_full[:split], X_full[split:]
    y_train, y_val = y_full[:split], y_full[split:]

    # ── tf.data.Dataset pipeline ─────────────────────────────────────────────
    # Wrapping arrays in tf.data.Dataset enables prefetching and an efficient
    # GPU pipeline. shuffle=True randomizes window order before each epoch;
    # False preserves temporal order (recommended for time-series tasks).
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    if shuffle:
        train_ds = train_ds.shuffle(
            buffer_size=min(8192, len(X_train)), seed=CONFIG["seed"]
        )
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds   = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

    # Free NumPy arrays after building datasets to reduce peak memory usage
    del X_train, y_train, X_val, y_val, X_full, y_full
    gc.collect()

    # ── Model construction ────────────────────────────────────────────────────
    # Input features = all columns except the separate target column
    input_shape = (n_steps, data.shape[1] - 1)
    model = build_model(
        input_shape,
        n_outputs=n_outputs,
        rnn_units=rnn_units,
        dropout=dropout,
        use_batchnorm=internal_norm,
        optimizer=optimizer,
        learning_rate=learning_rate,
        loss=loss,
    )

    # ── Early stopping ────────────────────────────────────────────────────────
    if monitor not in ("val_loss", "loss"):
        raise ValueError(f"monitor must be 'val_loss' or 'loss', got: '{monitor}'")

    # Resolve minimum improvement threshold (None maps to 0.0)
    _min_delta = min_delta if min_delta is not None else 0.0

    early_stop = EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=_min_delta,
        restore_best_weights=True,  # Revert to best epoch weights on stop
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=0,
        callbacks=[early_stop],
    )

    # ── Save weights and scalers (optional) ───────────────────────────────────
    if save_weights:
        if results_dir is None:
            raise ValueError("results_dir is required when save_weights=True.")
        xtal_str  = str(xtal_id) if xtal_id is not None else "xtal_unknown"
        norm_str  = ext_norm_method.lower()
        weights_dir = os.path.join(
            results_dir, f"{xtal_str}_multi_{n_steps}_{n_outputs}_{norm_str}"
        )
        os.makedirs(weights_dir, exist_ok=True)
        model.save_weights(
            os.path.join(weights_dir, f"model_{n_steps}steps.weights.h5")
        )
        joblib.dump(scaler_X, os.path.join(weights_dir, f"scaler_X_{n_steps}steps.pkl"))
        joblib.dump(scaler_y, os.path.join(weights_dir, f"scaler_y_{n_steps}steps.pkl"))
        # Save variable list to detect mismatches on future loads
        with open(os.path.join(weights_dir, "n_vars.txt"), "w") as f:
            f.write(",".join(var) if var else "__no_variables__")
        print(f"Weights and scalers saved to: {weights_dir}")

    return model, history, scaler_X, scaler_y


# ──────────────────────────────────────────────────────────────────────────────
# 9. EVALUATION (shared prediction logic)
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_predictions(
    model, df_test, var, n_steps, n_outputs, stride,
    scaler_X, scaler_y, target_var="calibration",
    time_col="laser_datetime",
):
    """
    Internal helper: scale, window, predict, and align predictions on the
    time axis. Overlapping predictions at the same time index are averaged.

    Args:
        model     : trained Keras model.
        df_test   : test DataFrame with the same columns used during training.
        var       : list of predictor column names.
        n_steps   : look-back window size.
        n_outputs : forecast horizon.
        stride    : step between windows.
        scaler_X  : fitted scaler for input features (or None).
        scaler_y  : fitted scaler for the target column (or None).
        target_var: name of the target column.
        time_col  : column used for temporal sorting.

    Returns:
        Tuple (y_pred, y_true, valid_time) on the original (unscaled) scale.
    """
    df_test   = df_test.sort_values(time_col).copy()
    data_test = df_test[var + [target_var]].copy()
    data_test["target"] = data_test[target_var].copy()

    # Apply training scalers (transform only, never re-fit on test data)
    if scaler_X is not None:
        data_test.loc[:, var + [target_var]] = scaler_X.transform(
            data_test[var + [target_var]]
        )
    if scaler_y is not None:
        data_test.loc[:, "target"] = scaler_y.transform(data_test[["target"]])

    test_array = data_test.values
    if len(test_array) < n_steps + n_outputs:
        raise ValueError("Test set is too short to generate any windows.")

    X_test, _ = split_sequences(test_array, n_steps, n_outputs, stride)

    if X_test.size == 0:
        raise ValueError("No test windows were generated (X_test is empty).")

    yhat = model.predict(X_test, verbose=0)   # (n_windows, n_outputs)
    if yhat.ndim == 1:
        yhat = yhat.reshape(-1, 1)

    # ── Temporal alignment: average overlapping predictions ──────────────────
    # Each time step may be covered by multiple overlapping forecast windows.
    # Collect all predictions per time index and average them.
    max_idx   = len(test_array)
    pred_list = [[] for _ in range(max_idx)]
    for i in range(len(yhat)):
        for j in range(n_outputs):
            idx = i * stride + n_steps + j
            if idx < max_idx:
                pred_list[idx].append(yhat[i, j])

    y_pred, y_true, valid_time = [], [], []
    for i in range(n_steps, max_idx):
        if pred_list[i]:
            y_pred.append(np.mean(pred_list[i]))
            y_true.append(data_test["target"].values[i])
            valid_time.append(df_test[time_col].values[i])

    if not y_pred:
        raise ValueError("No valid predictions after reconstructing overlapping windows.")

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Inverse-transform both arrays back to the original calibration scale
    if scaler_y is not None:
        y_true = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    return y_pred, y_true, np.array(valid_time)


def evaluate_model(
    model, history, df_test, var, n_steps, n_outputs=1, stride=1,
    scaler_X=None, scaler_y=None,
    target_var="calibration", time_col="laser_datetime",
    metrics: list = None,
):
    """
    Evaluate the model on the test set and return numeric metrics. No plots.

    Args:
        model     : trained Keras model.
        history   : Keras History object returned by model.fit().
        df_test   : test DataFrame.
        var       : list of predictor column names.
        n_steps   : look-back window size.
        n_outputs : forecast horizon.
        stride    : step between windows.
        scaler_X  : fitted input scaler (or None).
        scaler_y  : fitted target scaler (or None).
        target_var: name of the target column.
        time_col  : column used for temporal sorting.
        metrics   : list of metrics to compute (see _compute_metrics).
                    None → only "mape" (backward-compatible default).

    Returns:
        Tuple (mape, y_true, y_pred, metrics_dict).
    """
    y_pred, y_true, _ = _prepare_predictions(
        model, df_test, var, n_steps, n_outputs, stride,
        scaler_X, scaler_y, target_var, time_col,
    )

    _metrics   = metrics if metrics is not None else ["mape"]
    results    = _compute_metrics(y_true, y_pred, _metrics)

    for name, value in results.items():
        print(f"  {name.upper():6s}: {value:.4f}")

    mape = results.get("mape", 100 * mean_absolute_percentage_error(y_true, y_pred))
    return mape, y_true, y_pred, results


def evaluate_and_plot_model(
    model, history, df_test, var, n_steps, n_outputs=1, stride=1,
    scaler_X=None, scaler_y=None, crystal_id=None,
    target_var="calibration", plot_ratio=True,
    results_dir=None, time_col="laser_datetime",
    metrics: list = None,
    loss: str = None,
):
    """
    Evaluate the model, compute metrics, and generate diagnostic plots:
        Figure 1 — Training vs validation loss curve.
        Figure 2 — Predicted vs actual values (with optional ratio subplot).

    Args:
        model       : trained Keras model.
        history     : Keras History object returned by model.fit().
        df_test     : test DataFrame.
        var         : list of predictor column names.
        n_steps     : look-back window size.
        n_outputs   : forecast horizon.
        stride      : step between windows.
        scaler_X    : fitted input scaler (or None).
        scaler_y    : fitted target scaler (or None).
        crystal_id  : crystal identifier for plot titles and file names.
        target_var  : name of the target column.
        plot_ratio  : if True, adds a lower subplot showing the true/pred ratio.
        results_dir : directory to save figures (None = do not save).
        time_col    : column used for temporal sorting.
        metrics     : list of metrics to compute (see _compute_metrics).
                      None → ["mape", "smape", "mae", "rmse", "maxae", "r2"].
        loss        : loss function name used during training; only used to label
                      the Y-axis of the loss curve ("mse" | "mae" | "huber" |
                      "logcosh" | None → "mse").

    Returns:
        Tuple (mape, y_true, y_pred).
    """
    y_pred, y_true, valid_time = _prepare_predictions(
        model, df_test, var, n_steps, n_outputs, stride,
        scaler_X, scaler_y, target_var, time_col,
    )

    _metrics   = metrics if metrics is not None else ["mape", "smape", "mae", "rmse", "maxae", "r2"]
    results    = _compute_metrics(y_true, y_pred, _metrics)
    # Build a compact metric string for the plot title
    metrics_str = "  |  ".join(
        f"{name.upper()}: {value:.4f}" for name, value in results.items()
    )
    mape = results.get("mape", 100 * mean_absolute_percentage_error(y_true, y_pred))

    # ── Figure 1: loss curve ──────────────────────────────────────────────────
    # history.history is the dict produced by model.fit(); loss_label reflects
    # the actual loss function used during training.
    loss_label = (loss or "mse").upper()
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    ax_loss.plot(history.history["loss"],     label="Train")
    ax_loss.plot(history.history["val_loss"], label="Validation")
    ax_loss.set_title(f"Training vs validation loss (n_steps={n_steps})")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel(f"Loss ({loss_label})")
    ax_loss.legend()
    plt.tight_layout()
    if results_dir and crystal_id is not None:
        fig_loss.savefig(
            os.path.join(results_dir, f"loss_crystal_{crystal_id}_{n_steps}.png"),
            dpi=300,
        )
    plt.show()

    # ── Figure 2: predicted vs actual (+ optional ratio subplot) ─────────────
    crystal_str = "" if crystal_id is None else f" — crystal {crystal_id}"
    title = (
        f"Actual vs predicted — horizon {n_steps}"
        f"{crystal_str}\n{metrics_str}"
    )

    if plot_ratio:
        # Ratio subplot helps visualize systematic over- or under-prediction
        ratio = y_true / y_pred
        fig, axes = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        axes[0].plot(valid_time, y_true, "-", color="blue",   label="Actual")
        axes[0].plot(valid_time, y_pred, "-", color="orange", label="Predicted")
        axes[0].set_ylabel("Calibration")
        axes[0].set_title(title)
        axes[0].legend(fontsize=12)

        axes[1].plot(valid_time, ratio, "-", color="black")
        axes[1].axhline(1.0, color="red", linestyle="--", linewidth=1.5)  # Perfect ratio line
        axes[1].set_ylabel("True / Pred")
        axes[1].set_xlabel("Time")
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(valid_time, y_true, "-", color="blue",   label="Actual")
        ax.plot(valid_time, y_pred, "-", color="orange", label="Predicted")
        ax.set_ylabel("Calibration")
        ax.set_xlabel("Time")
        ax.set_title(title)
        ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()
    if results_dir:
        filename = (
            "prediction_vs_actual.png"
            if crystal_id is None
            else f"prediction_vs_actual_crystal_{crystal_id}_{n_steps}.png"
        )
        fig.savefig(os.path.join(results_dir, filename), dpi=300)
    plt.show()

    return mape, y_true, y_pred


# ──────────────────────────────────────────────────────────────────────────────
# 10. RESULTS ACCUMULATION AND REPORTING (grid search)
# ──────────────────────────────────────────────────────────────────────────────

# Global accumulators: keyed by configuration tuple, updated after each grid run
accumulated_metrics_global = defaultdict(lambda: defaultdict(float))
config_details_global   = {}


def _config_key(r: dict) -> tuple:
    """
    Generate a hashable key that uniquely identifies a model configuration.

    All hyperparameters that the grid can vary are included to avoid collisions:
    two runs differing only in learning_rate, loss, or optimizer would otherwise
    accumulate into the same entry.

    Args:
        r: Result dictionary produced by a single grid run.

    Returns:
        Tuple used as dictionary key in the global accumulators.
    """
    return (
        tuple(r["var"]),
        tuple(r["rnn_units"]),
        r["ext_norm_method"],
        r["internal_norm"],
        r.get("dropout", 0.2),
        r.get("optimizer",     "adam")     or "adam",
        r.get("learning_rate", 1e-3),      # included: the grid can vary this
        r.get("loss",          "mse")      or "mse",
        r.get("monitor",       "val_loss"),
        r.get("min_delta",     0.0)        or 0.0,
        r.get("shuffle",       False),
    )


def accumulate_results(results_list: list) -> None:
    """
    Accumulate multi-metric values across grid runs into the global dictionaries.

    Each call appends the horizon-level results to config_details_global
    and adds to the cumulative totals in accumulated_metrics_global.

    Args:
        results_list: List of result dictionaries, one per completed run.
    """
    for r in results_list:
        key = _config_key(r)
        if key not in config_details_global:
            config_details_global[key] = {
                "n_steps": [], "n_outputs": [], "stride": [],
                "metrics": defaultdict(list),
            }
        config_details_global[key]["n_steps"]  .append(r["n_steps"])
        config_details_global[key]["n_outputs"].append(r["n_outputs"])
        config_details_global[key]["stride"]   .append(r["stride"])

        for metric_name, value in r["evaluated_metrics"].items():
            accumulated_metrics_global[key][metric_name] += value
            config_details_global[key]["metrics"][metric_name].append(value)


# Metrics where a higher value is better (used for reference comparison)
METRIC_HIGHER_IS_BETTER = {"r2"}


def print_results_txt(reference_metrics: dict, output_file: str) -> None:
    """
    Write accumulated grid search results to a text file.

    For each configuration, reports cumulative and per-horizon metric values,
    and flags which horizons outperform the provided reference baselines.

    Args:
        reference_metrics : Dict of {metric_name: {horizon: baseline_value}}.
                            Obtained from cfg['grid_fixed']['reference_metrics'].
        output_file       : Path to the output text file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for key, totals in accumulated_metrics_global.items():
            (var, rnn_units, ext_norm_method, internal_norm,
              dropout_k, optimizer_k, learning_rate_k, loss_k,
              monitor_k, min_delta_k, shuffle_k) = key
            det = config_details_global[key]

            f.write("Configuration:\n")
            f.write(f"  Variables:   {var}\n")
            f.write(f"  rnn_units:   {rnn_units}  |  dropout: {dropout_k}\n")
            f.write(f"  norm:        {ext_norm_method}  |  batchnorm: {internal_norm}\n")
            f.write(f"  optimizer:   {optimizer_k}  |  lr: {learning_rate_k}  |  loss: {loss_k}\n")
            f.write(f"  monitor:     {monitor_k}  |  min_delta: {min_delta_k}  |  shuffle: {shuffle_k}\n")

            for metric_name in list(det["metrics"].keys()):
                cumulative = totals.get(metric_name, 0.0)
                f.write(f"  {metric_name.upper()} cumulative: {cumulative:.4f}\n")
                f.write("  Per-horizon detail:\n")

                values_list = det["metrics"][metric_name]
                better_horizons = []
                for ns, no, st, val in zip(det["n_steps"], det["n_outputs"],
                                           det["stride"], values_list):
                    f.write(f"    (n_steps={ns}, n_outputs={no}, stride={st})"
                            f" → {metric_name.upper()}: {val:.4f}\n")
                    ref_dict = reference_metrics.get(metric_name, {})
                    # Check if non-overlapping horizon outperforms the baseline
                    if ns == no == st and ns in ref_dict:
                        is_better = (
                            val > ref_dict[ns] if metric_name in METRIC_HIGHER_IS_BETTER
                            else val < ref_dict[ns]
                        )
                        if is_better:
                            better_horizons.append(ns)
                f.write(f"  Horizons beating reference: {better_horizons}\n")

            f.write("-" * 80 + "\n\n")


# ──────────────────────────────────────────────────────────────────────────────
# 11. MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

def run_grid_search(df_train, df_test, cfg=CONFIG, output_file=None):
    """
    Train one model per forecast horizon in cfg['grid_fixed']['horizons']
    and for every combination in the Cartesian product of cfg['grid'].

    Results are accumulated after each run and written to a text report.

    Args:
        df_train : Training DataFrame with engineered features.
        df_test  : Test DataFrame with engineered features.
        cfg      : Global configuration dictionary (see CONFIG).
        output_file : Path to the output results file. Defaults to
                   cfg['results_dir'] / cfg['results_file'].
    """
    output_file = output_file or os.path.join(cfg["results_dir"], cfg["results_file"])
    g          = cfg["grid"]
    gf         = cfg["grid_fixed"]
    horizons_list = gf["horizons"]

    # Build the Cartesian product of all grid parameters
    combinations = list(product(
        g["rnn_units"],
        g["variables"],
        g["norm_method"],
        g["internal_norm"],
        g.get("dropout",      [0.2]),
        g["shuffle"],
        g.get("optimizer",    [None]),
        g["learning_rate"],
        g.get("loss",         [None]),
        g.get("monitor",      ["val_loss"]),
        g.get("min_delta",    [None]),
    ))

    for a in horizons_list:
        # batch_size: None → dynamic based on horizon size
        batch = gf["batch_size"]
        if batch is None:
            batch = 32 if a >= 48 else 128

        for i, (rnn_units, cur_var, norm_key, batchnorm,
                dropout, shuffle_flag, optimizer_name, learning_rate, loss_name, monitor_metric, min_delta_val) in enumerate(combinations):
            try:
                reset_environment(seed=cfg["seed"], use_cpu=gf.get("use_cpu", False))

                model, history, scaler_X, scaler_y = train_model(
                    df_train=df_train, var=cur_var,
                    target_var=cfg["target_var"],
                    n_steps=a, n_outputs=a, stride=a,
                    rnn_units=rnn_units, dropout=dropout,
                    epochs=gf["epochs"], batch_size=batch,
                    optimizer=optimizer_name, learning_rate=learning_rate,
                    loss=loss_name,
                    ext_norm_method=norm_key, internal_norm=batchnorm,
                    shuffle=shuffle_flag,
                    val_split_ratio=gf["val_split"],
                    patience=gf["patience"],
                    monitor=monitor_metric,
                    min_delta=min_delta_val,
                    save_weights=gf["save_weights"],
                    results_dir=cfg["results_dir"],
                    xtal_id=cfg["xtal_id"],
                    time_col=cfg["time_col"],
                )

                mape, _, _, results_dict = evaluate_model(
                    model, history, df_test, cur_var,
                    n_steps=a, n_outputs=a, stride=a,
                    scaler_X=scaler_X, scaler_y=scaler_y,
                    time_col=cfg["time_col"],
                    metrics=gf.get("test_metrics", ["mape"]),
                )

                accumulate_results([{
                    "var": cur_var, "n_steps": a,
                    "n_outputs": a, "stride": a,
                    "rnn_units": rnn_units, "batch_size": batch,
                    "ext_norm_method": norm_key, "internal_norm": batchnorm,
                    "dropout": dropout,
                    "optimizer": optimizer_name,  "learning_rate": learning_rate,
                    "loss": loss_name, "monitor": monitor_metric,
                    "min_delta": min_delta_val if min_delta_val is not None else 0.0,
                    "shuffle": shuffle_flag, "evaluated_metrics": results_dict,
                }])
                print_results_txt(gf.get("reference_metrics", {}), output_file)

            except Exception:
                print(f"[ERROR] n_steps={a} | rnn={rnn_units}:")
                traceback.print_exc()
            finally:
                # Release GPU memory and Python objects between runs
                tf.keras.backend.clear_session()
                gc.collect()


def run_single(df_train, df_test, cfg=CONFIG):
    """
    Train and evaluate a single configuration (defined in CONFIG['default']).

    Args:
        df_train : Training DataFrame with engineered features.
        df_test  : Test DataFrame with engineered features.
        cfg      : Global configuration dictionary (see CONFIG).
    """
    d = cfg["default"]
    reset_environment(seed=cfg["seed"], use_cpu=d.get("use_cpu", False))

    # batch_size: None → dynamic (32 if n_steps >= 48, 128 if n_steps < 48)
    batch = d["batch_size"]
    if batch is None:
        batch = 32 if d["n_steps"] >= 48 else 128

    model, history, scaler_X, scaler_y = train_model(
        df_train=df_train,
        var=d["variables"],
        target_var=cfg["target_var"],
        n_steps=d["n_steps"],
        n_outputs=d["n_outputs"],
        stride=d["stride"],
        rnn_units=d["rnn_units"],
        dropout=d["dropout"],
        epochs=d["epochs"],
        batch_size=batch,
        optimizer=d.get("optimizer", None),
        learning_rate=d.get("learning_rate", 1e-3),
        loss=d.get("loss", None),
        ext_norm_method=d["norm_method"],
        internal_norm=d["internal_norm"],
        shuffle=d.get("shuffle", False),
        val_split_ratio=d["val_split"],
        patience=d["patience"],
        monitor=d.get("monitor", "val_loss"),
        min_delta=d.get("min_delta", None),
        save_weights=d.get("save_weights", False),
        results_dir=cfg["results_dir"],
        xtal_id=cfg["xtal_id"],
        time_col=cfg["time_col"],
    )

    evaluate_and_plot_model(
        model=model, history=history,
        df_test=df_test, var=d["variables"],
        n_steps=d["n_steps"], n_outputs=d["n_outputs"], stride=d["stride"],
        scaler_X=scaler_X, scaler_y=scaler_y,
        crystal_id=cfg["xtal_id"],
        plot_ratio=True,
        results_dir=cfg["results_dir"],
        time_col=cfg["time_col"],
        metrics=d.get("test_metrics", None),
        loss=d.get("loss", None),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 12. LOADING A SAVED MODEL
# ──────────────────────────────────────────────────────────────────────────────

def load_model(cfg=CONFIG):
    """
    Reconstruct a previously saved multi-output model (save_weights=True).

    All configuration is taken from cfg['default']. Loads from:
        <results_dir>/<xtal_id>_multi_<n_steps>_<n_outputs>_<norm>/model_<n_steps>steps.weights.h5
        <results_dir>/<xtal_id>_multi_<n_steps>_<n_outputs>_<norm>/scaler_X_<n_steps>steps.pkl
        <results_dir>/<xtal_id>_multi_<n_steps>_<n_outputs>_<norm>/scaler_y_<n_steps>steps.pkl
        <results_dir>/<xtal_id>_multi_<n_steps>_<n_outputs>_<norm>/n_vars.txt

    The variable list stored in n_vars.txt is checked against CONFIG to prevent
    silent mismatches when the configuration has changed since training.

    Args:
        cfg: Global configuration dictionary (see CONFIG).

    Returns:
        Tuple (model, scaler_X, scaler_y) ready for evaluation.

    Raises:
        FileNotFoundError : If n_vars.txt is missing (old or incompatible model).
        ValueError        : If the saved variable list differs from CONFIG.
    """
    d         = cfg["default"]
    n_steps   = d["n_steps"]
    n_outputs = d["n_outputs"]
    xtal_str  = str(cfg["xtal_id"])
    norm_str  = d["norm_method"].lower()
    weights_dir = os.path.join(
        cfg["results_dir"], f"{xtal_str}_multi_{n_steps}_{n_outputs}_{norm_str}"
    )

    # ── Step 1: Validate saved variables against current CONFIG ───────────────
    n_vars_path = os.path.join(weights_dir, "n_vars.txt")
    if not os.path.exists(n_vars_path):
        raise FileNotFoundError(
            f"n_vars.txt not found in {weights_dir}. "
            f"The model may be outdated or incompatible."
        )
    with open(n_vars_path, "r") as f:
        raw = f.read().strip()
    training_vars = ([] if raw == "__no_variables__"
                          else [v.strip() for v in raw.split(",")])

    if training_vars != list(d["variables"]):
        raise ValueError(
            f"Variable mismatch:\n"
            f"  Saved model: {training_vars}\n"
            f"  Current CONFIG: {list(d['variables'])}\n"
            f"Cannot load these weights without retraining."
        )

    # ── Step 2: Rebuild the model architecture ────────────────────────────────
    n_enc_features = len(d["variables"]) + 1   # predictor vars + calibration column
    input_shape    = (n_steps, n_enc_features)
    model = build_model(
        input_shape,
        n_outputs=d["n_outputs"],
        rnn_units=d["rnn_units"],
        dropout=d["dropout"],
        use_batchnorm=d["internal_norm"],
    )

    # Dummy forward pass to initialize layer weights before loading saved values
    dummy_X = np.zeros((1, n_steps, n_enc_features), dtype=np.float32)
    model.predict(dummy_X, verbose=0)

    # ── Step 3: Load weights and scalers ─────────────────────────────────────
    model.load_weights(
        os.path.join(weights_dir, f"model_{n_steps}steps.weights.h5")
    )
    scaler_X = joblib.load(os.path.join(weights_dir, f"scaler_X_{n_steps}steps.pkl"))
    scaler_y = joblib.load(os.path.join(weights_dir, f"scaler_y_{n_steps}steps.pkl"))

    print(f"Model ({n_steps} steps) loaded from: {weights_dir}")
    return model, scaler_X, scaler_y


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Option A: train and evaluate a single configuration ───────────────────
    df_train, df_test = prepare_splits(CONFIG)
    run_single(df_train, df_test, CONFIG)           # ← default mode

    # ── Option B: run a full grid search ──────────────────────────────────────
    #run_grid_search(df_train, df_test, CONFIG)      # ← uncomment for grid search

    # ── Option C: load a previously saved model and evaluate ──────────────────
    # Requires a prior run with save_weights=True in CONFIG['default'].
    # Expected folder example: <results_dir>/30600_multi_96_96_minmax/
'''
    _, df_test = prepare_splits(CONFIG)
    model, scaler_X, scaler_y = load_model(cfg=CONFIG)
    _d = CONFIG["default"]
    evaluate_and_plot_model(
        model=model, history=type("H", (), {"history": {"loss": [], "val_loss": []}})(),
        df_test=df_test, var=_d["variables"],
        n_steps=_d["n_steps"], n_outputs=_d["n_outputs"], stride=_d["stride"],
        scaler_X=scaler_X, scaler_y=scaler_y,
        crystal_id=CONFIG["xtal_id"],
        plot_ratio=True,
        results_dir=CONFIG["results_dir"],
        time_col=CONFIG["time_col"],
        loss=_d.get("loss", None),
        metrics=_d.get("test_metrics", None),
    )'''