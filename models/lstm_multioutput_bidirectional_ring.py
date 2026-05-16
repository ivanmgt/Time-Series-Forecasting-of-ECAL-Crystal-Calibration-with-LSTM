# =============================================================================
# lstm_multioutput_bidirectional_ring.py
# Multi-output LSTM bidirectional model WITHOUT Embedding for multi-crystal forecasting
# across multiple rings. Each crystal is normalized independently.
# Adaptable to a different set of rings by modifying CONFIG.
#
# NOTE: Due to the massive data volume and architecture type, this model
# only runs efficiently on professional GPUs (e.g., NVIDIA A5000 or L40S).
#
# KEY DIFFERENCES compared to lstm_multioutput.py (single-crystal):
#   - Loads all crystals from an active ring simultaneously.
#   - Each crystal has its own independent scaler (scalers_X / scalers_y dicts).
#   - split_sequences_by_xtal generates windows per crystal with per-crystal scaling.
#   - Evaluation includes both individual-crystal metrics and ring-level aggregates
#     (wmape_pond, mae_global, rmse_global).
#   - Grid search uses a "config inheritance" system (grid_configs list) instead
#     of a Cartesian product: each entry overrides only the parameters it specifies,
#     inheriting the rest from CONFIG["default"].
#   - _build_loss returns a string (not an instance) because model.compile()
#     accepts string names directly, unlike the GradientTape loop in seq2seq variants.
# =============================================================================

# ──────────────────────────────────────────────────────────────────────────────
#  0. CENTRAL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Paths (add/remove rings as needed) ───────────────────────────────────
    "data_sources": {
        "ring_1":   "/data/plus_z_1/ring_1.csv",
        # "ring_m1":  "/data/minus_z_1/ring_-1.csv",
        # "ring_50":  "/data/plus_z_6/ring_50.csv",
        # "ring_m36": "/data/minus_z_4/ring_-36.csv",
    },
    "active_ring":  "ring_1",          # Which of the above rings to load
    "results_dir":  "/results",
    "results_file": "results_multi_bi_ring.txt",

    # ── Data filtering ───────────────────────────────────────────────────────
    "target_var":   "calibration",    # Target column to predict
    "time_col":     "laser_datetime", # Column used for temporal ordering
    "calib_min":    0.7,              # Minimum accepted target value
    "calib_max":    1.0,              # Maximum accepted target value
    # train_year as a list to support multi-year training
    "train_year":   [2016],
    "test_year":    2017,             # Year used for external evaluation

    # ── Columns to retain after loading ──────────────────────────────────────
    "keep_cols": ["xtal_id", "calibration", "int_deliv_inv_ub",
                  "laser_datetime", "time"],

    # ── Reproducibility ──────────────────────────────────────────────────────
    "seed": 1234,

    # =========================================================================
    #  DEFAULT CONFIGURATION (run_single)
    # =========================================================================
    "default": {
        # ── Predictor variables ───────────────────────────────────────────────
        "variables":      ["int_deliv_inv_ub", "dint_dtime"],  # var1
        # n_steps == n_outputs == stride for non-overlapping windows
        "n_steps":        24,
        "n_outputs":      24,
        "stride":         24,

        # ── Architecture ─────────────────────────────────────────────────────
        "rnn_units":      [128, 128],
        "internal_norm":  False,  # Apply BatchNormalization after each LSTM layer
        "dropout":        0.2,    # Dropout rate applied after each LSTM layer

        # ── Training ─────────────────────────────────────────────────────────
        "epochs":         1,
        # batch_size: None → dynamic (32 if n_steps >= 48, 128 if n_steps < 48)
        "batch_size":     None,

        # ── Optimizer ────────────────────────────────────────────────────────
        # Options: "adam" | "adamw" | "sgd" | "rmsprop" | "adagrad"
        # None → use "adam"
        "optimizer":      None,
        # Optimizer learning rate
        "learning_rate":  1e-3,

        # ── Loss function ─────────────────────────────────────────────────────
        # Options: "mse" | "mae" | "huber" | "logcosh"
        # None → use "mse"
        # mse     : strongly penalizes large errors; standard choice for regression.
        # mae     : more robust to outliers; converges more slowly.
        # huber   : hybrid mse/mae loss; mse for small errors, mae for large ones.
        # logcosh : similar to Huber but differentiable over the entire domain.
        "loss":           None,

        # ── Early stopping ────────────────────────────────────────────────────
        "patience":       20,    # Number of epochs without improvement before stopping
        # Training history metric to monitor: "val_loss" or "loss"
        "monitor":        "val_loss",
        # Minimum improvement required to reset the patience counter.
        # None → 0.0 (any improvement counts, with no minimum threshold)
        "min_delta":      None,

        # ── Data normalization ────────────────────────────────────────────────
        # Options: "MinMax" | "Standard" | "PowerTransformer" | "None"
        "norm_method":    "MinMax",

        # ── Train/validation split ────────────────────────────────────────────
        "val_split":      0.1,   # Fraction of data used for validation (0–1)

        # ── Shuffle windows before training ───────────────────────────────────
        # True  → shuffles windows before the train/val split
        # False → preserves temporal order (recommended for time series)
        "shuffle":        False,

        # ── Hardware ──────────────────────────────────────────────────────────
        # True  → force CPU usage (reproducible, slower)
        # False → use GPU if available
        "use_cpu":        False,

        # ── Individual crystal metrics (run_single → evaluate_and_plot_model) ─
        # Computed on the original scale for the crystal selected via plot_xtal_id.
        # Only used in run_single; the grid evaluates ring-level metrics, not per crystal.
        # Available: "mape" | "smape" | "mae" | "rmse" | "maxae" | "r2"
        # - mape  : Mean Absolute Percentage Error (×100). Main project metric.
        # - smape : Symmetric MAPE; more robust when the target approaches 0.
        # - mae   : Mean Absolute Error; expressed in the same units as the target.
        # - rmse  : Root Mean Squared Error; penalizes large errors.
        # - maxae : Maximum absolute error; indicates the worst individual case.
        # - r2    : Coefficient of determination (1=perfect, 0=mean baseline, <0=worse).
        #
        # Interpretation guide for calibration series (range ~0.7–1.0):
        #   mape  : mean percentage error; directly comparable across horizons.
        #   smape : more stable than mape when true values approach zero.
        #   mae   : error in calibration units; easy to interpret physically.
        #   rmse  : like mae but penalizes error peaks; useful for detecting failures.
        #   maxae : worst-case point error; relevant for reconstruction guarantees.
        #   r2    : fraction of variance explained; 1=perfect, 0=mean baseline, <0=worse.
        "test_metrics":   ["mape", "smape", "mae", "rmse", "maxae", "r2"],

        # ── Ring-level metrics (run_single → evaluate_all_xtals) ─────────────
        # Computed by aggregating predictions across ALL crystals in the ring.
        # Unlike test_metrics (single crystal), these measure the model's global
        # behavior over the entire ring.
        # Available: "wmape_pond" | "mae_global" | "rmse_global"
        # - wmape_pond  : True per-crystal WMAPE, weighted by number of observations.
        #                 More robust than the average of individual MAPEs when
        #                 true values are small.
        # - mae_global  : MAE over all concatenated crystal predictions.
        # - rmse_global : RMSE over all concatenated crystal predictions.
        "ring_metrics":   ["wmape_pond", "mae_global", "rmse_global"],

        # ── Saving ───────────────────────────────────────────────────────────
        "save_weights":   True,
        "plot_xtal_id":   30600,  # Crystal ID used for single-crystal plots in run_single
    },

    # =========================================================================
    # FIXED GRID PARAMETERS (not combined, applied to all runs)
    # =========================================================================
    "grid_fixed": {
        # ── Training epochs ───────────────────────────────────────────────────
        "epochs":         1,
        # ── Forecast horizons to evaluate ─────────────────────────────────────
        "horizons":       [1, 12, 24, 36, 48, 60, 72, 84, 96],
        # ── Ring-level metrics (fixed for the entire grid) ────────────────────
        # Available: "wmape_pond" | "mae_global" | "rmse_global"
        "ring_metrics":   ["wmape_pond", "mae_global", "rmse_global"],
        # ── Reference metric values per horizon (baseline to outperform) ───────
        # Dict of dicts: {metric_name: {horizon: reference_value}}
        # wmape_pond, mae_global, rmse_global → outperformed if value < reference[horizon]
        "reference_metrics": {
            "wmape_pond": {
                1: 0.075, 12: 0.209, 24: 0.246, 36: 0.278,
                48: 0.319, 60: 0.345, 72: 0.377, 84: 0.402, 96: 0.427,
            },
            "mae_global": {
                # Replace with actual historical reference values
                1: 1.0, 12: 1.0, 24: 1.0, 36: 1.0,
                48: 1.0, 60: 1.0, 72: 1.0, 84: 1.0, 96: 1.0,
            },
            "rmse_global": {
                # Replace with actual historical reference values
                1: 1.0, 12: 1.0, 24: 1.0, 36: 1.0,
                48: 1.0, 60: 1.0, 72: 1.0, 84: 1.0, 96: 1.0,
            },
        },
        # batch_size: None → dynamic (32 if n_steps >= 48, 128 if n_steps < 48)
        "batch_size":     None,
        # ── Fixed training parameters for the grid ────────────────────────────
        "val_split":      0.1,
        "patience":       20,
        # ── Saving ───────────────────────────────────────────────────────────
        # True  → save model weights to results_dir at each horizon.
        # False → train without saving (faster for grid exploration).
        "save_weights":   False,
        # ── Hardware ──────────────────────────────────────────────────────────
        "use_cpu":        False,
    },

    # =========================================================================
    # GRID CONFIGURATIONS (inheritance-style)
    # Each entry is a dict with the specific parameters for that run.
    # _run_one_config iterates each entry over all horizons in grid_fixed.
    # Unspecified fields inherit their value from CONFIG["default"].
    # =========================================================================
    "grid_configs": [
        # ── config 11 ─────────────────────────────────────────────────────────
        {
            "id":           11,
            "rnn_units":    [512, 512],
            "variables":    "var6",   # Key from VARIABLE_SETS
            "shuffle":      True,
            # Optional fields below; if omitted, inherited from default:
            # "dropout":       0.2,
            # "learning_rate": 1e-3,
            # "optimizer":    None,   # None → "adam"
            # "loss":         None,   # None → "mse"
            # "norm_method":  "MinMax",
            # "internal_norm": False,
            # "monitor":      "val_loss",
            # "min_delta":    None,
        },
        # ── config 15 ─────────────────────────────────────────────────────────
        # {"id": 15, "rnn_units": [1024, 1024], "variables": "var6", "shuffle": True},
        # ── config 10 ─────────────────────────────────────────────────────────
        # {"id": 10, "rnn_units": [512, 512],   "variables": "var5", "shuffle": True},
        # ── config 7 ──────────────────────────────────────────────────────────
        # {"id": 7,  "rnn_units": [256, 256],   "variables": "var6", "shuffle": True},
        # ── config 3 ──────────────────────────────────────────────────────────
        # {"id": 3,  "rnn_units": [128, 128],   "variables": "var6", "shuffle": True},
    ],
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os
import gc
import random
import traceback
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, LSTM, Input, Bidirectional
)
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


def reset_environment(seed: int = 1234, use_cpu: bool = False) -> None:
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
_seed = CONFIG["seed"]
_set_env_vars(_seed)
random.seed(_seed)
np.random.seed(_seed)
tf.random.set_seed(_seed)
tf.keras.utils.set_random_seed(_seed)
tf.config.experimental.enable_op_determinism()

# ── GPU memory growth (must be configured before any other TF operation) ──────
# Prevents TF from reserving all VRAM at startup → avoids OOM in long sessions.
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
    Load the active ring CSV, filter by calibration range, parse datetime
    columns, and sort by (xtal_id, time_col).

    Args:
        cfg: Global configuration dictionary (see CONFIG).

    Returns:
        Filtered and sorted DataFrame containing all crystals of the active ring.
    """
    path = cfg["data_sources"][cfg["active_ring"]]
    df   = pd.read_csv(path)[cfg["keep_cols"]].copy()
    df   = df[(df[cfg["target_var"]] >= cfg["calib_min"]) &
              (df[cfg["target_var"]] <= cfg["calib_max"])]
    # Detect date columns dynamically (avoids hardcoding column names)
    date_cols = [c for c in ["time", "laser_datetime"] if c in df.columns]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S")
    return df.sort_values(["xtal_id", cfg["time_col"]]).reset_index(drop=True)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features PER CRYSTAL within the DataFrame:
      dint_dtime, delta_lumi, delta_t, days_no_int.

    diff() is applied independently for each xtal_id to prevent cross-crystal
    contamination. Crystals with fewer than 2 rows are skipped with a warning.

    Args:
        df: DataFrame with all crystals of a ring, sorted by (xtal_id, time).

    Returns:
        Concatenated DataFrame with engineered features for all valid crystals.
    """
    fragments = []
    for xtal in df["xtal_id"].unique():
        i = df[df["xtal_id"] == xtal].copy().reset_index(drop=True)

        # Guard: a crystal with a single row cannot compute diff() at index 1
        if len(i) < 2:
            print(f"[WARNING] Crystal {xtal} has fewer than 2 rows. Skipped.")
            continue

        dt_sec = i["time"].diff().dt.total_seconds()
        d_lum  = i["int_deliv_inv_ub"].diff()

        i["dint_dtime"]   = d_lum / dt_sec         # Luminosity rate of change
        i["delta_lumi"]   = d_lum                  # Absolute luminosity increment
        i["delta_t"]      = i["laser_datetime"].diff().dt.total_seconds()
        i["days_no_int"]  = i["time"].diff().dt.days

        # ── Edge case and outlier corrections ─────────────────────────────────
        i.loc[0, "dint_dtime"] = i.loc[1, "dint_dtime"]
        i.loc[i["days_no_int"] > 0,  "dint_dtime"]               = 0
        i.loc[0, ["days_no_int", "delta_t", "delta_lumi"]]       = 0
        i.loc[i["days_no_int"] > 30, ["delta_lumi", "delta_t"]]  = 0
        i["dint_dtime"] = i["dint_dtime"].bfill().ffill()
        i.loc[~np.isfinite(i["dint_dtime"]), "dint_dtime"]        = 0

        fragments.append(i)

    return pd.concat(fragments, ignore_index=True)


def prepare_splits(cfg: dict = CONFIG):
    """
    Load data, apply feature engineering per year, and return
    (df_train, df_test) containing all crystals of the active ring.

    Feature engineering is applied year-by-year so that diff()-based
    computations reset at each period boundary.
    Accepts train_year as either a scalar or a list.

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

    train_years = (cfg["train_year"] if isinstance(cfg["train_year"], list)
                   else [cfg["train_year"]])

    df_train = feature_engineering(
        df[yr.isin(train_years)].copy().reset_index(drop=True)
    )
    df_test  = feature_engineering(
        df[yr == cfg["test_year"]].copy().reset_index(drop=True)
    )

    if len(df_train) == 0:
        raise ValueError(
            f"df_train is empty. Check that 'train_year'={cfg['train_year']} "
            f"exists in the CSV and that the filters calib_min/calib_max "
            f"({cfg['calib_min']}–{cfg['calib_max']}) do not discard all rows."
        )
    if len(df_test) == 0:
        raise ValueError(
            f"df_test is empty. Check that 'test_year'={cfg['test_year']} "
            f"exists in the CSV and that the filters calib_min/calib_max "
            f"({cfg['calib_min']}–{cfg['calib_max']}) do not discard all rows."
        )

    return df_train, df_test


# ──────────────────────────────────────────────────────────────────────────────
# 4. VARIABLE GROUPS (adaptable)
# ──────────────────────────────────────────────────────────────────────────────
# Pre-defined input feature sets. Referenced by key in grid_configs entries
# (e.g., "variables": "var6") and resolved via VARIABLE_SETS in _run_one_config.
VARIABLE_SETS = {
    "var1": ["int_deliv_inv_ub", "dint_dtime"],
    "var2": ["int_deliv_inv_ub", "dint_dtime", "delta_lumi"],
    "var3": ["int_deliv_inv_ub", "dint_dtime", "delta_t"],
    "var4": ["int_deliv_inv_ub"],
    "var5": ["dint_dtime"],
    "var6": ["delta_lumi"],
}


# ──────────────────────────────────────────────────────────────────────────────
# 5. WINDOWING
# ──────────────────────────────────────────────────────────────────────────────

def split_sequences(
    sequences: np.ndarray,
    n_steps: int,
    n_outputs: int = 1,
    stride: int = 1,
) -> tuple:
    """
    Simple sliding-window function for a single crystal.
    The target is assumed to be the last column.

    Args:
        sequences : 2-D array of shape (T, n_cols).
        n_steps   : look-back window size.
        n_outputs : forecast horizon (number of future steps to predict).
        stride    : step size between consecutive windows.

    Returns:
        X (N, n_steps, n_features) — input windows (all columns except the last).
        y (N, n_outputs)           — target windows (last column only).

    Raises:
        AssertionError: If sequences has fewer than 2 columns.
    """
    assert sequences.shape[1] >= 2, \
        "At least 2 columns are required (features + target)."
    X_list, y_list = [], []
    T     = sequences.shape[0]
    y_col = sequences.shape[1] - 1
    for i in range(0, T - n_steps - n_outputs + 1, stride):
        end_ix  = i + n_steps
        out_end = end_ix + n_outputs
        X_list.append(sequences[i:end_ix, :-1])
        y_list.append(sequences[end_ix:out_end, y_col])
    return np.array(X_list), np.array(y_list)


def split_sequences_by_xtal(
    df: pd.DataFrame,
    var: list,
    n_steps: int,
    target_var: str,
    n_outputs: int = 1,
    stride: int = 1,
    norm_method: str = "MinMax",
) -> tuple:
    """
    Generate sliding windows for all crystals with independent per-crystal scaling.

    Each crystal is fitted with its own scaler to prevent data leakage across
    crystals with different calibration baselines.

    Args:
        df          : DataFrame with all crystals of a ring.
        var         : List of predictor column names.
        n_steps     : Look-back window size.
        target_var  : Name of the target column.
        n_outputs   : Forecast horizon.
        stride      : Step size between consecutive windows.
        norm_method : Normalization method (see _make_scaler).

    Returns:
        X          (N, n_steps, n_features) — concatenated windows across all crystals.
        y          (N, n_outputs)           — concatenated targets.
        scalers_X  dict {xtal_id: scaler_X} — one fitted input scaler per crystal.
        scalers_y  dict {xtal_id: scaler_y} — one fitted target scaler per crystal.
    """
    X_all, y_all         = [], []
    scalers_X, scalers_y = {}, {}

    for xtal in df["xtal_id"].unique():
        df_xtal           = df[df["xtal_id"] == xtal].copy()
        df_xtal["target"] = df_xtal[target_var]

        scaler_X = _make_scaler(norm_method)
        scaler_y = _make_scaler(norm_method)

        if scaler_X is not None:
            df_xtal[var + [target_var]] = scaler_X.fit_transform(
                df_xtal[var + [target_var]]
            )
        if scaler_y is not None:
            df_xtal["target"] = scaler_y.fit_transform(df_xtal[["target"]])

        scalers_X[xtal] = scaler_X
        scalers_y[xtal] = scaler_y

        # Build sequence array: [var1, ..., target_var (scaled), target (scaled)]
        # split_sequences uses the last column as y, so "target" is appended last.
        seq_array = df_xtal[var + [target_var, "target"]].values
        X_x, y_x  = split_sequences(seq_array, n_steps, n_outputs, stride)
        X_all.extend(X_x)
        y_all.extend(y_x)

    return np.array(X_all), np.array(y_all), scalers_X, scalers_y


# ──────────────────────────────────────────────────────────────────────────────
# 6. MULTI-OUTPUT LSTM MODEL (without Embedding)
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    input_shape: tuple,
    n_outputs: int,
    rnn_units: list = None,
    dropout: float = 0.2,
    use_batchnorm: bool = False,
) -> Model:
    """
    Build a multi-output LSTM bidirectional model without an Embedding layer.

    Unlike the Embedding variant, xtal_id is NOT passed as an input here.
    The model learns to generalize across all crystals using only the time-series
    features. This requires sufficient data per crystal and a large enough model.

    Args:
        input_shape  : Tuple (n_steps, n_features).
        n_outputs    : Number of future steps to predict (forecast horizon).
        rnn_units    : Units per LSTM layer. None → [128, 32, 32].
        dropout      : Dropout rate applied after each LSTM layer.
        use_batchnorm: If True, inserts BatchNormalization after each LSTM.

    Returns:
        Compiled-ready Keras Model (call model.compile() separately).
    """
    if rnn_units is None:
        rnn_units = [128, 32, 32]

    inputs = Input(shape=input_shape, name="ts_input")
    x      = inputs

    for i, units in enumerate(rnn_units):
        return_sequences = i < len(rnn_units) - 1
        x = Bidirectional(LSTM(units, return_sequences=return_sequences, name=f"LSTM_{i+1}"))(x)
        if use_batchnorm:
            x = BatchNormalization(name=f"BN_{i+1}")(x)
        x = Dropout(dropout, name=f"Dropout_{i+1}")(x)

    outputs = Dense(n_outputs, activation="linear", name="output")(x)

    return Model(inputs=inputs, outputs=outputs, name="rnn_multioutput")


# ──────────────────────────────────────────────────────────────────────────────
# 7. NORMALIZATION AND COMPILATION HELPERS
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


def _build_optimizer(name: str, learning_rate: float):
    """
    Instantiate a Keras optimizer from its string name.

    Args:
        name          : "adam" | "adamw" | "sgd" | "rmsprop" | "adagrad"
                        None   → defaults to "adam"
        learning_rate : Learning rate to apply to the optimizer.

    Returns:
        A tf.keras.optimizers instance ready for model.compile().

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
    Validate and return the loss function name as a string.

    Returns a string (not an instance) because this model uses model.compile(),
    which accepts loss names as strings directly. This differs from the seq2seq
    variants which use a GradientTape loop and require a callable loss instance.

    Args:
        name : "mse" | "mae" | "huber" | "logcosh"
               None  → defaults to "mse"

    Returns:
        Validated loss name string for use in model.compile(loss=...).

    Raises:
        ValueError: If an unrecognized loss name is provided.
    """
    name = (name or "mse").lower()
    options = {"mse", "mae", "huber", "logcosh"}
    if name not in options:
        raise ValueError(f"Invalid loss: '{name}'. Options: {options}")
    return name


# ──────────────────────────────────────────────────────────────────────────────
# 8. METRICS
# ──────────────────────────────────────────────────────────────────────────────

# Metrics where a higher value is better (used for reference comparison)
METRIC_HIGHER_IS_BETTER = {"r2"}


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     metrics: list) -> dict:
    """
    Compute evaluation metrics for a single crystal on the original scale.

    Args:
        y_true  : ground-truth values (1-D array).
        y_pred  : model predictions (1-D array).
        metrics : list of metric names to compute.
                  Available: "mape" | "smape" | "mae" | "rmse" | "maxae" | "r2"

    Returns:
        Dictionary {metric_name: value}.
    """
    results = {}
    for m in metrics:
        name = m.lower()
        if name == "mape":
            results["mape"]  = 100 * mean_absolute_percentage_error(y_true, y_pred)
        elif name == "smape":
            denom = (np.abs(y_true) + np.abs(y_pred)) / 2
            results["smape"] = float(
                np.mean(np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom)) * 100
            )
        elif name == "mae":
            results["mae"]   = mean_absolute_error(y_true, y_pred)
        elif name == "rmse":
            results["rmse"]  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        elif name == "maxae":
            results["maxae"] = float(np.max(np.abs(y_true - y_pred)))
        elif name == "r2":
            results["r2"]    = r2_score(y_true, y_pred)
        else:
            print(f"[WARNING] Unknown metric ignored: '{m}'")
    return results


def _compute_ring_metrics(all_y_true: np.ndarray, all_y_pred: np.ndarray,
                           df_wmape: "pd.DataFrame",
                           metrics: list) -> dict:
    """
    Compute global metrics aggregated over ALL crystals of the ring.

    Args:
        all_y_true : 1-D array — concatenated ground-truth values across all crystals.
        all_y_pred : 1-D array — concatenated predictions across all crystals.
        df_wmape   : DataFrame with columns {"xtal_id", "wmape", "n"}.
                     wmape_i = Σ|y_i − ŷ_i| / Σ|y_i| × 100 per crystal.
                     n_i     = number of observations for crystal i.
        metrics    : list of metric names to compute.
                     Available: "wmape_pond" | "mae_global" | "rmse_global"

    Returns:
        Dictionary containing only the requested metrics.

    Metric definitions:
        wmape_pond  : weighted WMAPE across crystals, using observation count as weight.
                      More robust than a simple average when crystals have different
                      amounts of data or when true values approach zero.
        mae_global  : mean absolute error over the full concatenated prediction array.
        rmse_global : root mean squared error over the full concatenated prediction array.
    """
    results = {}
    for m in metrics:
        name = m.lower()
        if name == "wmape_pond":
            results["wmape_pond"] = float(
                (df_wmape["wmape"] * df_wmape["n"]).sum() / df_wmape["n"].sum()
            )
        elif name == "mae_global":
            results["mae_global"] = float(np.mean(np.abs(all_y_true - all_y_pred)))
        elif name == "rmse_global":
            results["rmse_global"] = float(
                np.sqrt(np.mean((all_y_true - all_y_pred) ** 2))
            )
        else:
            print(f"[WARNING] Unknown ring metric ignored: '{m}'")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 9. TRAINING
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
    optimizer:       str   = None,      # None → "adam"; see _build_optimizer()
    learning_rate:   float = 1e-3,
    loss:            str   = None,      # None → "mse";  see _build_loss()
    ext_norm_method: str   = "MinMax",
    internal_norm:   bool  = False,
    val_split_ratio: float = 0.1,
    patience:        int   = 20,
    monitor:         str   = "val_loss",
    min_delta:       float = None,      # None → 0.0
    shuffle:         bool  = False,
    save_weights:    bool  = False,
    results_dir:     str   = None,
    ring_name:       str   = None,
    time_col:        str   = "laser_datetime",
    use_cpu:         bool  = False,
):
    """
    Train the multi-output LSTM bidirectional over all crystals in df_train, with independent
    per-crystal normalization. Uses model.fit() with Keras EarlyStopping.

    If save_weights=True, the following files are written:
        <results_dir>/<ring>_multi_bi_<n_steps>_<n_outputs>_<norm>/model_<n_steps>steps.weights.h5
        <results_dir>/<ring>_multi_bi_<n_steps>_<n_outputs>_<norm>/scaler_X_<xtal_id>.pkl
        <results_dir>/<ring>_multi_bi_<n_steps>_<n_outputs>_<norm>/scaler_y_<xtal_id>.pkl
        <results_dir>/<ring>_multi_bi_<n_steps>_<n_outputs>_<norm>/n_vars.txt

    Args:
        df_train        : Training DataFrame with all crystals and engineered features.
        var             : List of predictor column names.
        target_var      : Name of the target column to predict.
        n_steps         : Look-back window size.
        n_outputs       : Forecast horizon (number of future steps to predict).
        stride          : Step size between consecutive windows.
        rnn_units       : Units per LSTM layer. None → [128, 32, 32].
        dropout         : Dropout rate applied after each LSTM layer.
        epochs          : Maximum number of training epochs.
        batch_size      : Mini-batch size for training.
        optimizer       : Optimizer name. None → "adam".
        learning_rate   : Optimizer learning rate.
        loss            : Loss function name. None → "mse".
        ext_norm_method : External normalization method applied independently per crystal.
        internal_norm   : If True, inserts BatchNormalization after each LSTM.
        val_split_ratio : Fraction of training data used for validation.
        patience        : Early stopping patience (epochs without improvement).
        monitor         : Metric to monitor for early stopping ("val_loss" or "loss").
        min_delta       : Minimum improvement required to reset the patience counter.
        shuffle         : If True, shuffles windows before the train/val split.
        save_weights    : If True, saves model weights and scalers to disk.
        results_dir     : Directory for saving weights (required if save_weights=True).
        ring_name       : Ring identifier used in folder naming.
        time_col        : Column name used for temporal sorting.
        use_cpu         : If True, forces CPU-only execution.

    Returns:
        Tuple (model, history, scalers_X, scalers_y).
        history is a Keras History object (access via history.history["loss"]).
    """
    if rnn_units is None:
        rnn_units = [128, 32, 32]

    df_train = df_train.sort_values(["xtal_id", time_col]).copy()

    # Generate windows with independent per-crystal scaling
    X_full, y_full, scalers_X, scalers_y = split_sequences_by_xtal(
        df_train, var, n_steps, n_outputs=n_outputs,
        stride=stride, norm_method=ext_norm_method,
        target_var=target_var,
    )

    # Controlled shuffle: only if the flag is set (preserves temporal order otherwise)
    if shuffle:
        perm   = np.random.permutation(len(X_full))
        X_full = X_full[perm]
        y_full = y_full[perm]

    split    = int(len(X_full) * (1 - val_split_ratio))
    X_train, X_val = X_full[:split], X_full[split:]
    y_train, y_val = y_full[:split], y_full[split:]

    del X_full, y_full
    gc.collect()

    input_shape   = (X_train.shape[1], X_train.shape[2])
    _batch_size = batch_size if batch_size is not None else 32


    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.batch(_batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(_batch_size).prefetch(tf.data.AUTOTUNE)


    del X_train, y_train, X_val, y_val
    gc.collect()
    
    loss_str      = _build_loss(loss)
    optimizer_obj = _build_optimizer(optimizer, learning_rate)
    _min_delta    = min_delta if min_delta is not None else 0.0

    model = build_model(
        input_shape=input_shape,
        n_outputs=n_outputs,
        rnn_units=rnn_units,
        dropout=dropout,
        use_batchnorm=internal_norm,
    )
    model.compile(
        optimizer=optimizer_obj,
        loss=loss_str,
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )

    if monitor not in ("val_loss", "loss"):
        raise ValueError(f"monitor must be 'val_loss' or 'loss', got: '{monitor}'")

    early_stop = EarlyStopping(
        monitor=monitor,
        patience=patience,
        min_delta=_min_delta,
        restore_best_weights=True,
    )

    history = model.fit(
         train_ds,                      
         validation_data=val_ds,         
         epochs=epochs,
         verbose=0,
         callbacks=[early_stop])


    # ── Save weights and scalers (optional) ───────────────────────────────────
    if save_weights:
        if results_dir is None:
            raise ValueError("results_dir is required when save_weights=True.")
        ring_str    = ring_name or "ring_unknown"
        norm_str    = ext_norm_method.lower()
        weights_dir = os.path.join(
            results_dir, f"{ring_str}_multi_bi_{n_steps}_{n_outputs}_{norm_str}"
        )
        os.makedirs(weights_dir, exist_ok=True)
        model.save_weights(os.path.join(weights_dir, f"model_{n_steps}steps.weights.h5"))
        for xtal, sx in scalers_X.items():
            joblib.dump(sx, os.path.join(weights_dir, f"scaler_X_{xtal}.pkl"))
        for xtal, sy in scalers_y.items():
            joblib.dump(sy, os.path.join(weights_dir, f"scaler_y_{xtal}.pkl"))
        # Save variable list to detect mismatches on future loads
        with open(os.path.join(weights_dir, "n_vars.txt"), "w", encoding="utf-8") as f:
            f.write(",".join(var) if var else "__no_variables__")
        print(f"Weights saved to: {weights_dir}")

    return model, history, scalers_X, scalers_y


# ──────────────────────────────────────────────────────────────────────────────
# 10. EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_predictions_xtal(
    model, df_xtal, var, n_steps, n_outputs, stride,
    scaler_X, scaler_y,
    target_var="calibration", time_col="laser_datetime",
):
    """
    Internal helper: scale, window, predict, and align predictions on the
    time axis for a SINGLE crystal.

    Args:
        model     : trained multi-output LSTM model.
        df_xtal   : test DataFrame for one crystal only.
        var       : list of predictor column names.
        n_steps   : look-back window size.
        n_outputs : forecast horizon.
        stride    : step size between windows.
        scaler_X  : fitted input scaler for this crystal (or None).
        scaler_y  : fitted target scaler for this crystal (or None).
        target_var: name of the target column.
        time_col  : column used for temporal sorting.

    Returns:
        Tuple (y_pred, y_true, valid_time) on the original (unscaled) scale.

    Raises:
        ValueError: If the test segment is too short or produces no valid predictions.
    """
    df_xtal   = df_xtal.sort_values(time_col).copy()
    data_test = df_xtal[var + [target_var]].copy()
    data_test["target"] = data_test[target_var].copy()

    # Apply training scalers (transform only, never re-fit on test data)
    if scaler_X is not None:
        data_test.loc[:, var + [target_var]] = scaler_X.transform(
            data_test[var + [target_var]]
        )
    if scaler_y is not None:
        data_test.loc[:, "target"] = scaler_y.transform(data_test[["target"]])

    test_array = data_test.values

    # Guard 1: minimum length before windowing
    if len(test_array) < n_steps + n_outputs:
        raise ValueError("Test segment is too short to generate any windows.")

    X_test, _ = split_sequences(test_array, n_steps, n_outputs, stride)

    # Guard 2: empty windowing — stride > 1 may trigger this even if Guard 1 passes
    if X_test.size == 0:
        raise ValueError(
            f"No test windows were generated (X_test is empty). "
            f"Check that the test segment has enough samples for "
            f"n_steps + n_outputs = {n_steps + n_outputs} with stride = {stride}."
        )

    yhat = model.predict(X_test, verbose=0)
    if yhat.ndim == 1:
        yhat = yhat.reshape(-1, 1)

    # ── Temporal alignment: average overlapping predictions ──────────────────
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
            valid_time.append(df_xtal[time_col].values[i])

    # Guard 3: empty reconstruction after overlap averaging
    if not y_pred:
        raise ValueError(
            f"No valid predictions after reconstructing overlapping windows. "
            f"Check that n_steps={n_steps}, n_outputs={n_outputs}, and "
            f"stride={stride} are consistent with the size of the test segment."
        )

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Inverse-transform both arrays back to the original calibration scale.
    # If scaler_y is None (norm_method="None"), values are already on the
    # original scale and no inverse transform is needed.
    if scaler_y is not None:
        y_true = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    return y_pred, y_true, np.array(valid_time)


def evaluate_all_xtals(
    model, df_test, var, n_steps, n_outputs=1, stride=1,
    scalers_X=None, scalers_y=None,
    target_var="calibration", time_col="laser_datetime",
    ring_metrics: list = None,
):
    """
    Evaluate the model over ALL crystals in the ring by calling
    _prepare_predictions_xtal per crystal. No individual crystal plots are generated.

    Crystals without a matching scaler are skipped with an explicit warning.

    Args:
        model        : trained multi-output LSTM model.
        df_test      : test DataFrame with all ring crystals.
        var          : list of predictor column names.
        n_steps      : look-back window size.
        n_outputs    : forecast horizon.
        stride       : step size between windows.
        scalers_X    : dict {xtal_id: scaler_X} from training.
        scalers_y    : dict {xtal_id: scaler_y} from training.
        target_var   : name of the target column.
        time_col     : column used for temporal sorting.
        ring_metrics : list of ring-level metrics to compute.
                       Available: "wmape_pond" | "mae_global" | "rmse_global"
                       None → all three.

    Returns:
        Tuple (metrics_dict, df_per_crystal).
        metrics_dict contains the requested ring-level metrics.
        df_per_crystal has columns {"xtal_id", "wmape", "n"}.
        The primary scalar for grid accumulation is metrics_dict["wmape_pond"].
    """
    _ring_metrics = ring_metrics if ring_metrics is not None else [
        "wmape_pond", "mae_global", "rmse_global"
    ]

    results_per_xtal       = []
    all_y_true, all_y_pred = [], []

    for xtal in df_test["xtal_id"].unique():
        scaler_X = scalers_X.get(xtal) if scalers_X else None
        scaler_y = scalers_y.get(xtal) if scalers_y else None

        if scalers_X and scaler_X is None:
            print(f"[WARNING] Crystal {xtal} has no scaler_X (not in training set). Skipped.")
            continue
        if scalers_y and scaler_y is None:
            print(f"[WARNING] Crystal {xtal} has no scaler_y (not in training set). Skipped.")
            continue

        df_xtal = df_test[df_test["xtal_id"] == xtal]
        try:
            y_pred, y_true, _ = _prepare_predictions_xtal(
                model, df_xtal, var, n_steps, n_outputs, stride,
                scaler_X, scaler_y,
                target_var=target_var, time_col=time_col,
            )
            # True per-crystal WMAPE: Σ|y_i − ŷ_i| / Σ|y_i| × 100
            num     = np.sum(np.abs(y_true - y_pred))
            den     = np.sum(np.abs(y_true))
            wmape_i = (num / den * 100) if den != 0 else 0.0

            results_per_xtal.append({
                "xtal_id": xtal,
                "wmape":   wmape_i,
                "n":       len(y_true),
            })
            all_y_true.append(y_true.flatten())
            all_y_pred.append(y_pred.flatten())

        except Exception:
            print(f"[ERROR] Crystal {xtal}:")
            traceback.print_exc()
            continue

    if not results_per_xtal:
        return {m: np.nan for m in _ring_metrics}, pd.DataFrame()

    df_res     = pd.DataFrame(results_per_xtal)
    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    final_metrics = _compute_ring_metrics(y_true_all, y_pred_all, df_res, _ring_metrics)
    return final_metrics, df_res


def evaluate_and_plot_model(
    model, history, df_test, var, n_steps, n_outputs=1, stride=1,
    scalers_X=None, scalers_y=None, xtal_id=None,
    target_var="calibration", plot_ratio=True,
    results_dir=None, time_col="laser_datetime",
    metrics: list = None,
    loss: str = None,
):
    """
    Evaluate a specific crystal (or the first valid one) and generate diagnostic plots:
        Figure 1 — Training vs validation loss curve.
        Figure 2 — Predicted vs actual values (with optional ratio subplot).

    history can be either a Keras History object or a plain dict. Both are
    handled transparently via a hasattr(history, "history") check.

    Args:
        model       : trained multi-output LSTM model.
        history     : Keras History object or plain dict {"loss": [...], "val_loss": [...]}.
        df_test     : test DataFrame with all ring crystals.
        var         : list of predictor column names.
        n_steps     : look-back window size.
        n_outputs   : forecast horizon.
        stride      : step between windows.
        scalers_X   : dict {xtal_id: scaler_X} from training.
        scalers_y   : dict {xtal_id: scaler_y} from training.
        xtal_id     : crystal ID to plot. None → first valid crystal in df_test.
        target_var  : name of the target column.
        plot_ratio  : if True, adds a lower subplot showing the true/pred ratio.
        results_dir : directory to save figures (None = do not save).
        time_col    : column used for temporal sorting.
        metrics     : list of individual-crystal metrics to compute.
                      None → ["mape", "smape", "mae", "rmse", "maxae", "r2"].
        loss        : loss function name; used only to label the loss curve Y-axis.
                      None → "mse".

    Returns:
        Dictionary of computed metrics for the selected crystal.
    """
    xtals_test = df_test["xtal_id"].unique().tolist()
    if xtal_id is not None:
        xtal = xtal_id
    else:
        # Auto-select: first crystal that has a matching scaler
        xtal = next(
            (x for x in xtals_test if not scalers_X or x in scalers_X),
            xtals_test[0],
        )

    scaler_X = scalers_X.get(xtal) if scalers_X else None
    scaler_y = scalers_y.get(xtal) if scalers_y else None

    if scalers_X and scaler_X is None:
        raise ValueError(f"Crystal {xtal} has no scaler. Select another with xtal_id=<id>.")

    df_xtal = df_test[df_test["xtal_id"] == xtal]
    y_pred, y_true, valid_time = _prepare_predictions_xtal(
        model, df_xtal, var, n_steps, n_outputs, stride,
        scaler_X, scaler_y, target_var, time_col,
    )

    # Individual crystal metrics
    _metrics = metrics if metrics is not None else [
        "mape", "smape", "mae", "rmse", "maxae", "r2"
    ]
    results = _compute_metrics(y_true, y_pred, _metrics)

    print(f"\n── Crystal {xtal} metrics ───────────────────────")
    for name, value in results.items():
        print(f"  {name.upper():6s}: {value:.4f}")
    print("────────────────────────────────────────────────")

    # ── Figure 1: loss curve ──────────────────────────────────────────────────
    # Handle both Keras History objects and plain dicts transparently
    loss_label    = (loss or "mse").upper()
    hist_loss     = (history.history["loss"]     if hasattr(history, "history")
                     else history["loss"])
    hist_val_loss = (history.history["val_loss"] if hasattr(history, "history")
                     else history["val_loss"])

    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    ax_loss.plot(hist_loss,     label="Train")
    ax_loss.plot(hist_val_loss, label="Validation")
    ax_loss.set_title(f"Training vs validation loss (n_steps={n_steps})")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel(f"Loss ({loss_label})")
    ax_loss.legend()
    plt.tight_layout()
    if results_dir:
        fig_loss.savefig(
            os.path.join(results_dir, f"loss_xtal_{xtal}_{n_steps}.png"), dpi=300
        )
    plt.show()

    # ── Figure 2: predicted vs actual (+ optional ratio subplot) ─────────────
    suffix      = "" if xtal_id is not None else " (auto)"
    metrics_str = "  |  ".join(
        f"{name.upper()}: {value:.4f}" for name, value in results.items()
    )
    title = (
        f"Actual vs predicted — horizon {n_steps}"
        f" — crystal {xtal}{suffix}\n{metrics_str}"
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
            if xtal_id is None
            else f"prediction_vs_actual_xtal_{xtal}_{n_steps}.png"
        )
        fig.savefig(os.path.join(results_dir, filename), dpi=300)
    plt.show()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 11. RESULTS ACCUMULATION AND REPORTING (grid search)
# ──────────────────────────────────────────────────────────────────────────────

# Global accumulators: keyed by configuration tuple, updated after each grid run
accumulated_metrics_global = defaultdict(lambda: defaultdict(float))
config_details_global      = {}


def _config_key(r: dict) -> tuple:
    """
    Generate a hashable key that uniquely identifies a model configuration.

    All hyperparameters that the grid can vary are included to avoid collisions.

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
        r.get("dropout",       0.2),
        r.get("optimizer",     "adam") or "adam",
        r.get("learning_rate", 1e-3),
        r.get("loss",          "mse")  or "mse",
        r.get("monitor",       "val_loss"),
        r.get("min_delta",     0.0)    or 0.0,
        r["shuffle"],
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
                "cfg_ids":   [],
                "n_steps":   [],
                "n_outputs": [],
                "stride":    [],
                "metrics":   defaultdict(list),  # {metric_name: [val_h1, val_h2, ...]}
            }
        config_details_global[key]["cfg_ids"] .append(r.get("cfg_id", None))
        config_details_global[key]["n_steps"] .append(r["n_steps"])
        config_details_global[key]["n_outputs"].append(r["n_outputs"])
        config_details_global[key]["stride"]  .append(r["stride"])

        for metric_name, value in r["evaluated_metrics"].items():
            accumulated_metrics_global[key][metric_name] += value
            config_details_global[key]["metrics"][metric_name].append(value)


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
            (var, rnn_units,
             ext_norm_method, internal_norm,
             dropout_k, optimizer_k, learning_rate, loss_k,
             monitor_k, min_delta_k, shuffle) = key
            det = config_details_global[key]

            f.write("Configuration:\n")
            f.write(f"  cfg_ids:    {det['cfg_ids']}\n")
            f.write(f"  Variables:  {var}\n")
            f.write(f"  rnn_units:  {rnn_units}\n")
            f.write(f"  optimizer:  {optimizer_k}  |  lr: {learning_rate}\n")
            f.write(f"  loss:       {loss_k}  |  monitor: {monitor_k}"
                    f"  |  min_delta: {min_delta_k}\n")
            f.write(f"  norm:       {ext_norm_method}  |  batchnorm: {internal_norm}"
                    f"  |  dropout: {dropout_k}  |  shuffle: {shuffle}\n")

            for metric_name in list(det["metrics"].keys()):
                cumulative  = totals.get(metric_name, 0.0)
                values_list = det["metrics"][metric_name]
                f.write(f"  {metric_name.upper()} cumulative: {cumulative:.4f}\n")
                f.write("  Per-horizon detail:\n")

                better_horizons = []
                for ns, no, st, val in zip(det["n_steps"], det["n_outputs"],
                                           det["stride"], values_list):
                    f.write(f"    (n_steps={ns}, n_outputs={no}, stride={st})"
                            f" → {metric_name.upper()}: {val:.4f}\n")
                    ref_dict = reference_metrics.get(metric_name, {})
                    # Check if non-overlapping horizon outperforms the baseline
                    if ns == no == st and ns in ref_dict:
                        is_better = (
                            val > ref_dict[ns]
                            if metric_name in METRIC_HIGHER_IS_BETTER
                            else val < ref_dict[ns]
                        )
                        if is_better:
                            better_horizons.append(ns)
                f.write(f"  Horizons beating reference ({metric_name.upper()}):"
                        f" {better_horizons}\n")

            f.write("-" * 80 + "\n\n")


# ──────────────────────────────────────────────────────────────────────────────
# 12. MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────────────────

def _run_one_config(
    cfg_grid:      dict,
    df_train:      pd.DataFrame,
    df_test:       pd.DataFrame,
    horizons_list: list,
    cfg:           dict,
    output_file:   str,
) -> None:
    """
    Run the grid search for ONE configuration entry across all forecast horizons.

    Parameters are read from cfg_grid; unspecified ones inherit from cfg["default"].
    Fixed grid parameters (epochs, val_split, patience, etc.) come from
    cfg["grid_fixed"] and are never overridden by individual configs.

    Args:
        cfg_grid       : Single entry from cfg["grid_configs"].
        df_train       : Training DataFrame with all ring crystals.
        df_test        : Test DataFrame with all ring crystals.
        horizons_list  : List of forecast horizons (n_steps = n_outputs = stride).
        cfg            : Global configuration dictionary (see CONFIG).
        output_file    : Path to the text results file.
    """
    d   = cfg["default"]
    gf  = cfg["grid_fixed"]

    cur_var        = VARIABLE_SETS[cfg_grid["variables"]]
    rnn_units      = cfg_grid["rnn_units"]
    shuffle_flag   = cfg_grid["shuffle"]
    cfg_id         = cfg_grid["id"]

    # Optional parameters in cfg_grid → fall back to default if not specified
    dropout        = cfg_grid.get("dropout",       d["dropout"])
    optimizer_name = cfg_grid.get("optimizer",     d.get("optimizer",     None))
    learning_rate  = cfg_grid.get("learning_rate", d["learning_rate"])
    loss_name      = cfg_grid.get("loss",          d.get("loss",          None))
    norm_key       = cfg_grid.get("norm_method",   d["norm_method"])
    batchnorm      = cfg_grid.get("internal_norm", d["internal_norm"])
    monitor_metric = cfg_grid.get("monitor",       d.get("monitor",       "val_loss"))
    min_delta_val  = cfg_grid.get("min_delta",     d.get("min_delta",     None))
    ring_metrics   = gf["ring_metrics"]
    use_cpu        = gf["use_cpu"]
    save_w         = gf["save_weights"]

    for a in horizons_list:
        # batch_size: None → dynamic based on horizon size
        batch = gf["batch_size"]
        if batch is None:
            batch = 32 if a >= 48 else 128
        try:
            reset_environment(seed=cfg["seed"], use_cpu=use_cpu)

            model, history, scalers_X, scalers_y = train_model(
                df_train=df_train, var=cur_var,
                target_var=cfg["target_var"],
                n_steps=a, n_outputs=a, stride=a,
                rnn_units=rnn_units,
                dropout=dropout,
                epochs=gf["epochs"],
                batch_size=batch,
                optimizer=optimizer_name,
                learning_rate=learning_rate,
                loss=loss_name,
                ext_norm_method=norm_key,
                internal_norm=batchnorm,
                val_split_ratio=gf["val_split"],
                patience=gf["patience"],
                monitor=monitor_metric,
                min_delta=min_delta_val,
                shuffle=shuffle_flag,
                save_weights=save_w,
                results_dir=cfg["results_dir"],
                ring_name=cfg["active_ring"],
                time_col=cfg["time_col"],
                use_cpu=use_cpu,
            )

            metrics_dict, _ = evaluate_all_xtals(
                model, df_test, cur_var,
                n_steps=a, n_outputs=a, stride=a,
                scalers_X=scalers_X, scalers_y=scalers_y,
                time_col=cfg["time_col"],
                ring_metrics=ring_metrics,
            )

            # Print a compact summary line for this horizon run
            metrics_str = "  |  ".join(
                f"{k.upper()}: {v:.4f}" for k, v in metrics_dict.items()
            )
            print(f"  {metrics_str}")

            accumulate_results([{
                "cfg_id":           cfg_id,
                "var":              cur_var,
                "n_steps":          a,
                "n_outputs":        a,
                "stride":           a,
                "rnn_units":        rnn_units,
                "ext_norm_method":  norm_key,
                "internal_norm":    batchnorm,
                "dropout":          dropout,
                "optimizer":        optimizer_name,
                "learning_rate":    learning_rate,
                "loss":             loss_name,
                "monitor":          monitor_metric,
                "min_delta":        min_delta_val if min_delta_val is not None else 0.0,
                "shuffle":          shuffle_flag,
                "evaluated_metrics": metrics_dict,
            }])
            print_results_txt(gf["reference_metrics"], output_file)

        except Exception:
            print(f"[ERROR] cfg {cfg_id} | n_steps={a}:")
            traceback.print_exc()
        finally:
            # Release GPU memory and Python objects between runs
            tf.keras.backend.clear_session()
            gc.collect()


def run_grid_search(df_train, df_test, cfg=CONFIG, output_file=None):
    """
    Iterate over all configurations in cfg['grid_configs'] and all
    forecast horizons in cfg['grid_fixed']['horizons'].

    Each configuration inherits unspecified fields from CONFIG['default'].
    Results are accumulated and written to a text report after each run.

    Args:
        df_train    : Training DataFrame with all ring crystals.
        df_test     : Test DataFrame with all ring crystals.
        cfg         : Global configuration dictionary (see CONFIG).
        output_file : Path to the output results file. Defaults to
                      cfg['results_dir'] / cfg['results_file'].
    """
    output_file = output_file or os.path.join(cfg["results_dir"], cfg["results_file"])
    for cfg_grid in cfg["grid_configs"]:
        print(f"\n{'='*60}")
        print(f"Config {cfg_grid['id']} | "
              f"vars={cfg_grid['variables']} | "
              f"rnn={cfg_grid['rnn_units']} | "
              f"shuffle={cfg_grid['shuffle']}")
        print(f"{'='*60}")
        _run_one_config(
            cfg_grid=cfg_grid,
            df_train=df_train,
            df_test=df_test,
            horizons_list=cfg["grid_fixed"]["horizons"],
            cfg=cfg,
            output_file=output_file,
        )


def run_single(df_train, df_test, cfg=CONFIG):
    """
    Train and evaluate the configuration defined in CONFIG['default'].

    Runs training, then ring-level evaluation (all crystals), then
    a single-crystal plot for the crystal specified by plot_xtal_id.

    Args:
        df_train : Training DataFrame with all ring crystals.
        df_test  : Test DataFrame with all ring crystals.
        cfg      : Global configuration dictionary (see CONFIG).
    """
    d = cfg["default"]
    reset_environment(seed=cfg["seed"], use_cpu=d.get("use_cpu", False))

    # batch_size: None → dynamic (32 if n_steps >= 48, 128 if n_steps < 48)
    batch = d["batch_size"]
    if batch is None:
        batch = 32 if d["n_steps"] >= 48 else 128

    model, history, scalers_X, scalers_y = train_model(
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
        optimizer=d.get("optimizer",     None),
        learning_rate=d["learning_rate"],
        loss=d.get("loss",               None),
        ext_norm_method=d["norm_method"],
        internal_norm=d["internal_norm"],
        val_split_ratio=d["val_split"],
        patience=d["patience"],
        monitor=d.get("monitor",         "val_loss"),
        min_delta=d.get("min_delta",     None),
        shuffle=d["shuffle"],
        save_weights=d.get("save_weights", False),
        results_dir=cfg["results_dir"],
        ring_name=cfg["active_ring"],
        time_col=cfg["time_col"],
        use_cpu=d.get("use_cpu",         False),
    )

    # ── Ring-level evaluation (all crystals) ──────────────────────────────────
    ring_metrics_dict, df_res = evaluate_all_xtals(
        model, df_test, d["variables"],
        n_steps=d["n_steps"], n_outputs=d["n_outputs"], stride=d["stride"],
        scalers_X=scalers_X, scalers_y=scalers_y,
        time_col=cfg["time_col"],
        ring_metrics=d.get("ring_metrics", None),
    )
    print("\n── Ring-level metrics (all crystals) ────────────")
    for name, value in ring_metrics_dict.items():
        print(f"  {name.upper():12s}: {value:.4f}")
    print("────────────────────────────────────────────────")
    print(df_res.to_string(index=False))

    # ── Single-crystal plot ───────────────────────────────────────────────────
    evaluate_and_plot_model(
        model=model, history=history,
        df_test=df_test, var=d["variables"],
        n_steps=d["n_steps"], n_outputs=d["n_outputs"], stride=d["stride"],
        scalers_X=scalers_X, scalers_y=scalers_y,
        xtal_id=d.get("plot_xtal_id", None),
        plot_ratio=True,
        results_dir=cfg["results_dir"],
        time_col=cfg["time_col"],
        metrics=d.get("test_metrics", None),
        loss=d.get("loss", None),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 13. LOADING A SAVED MODEL
# ──────────────────────────────────────────────────────────────────────────────

def load_model(cfg=CONFIG):
    """
    Reconstruct the multi-output LSTM model previously saved with save_weights=True.

    Loads from:
        <results_dir>/<ring>_multi_bi_<n_steps>_<n_outputs>_<norm>/model_<n_steps>steps.weights.h5
        <results_dir>/<ring>_multi_bi_<n_steps>_<n_outputs>_<norm>/scaler_X_<xtal_id>.pkl
        <results_dir>/<ring>_multi_bi_<n_steps>_<n_outputs>_<norm>/scaler_y_<xtal_id>.pkl
        <results_dir>/<ring>_multi_bi_<n_steps>_<n_outputs>_<norm>/n_vars.txt

    The saved variable list in n_vars.txt is validated against the current CONFIG
    to prevent silent mismatches. All per-crystal scalers are loaded and cross-validated
    to ensure scalers_X and scalers_y cover the same set of crystal IDs.

    Args:
        cfg: Global configuration dictionary (see CONFIG).

    Returns:
        Tuple (model, scalers_X, scalers_y) ready for evaluation.

    Raises:
        FileNotFoundError : If n_vars.txt or scaler files are missing.
        ValueError        : If the saved variable list differs from CONFIG,
                            or if scalers_X and scalers_y have mismatched keys.
    """
    d           = cfg["default"]
    n_steps     = d["n_steps"]
    n_outputs   = d["n_outputs"]
    ring_str    = cfg["active_ring"]
    norm_str    = d["norm_method"].lower()
    weights_dir = os.path.join(
        cfg["results_dir"], f"{ring_str}_multi_bi_{n_steps}_{n_outputs}_{norm_str}"
    )

    # ── Step 1: Validate saved variables against current CONFIG ───────────────
    n_vars_path = os.path.join(weights_dir, "n_vars.txt")
    if not os.path.exists(n_vars_path):
        raise FileNotFoundError(
            f"n_vars.txt not found in {weights_dir}. "
            f"The model may be outdated or incompatible."
        )
    with open(n_vars_path, "r", encoding="utf-8") as f:
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
    # Encoder column order for external inference:
    #   1. Build the array in the original order [var1, ..., calibration]
    #   2. Apply scaler_X.transform() with that order
    #   3. Pass directly to the model (shape: n_steps × n_features)
    n_features  = len(d["variables"]) + 1   # [var1, ..., calibration]
    input_shape = (n_steps, n_features)
    model = build_model(
        input_shape=input_shape,
        n_outputs=n_outputs,
        rnn_units=d["rnn_units"],
        dropout=d["dropout"],
        use_batchnorm=d["internal_norm"],
    )

    # Dummy forward pass to initialize all layer weights before loading saved values
    dummy_X = np.zeros((1, n_steps, n_features), dtype=np.float32)
    model.predict(dummy_X, verbose=0)

    # ── Step 3: Load model weights ────────────────────────────────────────────
    model.load_weights(
        os.path.join(weights_dir, f"model_{n_steps}steps.weights.h5")
    )

    # ── Step 4: Load per-crystal scalers ──────────────────────────────────────
    scalers_X, scalers_y = {}, {}
    for fname in os.listdir(weights_dir):
        if fname.startswith("scaler_X_") and fname.endswith(".pkl"):
            raw_id = fname.replace("scaler_X_", "").replace(".pkl", "")
            try:
                xtal_id = int(raw_id)
            except ValueError:
                xtal_id = raw_id
            scalers_X[xtal_id] = joblib.load(os.path.join(weights_dir, fname))
        elif fname.startswith("scaler_y_") and fname.endswith(".pkl"):
            raw_id = fname.replace("scaler_y_", "").replace(".pkl", "")
            try:
                xtal_id = int(raw_id)
            except ValueError:
                xtal_id = raw_id
            scalers_y[xtal_id] = joblib.load(os.path.join(weights_dir, fname))

    if not scalers_X:
        raise FileNotFoundError(
            f"No scaler_X_*.pkl files found in {weights_dir}."
        )
    if not scalers_y:
        raise FileNotFoundError(
            f"No scaler_y_*.pkl files found in {weights_dir}."
        )
    # Cross-validate that both scaler dicts cover the same set of crystals
    if set(scalers_X.keys()) != set(scalers_y.keys()):
        only_X = set(scalers_X.keys()) - set(scalers_y.keys())
        only_y = set(scalers_y.keys()) - set(scalers_X.keys())
        raise ValueError(
            f"Mismatch between scalers_X and scalers_y:\n"
            f"  Only in scalers_X: {only_X}\n"
            f"  Only in scalers_y: {only_y}\n"
            f"The weights directory may be incomplete or corrupted."
        )

    print(f"Model loaded from: {weights_dir}")
    print(f"  Crystals with scalers: {sorted(scalers_X.keys())}")
    return model, scalers_X, scalers_y


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Option A: train and evaluate a single configuration ───────────────────
     df_train, df_test = prepare_splits(CONFIG)
     run_single(df_train, df_test, CONFIG)          # ← default mode

    # ── Option B: run a full grid search ──────────────────────────────────────
     #run_grid_search(df_train, df_test, CONFIG)     # ← uncomment for grid search

    # ── Option C: load a previously saved model and evaluate ──────────────────
    # Requires a prior run with save_weights=True in CONFIG['default'].
    # Expected folder example: <results_dir>/ring_1_multi_bi_24_24_minmax/
'''
    _, df_test = prepare_splits(CONFIG)
    model, scalers_X, scalers_y = load_model(cfg=CONFIG)
    _d = CONFIG["default"]
    evaluate_and_plot_model(
        model=model, history={"loss": [], "val_loss": []},
        df_test=df_test, var=_d["variables"],
        n_steps=_d["n_steps"], n_outputs=_d["n_outputs"], stride=_d["stride"],
        scalers_X=scalers_X, scalers_y=scalers_y,
        xtal_id=_d.get("plot_xtal_id", None),
        plot_ratio=True,
        results_dir=CONFIG["results_dir"],
        time_col=CONFIG["time_col"],
        metrics=_d.get("test_metrics", None),
        loss=_d.get("loss", None),
    )'''