# =============================================================================
# lstm_seq2seq_arch1.py
# Encoder–Decoder LSTM (Seq2Seq) model for time-series forecasting.
# Adaptable to any dataset through the CONFIG dictionary.
#
# KEY DIFFERENCE compared to lstm_multioutput.py:
#   - Uses a custom Keras Model subclass (Seq2Seq) with encoder and decoder.
#   - Decoder is autoregressive: it receives the previous target and exogenous
#     variables at each step.
#   - Supports three teacher-forcing modes: "forced", "recursive", "mixed".
#   - Uses a manual GradientTape training loop (not model.fit).
#   - split_sequences returns (X, dec_X, y): decoder input is required.
#   - _build_loss returns a loss instance (not a string) because GradientTape
#     requires a callable, unlike model.compile() which accepts strings.
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
    "results_file": "results_seq2seq_arch1.txt",

    # ── Data filtering ───────────────────────────────────────────────────────
    "xtal_id":      30600,            # Crystal ID used for training and evaluation
    "target_var":   "calibration",    # Target column to predict
    "time_col":     "laser_datetime", # Column used for temporal ordering
    "calib_min":    0.7,              # Minimum accepted target value
    "calib_max":    1.0,              # Maximum accepted target value
    "train_year":   [2016],           # List of years used for training
    "test_year":    2017,             # Year used for external evaluation

    # ── Columns to retain after loading ──────────────────────────────────────
    "keep_cols": [
        "xtal_id", "calibration", "int_deliv_inv_ub",
        "laser_datetime", "time"
    ],

    # ── Reproducibility ──────────────────────────────────────────────────────
    "seed": 1234,

    # =========================================================================
    #  DEFAULT CONFIGURATION (run_single)
    # =========================================================================
    "default": {
        # ── Predictor variables ───────────────────────────────────────────────
        "variables":      ["delta_lumi"],
        # n_steps == n_outputs == stride for non-overlapping windows
        "n_steps":        48,
        "n_outputs":      48,
        "stride":         48,

        # ── Architecture ─────────────────────────────────────────────────────
        "rnn_units":      [64, 64],  # Units per LSTM layer (shared by encoder and decoder)
        "internal_norm":  False,     # Apply BatchNormalization after each LSTM layer

        # ── Training ─────────────────────────────────────────────────────────
        "epochs":         1,
        # batch_size: None → dynamic (32 if n_steps >= 48, 128 if n_steps < 48)
        "batch_size":     None,

        # ── Optimizer ────────────────────────────────────────────────────────
        # Options: "adam" | "adamw" | "sgd" | "rmsprop" | "adagrad"
        # None → use "adam" (library default value)
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
        "patience":       50,   # Number of epochs without improvement before stopping
        # Training history metric to monitor: "val_loss" or "loss"
        "monitor":        "val_loss",
        # Minimum improvement required to reset the patience counter.
        # None → 0.0 (any improvement counts, with no minimum threshold)
        # Example: 1e-4 prevents stopping due to insignificant improvements
        "min_delta":      None,

        # ── Data normalization ────────────────────────────────────────────────
        # Options: "MinMax" | "Standard" | "PowerTransformer" | "None"
        "norm_method":    "MinMax",

        # ── Train/validation split ────────────────────────────────────────────
        "val_split":      0.1,   # Fraction of data used for validation (0–1)

        # ── Shuffle windows before training ───────────────────────────────────
        # True shuffles windows in the tf.data.Dataset (breaks temporal order)
        "shuffle":        False,

        # ── Teacher forcing mode ──────────────────────────────────────────────
        # "forced"    → decoder always receives the ground-truth value (more stable early on)
        # "recursive" → decoder always uses its own prediction (more realistic)
        # "mixed"     → gradual transition from forced to recursive during training
        "mode":           "forced",
        # The three parameters below only apply when mode == "mixed":
        "start_forcing":  1.0,  # Teacher forcing ratio at the start (1.0 = 100% ground truth)
        "end_forcing":    0.0,  # Ratio at the end of the decay (0.0 = 100% autoregressive)
        "decay_epochs":   30,   # Epochs to transition from start_forcing to end_forcing
        # Scheduled sampling per sample (True) vs. per sequence (False)
        "per_sample":     True,

        # ── Hardware ──────────────────────────────────────────────────────────
        # True  → force CPU usage (reproducible, slower)
        # False → use GPU if available
        "use_cpu":        False,

        # ── Evaluation metrics computed on the TEST set ───────────────────────
        # Metrics are computed on the original scale after training.
        # Available: "mape" | "smape" | "mae" | "rmse" | "maxae" | "r2"
        # - mape  : Mean Absolute Percentage Error (×100). Main project metric.
        # - smape : Symmetric MAPE; more robust when the target approaches 0.
        # - mae   : Mean Absolute Error; expressed in the same units as the target.
        # - rmse  : Root Mean Squared Error; penalizes large errors.
        # - maxae : Maximum absolute error; indicates the worst individual case.
        # - r2    : Coefficient of determination (1=perfect, 0=mean baseline, <0=worse).
        "test_metrics":   ["mape", "smape", "mae", "rmse", "maxae", "r2"],

        # ── Saving ───────────────────────────────────────────────────────────
        # True → save model weights and scalers after training
        "save_weights":   True,
    },

    # =========================================================================
    # FIXED GRID PARAMETERS (not combined, applied to all runs)
    # =========================================================================
    "grid_fixed": {
        # ── Training epochs ───────────────────────────────────────────────────
        "epochs": 1,
        # ── Forecast horizons to evaluate ─────────────────────────────────────
        "horizons": [1, 12, 24, 36, 48, 60, 72, 84, 96],
        # ── Evaluation metrics (fixed for the entire grid) ────────────────────
        "test_metrics":       ["mape", "mae", "r2"],
        # ── Reference metric values per horizon (baseline to outperform) ───────
        "reference_metrics": {
            "mape": {
                1: 0.09, 12: 0.327, 24: 0.397, 36: 0.456,
                48: 0.529, 60: 0.578, 72: 0.63, 84: 0.674, 96: 0.723,
            },
            "mae": {
                1: 1, 12: 1, 24: 1, 36: 1,
                48: 1, 60: 1, 72: 1, 84: 1, 96: 1,
            },
            "r2": {
                1: 0.1, 12: 0.1, 24: 0.1, 36: 0.1,
                48: 0.1, 60: 0.1, 72: 0.1, 84: 0.1, 96: 0.1,
            },
        },
        # batch_size: None → dynamic (32 if n_steps >= 48, 128 if n_steps < 48)
        "batch_size":         None,
        # ── Fixed training parameters ─────────────────────────────────────────
        "val_split":          0.1,
        "patience":           50,
        "start_forcing":      1.0,
        "end_forcing":        0.0,
        "decay_epochs":       30,
        # ── Hardware ──────────────────────────────────────────────────────────
        "use_cpu":            False,
        "save_weights":       False,  # Do not enable when testing many configurations
    },

    # =========================================================================
    # GRID CONFIGURATION (parameters combined through Cartesian product)
    # =========================================================================
    "grid": {
        # ── Architectures to explore ──────────────────────────────────────────
        "rnn_units": [
            [128, 128],
            [256, 256],
        ],
        # ── Variable sets ─────────────────────────────────────────────────────
        "variables": [
            ["delta_lumi"],
        ],
        # ── Normalizers ───────────────────────────────────────────────────────
        "norm_method": [
            "MinMax",
        ],
        "internal_norm": [
            False,
        ],
        # ── Shuffling of windows ──────────────────────────────────────────────
        "shuffle": [
            False,
        ],
        # ── Teacher forcing ───────────────────────────────────────────────────
        "mode": [
            "forced",
            # "mixed",
        ],
        "per_sample": [
            True,
        ],
        # ── Optimizer ────────────────────────────────────────────────────────
        # None → uses the default value defined in the code ("adam")
        "optimizer": [
            None,
        ],
        # ── Learning rate ─────────────────────────────────────────────────────
        "learning_rate": [
            1e-3,
        ],
        # ── Loss function ─────────────────────────────────────────────────────
        # None → uses the default value defined in the code ("mse")
        "loss": [
            None,
        ],
        # ── Metric to monitor for early stopping ──────────────────────────────
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
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, LSTM, LSTMCell
)
from tensorflow.keras.models import Model

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
    d_lum  = i["int_deliv_inv_ub"].diff()

    i["dint_dtime"]   = d_lum / dt_sec         # Luminosity rate of change
    i["delta_lumi"]   = d_lum                  # Absolute luminosity increment
    i["delta_t"]      = i["laser_datetime"].diff().dt.total_seconds()  # Time between laser shots
    i["days_no_int"]  = i["time"].diff().dt.days  # Days since last interaction

    # ── Edge case and outlier corrections ────────────────────────────────────
    # First row has no previous value; copy the rate from the next row
    i.loc[0, "dint_dtime"] = i.loc[1, "dint_dtime"]
    # Reset rate to 0 on days with no interaction (no luminosity delivered)
    i.loc[i["days_no_int"] > 0,  "dint_dtime"]               = 0
    # Zero-fill the first row for difference-based features
    i.loc[0, ["days_no_int", "delta_t", "delta_lumi"]]       = 0
    # Suppress spurious deltas after long gaps (> 30 days without interaction)
    i.loc[i["days_no_int"] > 30, ["delta_lumi", "delta_t"]]  = 0
    # Fill any remaining NaN/inf values in the rate column
    i["dint_dtime"] = i["dint_dtime"].bfill().ffill()
    i.loc[~np.isfinite(i["dint_dtime"]), "dint_dtime"]        = 0

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
# as the `var` argument to train_seq2seq() or evaluate_model().
VARIABLE_SETS = {
    "var1": ["int_deliv_inv_ub", "dint_dtime"],
    "var2": ["int_deliv_inv_ub", "dint_dtime", "delta_lumi"],
    "var3": ["int_deliv_inv_ub", "dint_dtime", "delta_t"],
    "var4": ["int_deliv_inv_ub"],
    "var5": ["dint_dtime"],
    "var6": ["delta_lumi"],
}


# ──────────────────────────────────────────────────────────────────────────────
# 5. WINDOWING FOR SEQ2SEQ
# ──────────────────────────────────────────────────────────────────────────────

def split_sequences(
    sequences: np.ndarray,
    n_steps: int,
    n_outputs: int = 1,
    stride: int = 1,
    shuffle_windows: bool = False,
    start_token=None,
    calib_col: int = None,
    y_col: int = None,
):
    """
    Generate sliding windows for Seq2Seq prediction (decoder with future exogenous variables).

    Expected column layout in `sequences`: [var1, ..., calibration, target]
        ─ calibration: second-to-last column by default (or specify with calib_col)
        ─ target:      last column by default             (or specify with y_col)

    # ── ENCODER INPUT COLUMN ORDER ────────────────────────────────────────────
    # IMPORTANT: the encoder receives [calibration★, var1★, var2★, ...]
    # This order DIFFERS from the one scaler_X was fitted on ([var1, ..., calibration]).
    # The reordering happens AFTER scaling, so the values are correct.
    # However, for external inference you must:
    #   1. Build the array in the original order [var1, ..., calibration]
    #   2. Apply scaler_X.transform() with that order
    #   3. Reorder to [calibration, var1, ...] before passing to the encoder
    #   4. The decoder receives [target★★, var1★, ...] without reordering

    Args:
        sequences     : 2-D array of shape (T, n_cols).
        n_steps       : number of past time steps used as encoder input (look-back window).
        n_outputs     : number of future steps to predict per window.
        stride        : step size between consecutive windows.
                        Set stride == n_steps for non-overlapping windows.
        shuffle_windows: if True, randomly permutes the generated windows.
        start_token   : initial value fed to the decoder at step 0.
                        None  → last known target value.
                        scalar → scalar broadcast to all samples.
                        array  → full first decoder input row.
        calib_col     : index of the calibration column (default: second-to-last).
        y_col         : index of the target column (default: last).

    Returns:
        X      (N, n_steps,   1 + n_vars)     ← [calibration, vars...] encoder input
        dec_X  (N, n_outputs, 1 + n_dec_vars) ← [target, vars...]      decoder input
        y      (N, n_outputs)                 ← future target values

    Raises:
        AssertionError: If column indices are invalid or overlap.
    """
    assert sequences.shape[1] >= 2, \
        "At least 2 columns are required (features + target)."

    X_list, dec_list, y_list = [], [], []
    T      = sequences.shape[0]
    n_cols = sequences.shape[1]

    y_col     = y_col     if y_col     is not None else n_cols - 1
    calib_col = calib_col if calib_col is not None else n_cols - 2

    assert y_col != calib_col,      "calibration and target cannot be the same column."
    assert 0 <= calib_col < n_cols, f"calib_col={calib_col} is out of range."
    assert 0 <= y_col     < n_cols, f"y_col={y_col} is out of range."

    # Decoder columns: all columns except calib_col and y_col
    dec_cols = [c for c in range(n_cols) if c not in (calib_col, y_col)]

    for i in range(0, T - n_steps - n_outputs + 1, stride):
        end_ix  = i + n_steps
        out_end = end_ix + n_outputs

        # ── Encoder: reorder columns putting calibration first ────────────────
        enc_raw = sequences[i:end_ix, :-1]
        calib   = enc_raw[:, calib_col]
        other   = np.delete(enc_raw, calib_col, axis=1)
        seq_x   = np.column_stack((calib, other))

        # ── Targets ───────────────────────────────────────────────────────────
        seq_y = sequences[end_ix:out_end, y_col]

        # ── Decoder input ─────────────────────────────────────────────────────
        dec_in   = np.zeros((n_outputs, len(dec_cols) + 1), dtype=sequences.dtype)
        last_row = sequences[end_ix - 1]

        if start_token is None:
            dec_in[0, 0]  = last_row[y_col]
            dec_in[0, 1:] = last_row[dec_cols]
        elif np.isscalar(start_token):
            dec_in[0, 0]  = start_token
            dec_in[0, 1:] = last_row[dec_cols]
        else:
            dec_in[0, :] = start_token

        if n_outputs > 1:
            future_vars = sequences[end_ix:out_end - 1][:, dec_cols]
            dec_in[1:]  = np.column_stack((
                sequences[end_ix:out_end - 1, y_col],
                future_vars,
            ))

        X_list.append(seq_x)
        dec_list.append(dec_in)
        y_list.append(seq_y)

    X     = np.array(X_list)
    dec_X = np.array(dec_list)
    y     = np.array(y_list)

    if shuffle_windows:
        idx = np.random.permutation(len(X))
        return X[idx], dec_X[idx], y[idx]
    return X, dec_X, y


# ──────────────────────────────────────────────────────────────────────────────
# 6. SEQ2SEQ MODEL (decoder with exogenous variables)
# ──────────────────────────────────────────────────────────────────────────────

class Seq2Seq(Model):
    """
    Stacked Encoder–Decoder LSTM with teacher forcing and scheduled sampling.
    The decoder receives the previous target and exogenous variables at each step.

    Args:
        encoder_units    : int or list of ints — units per encoder LSTM layer.
        decoder_units    : int or list of ints — same length as encoder_units.
        output_dim       : output dimension per time step (typically 1).
        n_outputs        : forecast horizon (number of future steps to predict).
        encoder_dropout  : dropout rate applied after each encoder LSTM layer.
        encoder_batchnorm: if True, inserts BatchNormalization after each encoder LSTM.
    """

    def __init__(
        self,
        encoder_units=128,
        decoder_units=128,
        output_dim=1,
        n_outputs=10,
        encoder_dropout=0.0,
        encoder_batchnorm=False,
    ):
        super().__init__()

        if isinstance(encoder_units, int):
            encoder_units = [encoder_units]
        if isinstance(decoder_units, int):
            decoder_units = [decoder_units]
        assert len(encoder_units) == len(decoder_units), (
            "encoder_units and decoder_units must have the same number of layers."
        )

        self.n_outputs     = n_outputs
        self.output_dim    = output_dim
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units

        # Encoder: stacked LSTM layers
        # IMPORTANT: flat separate lists (NOT a list of dicts) are used here so
        # Keras can track the layers correctly and save/load weights via
        # save_weights / load_weights without "0 variables received" errors.
        self.encoder_lstm_layers = [
            LSTM(u, return_sequences=True, return_state=True) for u in encoder_units
        ]
        self.encoder_bn_layers = [
            BatchNormalization() if encoder_batchnorm else None for _ in encoder_units
        ]
        self.encoder_drop_layers = [
            Dropout(encoder_dropout) if encoder_dropout > 0 else None for _ in encoder_units
        ]

        # Decoder: stacked LSTMCells
        cells = [LSTMCell(u) for u in decoder_units]
        self.decoder_cell = (
            tf.keras.layers.StackedRNNCells(cells) if len(cells) > 1 else cells[0]
        )

        # Output projection layer
        self.out_dense = Dense(output_dim)

    # ──────────────────────────────────────────────────────────────────────────
    def _encode(self, x, training):
        """Pass x through all encoder LSTM layers and return (seq_out, states)."""
        states = []
        for lstm, bn, drop in zip(
            self.encoder_lstm_layers,
            self.encoder_bn_layers,
            self.encoder_drop_layers,
        ):
            x, h, c = lstm(x, training=training)
            states += [h, c]
            if bn is not None:
                x = bn(x, training=training)
            if drop is not None:
                x = drop(x, training=training)
        return x, states

    def _init_dec_states(self, enc_states):
        """
        Convert [h1, c1, h2, c2, ...] to the state format expected by the decoder.

        Args:
            enc_states: flat list of encoder hidden and cell states.

        Returns:
            State tuple compatible with StackedRNNCells or a single LSTMCell.
        """
        if isinstance(self.decoder_cell, tf.keras.layers.StackedRNNCells):
            return [(enc_states[i], enc_states[i + 1])
                    for i in range(0, len(enc_states), 2)]
        return [enc_states[-2], enc_states[-1]]

    # ──────────────────────────────────────────────────────────────────────────
    def call(self, encoder_inputs, decoder_inputs,
             training=False, forcing_ratio=1.0, per_sample=False):
        """
        Forward pass for the Seq2Seq model.

        Args:
            encoder_inputs : (batch, n_steps,   n_enc_features)
            decoder_inputs : (batch, n_outputs, n_decoder_features)
            training       : bool — controls Dropout and BatchNorm behavior.
            forcing_ratio  : float in [0, 1] — probability of using the ground-truth
                             target as the next decoder input (teacher forcing).
            per_sample     : if True, the forcing decision is made independently
                             per sample in the batch (scheduled sampling);
                             if False, a single decision applies to the whole batch.

        Returns:
            Tensor of shape (batch, n_outputs, output_dim).
        """
        x = tf.cast(encoder_inputs, tf.float32)

        # ── Encoder ──────────────────────────────────────────────────────────
        _, enc_states = self._encode(x, training)
        dec_states    = self._init_dec_states(enc_states)

        # ── Decoder ──────────────────────────────────────────────────────────
        dec_inputs   = tf.cast(decoder_inputs, tf.float32)
        if dec_inputs.shape.ndims == 2:      # (batch, n_outputs) → (batch, n_outputs, 1)
            dec_inputs = tf.expand_dims(dec_inputs, -1)

        batch_size   = tf.shape(dec_inputs)[0]
        dec_len      = tf.shape(dec_inputs)[1]
        dec_feat_dim = tf.shape(dec_inputs)[2]
        dec_input    = dec_inputs[:, 0, :]   # initial token (batch, n_dec_features)
        outputs      = []

        for t in range(self.n_outputs):
            out, dec_states = self.decoder_cell(dec_input, dec_states, training=training)
            pred = self.out_dense(out)        # (batch, output_dim)
            outputs.append(pred)

            def build_pred_next():
                # safe_t clamps the index so that BOTH branches of tf.cond are
                # valid during @tf.function tracing, avoiding out-of-bounds when
                # t = n_outputs - 1 (index t+1 would exceed the tensor length).
                # In eager mode this guard is unnecessary but harmless.
                has_next  = tf.less(t + 1, dec_len)
                safe_t    = tf.minimum(t + 1, dec_len - 1)
                exog_next = tf.cond(
                    has_next,
                    lambda: dec_inputs[:, safe_t, 1:],
                    lambda: tf.zeros([batch_size, dec_feat_dim - 1], dtype=dec_inputs.dtype)
                )
                return tf.concat([pred, exog_next], axis=-1)

            if training:
                if per_sample:
                    # Scheduled sampling: forcing decision is made per sample.
                    # safe_t clamps the index so @tf.function can trace both
                    # branches without out-of-bounds when t = n_outputs - 1.
                    mask     = tf.cast(
                        tf.random.uniform([batch_size]) < forcing_ratio,
                        dec_inputs.dtype,
                    )
                    mask     = tf.expand_dims(mask, -1)          # (batch, 1)
                    safe_t   = tf.minimum(t + 1, dec_len - 1)
                    has_next = tf.cast(tf.less(t + 1, dec_len), dec_inputs.dtype)
                    # Multiplying by has_next zeros out the teacher signal at the
                    # last step — equivalent to tf.zeros_like but without an extra branch.
                    teacher  = dec_inputs[:, safe_t, :] * has_next
                    pred_next = build_pred_next()
                    dec_input = mask * teacher + (1.0 - mask) * pred_next
                else:
                    # Global decision per sequence — fully tensorized to be
                    # compatible with @tf.function without retracing per epoch.
                    use_teacher = tf.random.uniform([], dtype=tf.float32) < forcing_ratio
                    safe_t      = tf.minimum(t + 1, dec_len - 1)
                    has_next    = tf.cast(tf.less(t + 1, dec_len), dec_inputs.dtype)
                    teacher_val = dec_inputs[:, safe_t, :] * has_next
                    dec_input   = tf.cond(
                        use_teacher,
                        lambda: teacher_val,
                        lambda: build_pred_next(),
                    )
            else:
                dec_input = build_pred_next()   # autoregressive inference

        return tf.stack(outputs, axis=1)          # (batch, n_outputs, output_dim)


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
        A tf.keras.optimizers instance ready for use with GradientTape.

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


def _build_loss(name: str):
    """
    Instantiate a Keras loss function from its string name.

    Returns a loss instance (not a string) because train_seq2seq uses a manual
    GradientTape loop that requires a callable loss, unlike model.compile()
    which accepts both strings and instances.

    Args:
        name : "mse" | "mae" | "huber" | "logcosh"
               None  → defaults to "mse"

    Returns:
        A tf.keras.losses instance callable as loss_fn(y_true, y_pred).

    Raises:
        ValueError: If an unrecognized loss name is provided.
    """
    name = (name or "mse").lower()
    options = {
        "mse":     tf.keras.losses.MeanSquaredError,
        "mae":     tf.keras.losses.MeanAbsoluteError,
        "huber":   tf.keras.losses.Huber,
        "logcosh": tf.keras.losses.LogCosh,
    }
    if name not in options:
        raise ValueError(f"Invalid loss: '{name}'. Options: {list(options)}")
    return options[name]()


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     metrics: list) -> dict:
    """
    Compute the requested evaluation metrics on the original (unscaled) values.

    Args:
        y_true  : ground-truth values (1-D array).
        y_pred  : model predictions (1-D array).
        metrics : list of metric names to compute.
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

def train_seq2seq(
    df_train:        pd.DataFrame,
    var:             list,
    target_var:      str   = "calibration",
    n_steps:         int   = 10,
    n_outputs:       int   = 1,
    stride:          int   = 1,
    enc_units              = 128,
    dec_units              = 128,
    epochs:          int   = 50,
    batch_size:      int   = 32,
    mode:            str   = "mixed",    # "forced" | "recursive" | "mixed"
    start_forcing:   float = 1.0,
    end_forcing:     float = 0.0,
    decay_epochs:    int   = 30,
    per_sample:      bool  = False,
    optimizer:       str   = None,       # None → "adam"; see _build_optimizer()
    learning_rate:   float = 1e-3,
    loss:            str   = None,       # None → "mse";  see _build_loss()
    ext_norm_method: str   = "MinMax",
    internal_norm:   bool  = False,
    val_split_ratio: float = 0.1,
    shuffle:         bool  = True,       # True shuffles the tf.data.Dataset each epoch
    patience:        int   = 50,
    monitor:         str   = "val_loss", # "val_loss" | "loss"
    min_delta:       float = None,       # None → 0.0 (no minimum improvement threshold)
    save_weights:    bool  = False,
    results_dir:     str   = None,
    xtal_id:         int   = None,
    time_col:        str   = "laser_datetime",
):
    """
    Train the Seq2Seq model on df_train using a manual GradientTape loop.

    Using a manual loop (instead of model.fit) gives full control over the
    teacher forcing ratio at each epoch and allows @tf.function-compiled
    train and validation steps for efficient GPU execution.

    If save_weights=True, the following files are written:
        <results_dir>/<xtal_id>_seq2seq_arch1_<mode>_<n_steps>_<n_outputs>_<norm>/model_<n_steps>steps.weights.h5
        <results_dir>/<xtal_id>_seq2seq_arch1_<mode>_<n_steps>_<n_outputs>_<norm>/scaler_X_<n_steps>steps.pkl
        <results_dir>/<xtal_id>_seq2seq_arch1_<mode>_<n_steps>_<n_outputs>_<norm>/scaler_y_<n_steps>steps.pkl
        <results_dir>/<xtal_id>_seq2seq_arch1_<mode>_<n_steps>_<n_outputs>_<norm>/n_vars.txt

    Args:
        df_train        : Training DataFrame with feature-engineered columns.
        var             : List of predictor column names.
        target_var      : Name of the target column to predict.
        n_steps         : Look-back window size (number of past time steps as encoder input).
        n_outputs       : Number of future steps to predict simultaneously.
        stride          : Step size between consecutive windows.
        enc_units       : Units per encoder LSTM layer (int or list of ints).
        dec_units       : Units per decoder LSTM layer (int or list of ints).
        epochs          : Maximum number of training epochs.
        batch_size      : Mini-batch size for training.
        mode            : Teacher forcing mode — "forced" | "recursive" | "mixed".
        start_forcing   : Initial teacher forcing ratio (1.0 = full teacher forcing).
        end_forcing     : Final teacher forcing ratio (0.0 = fully autoregressive).
        decay_epochs    : Number of epochs for the transition from start to end forcing.
        per_sample      : If True, the forcing decision is made per sample (scheduled sampling).
        optimizer       : Optimizer name. None → "adam".
        learning_rate   : Optimizer learning rate.
        loss            : Loss function name. None → "mse".
        ext_norm_method : External (input) normalization method.
        internal_norm   : If True, inserts BatchNormalization after each encoder LSTM.
        val_split_ratio : Fraction of training data used for validation.
        shuffle         : If True, shuffles the training dataset each epoch.
        patience        : Early stopping patience (epochs without improvement).
        monitor         : Metric to monitor for early stopping ("val_loss" or "loss").
        min_delta       : Minimum improvement required to reset the patience counter.
        save_weights    : If True, saves model weights and scalers to disk.
        results_dir     : Directory for saving weights (required if save_weights=True).
        xtal_id         : Crystal identifier used in file naming.
        time_col        : Column name used for temporal sorting.

    Returns:
        Tuple (model, history, scaler_X, scaler_y).
        history is a plain dict {"loss": [...], "val_loss": [...]}.
    """
    df_train = df_train.sort_values(time_col).copy()

    # ── Feature and target preparation ───────────────────────────────────────
    data = df_train[var + [target_var]].copy()
    data["target"] = data[target_var].copy()

    # ── Scaling ───────────────────────────────────────────────────────────────
    # scaler_X is fitted on [var1, ..., calibration] (original column order).
    # NOTE: split_sequences reorders the encoder input to [calibration, var1, ...].
    # For external inference, always transform with this original order,
    # then reorder manually before passing to the model.
    scaler_X = _make_scaler(ext_norm_method)
    scaler_y = _make_scaler(ext_norm_method)

    if scaler_X is not None:
        data[var + [target_var]] = scaler_X.fit_transform(data[var + [target_var]])
    if scaler_y is not None:
        data["target"] = scaler_y.fit_transform(data[["target"]])

    # ── Sliding window generation ─────────────────────────────────────────────
    X_full, decX_full, y_full = split_sequences(
        data.values, n_steps, n_outputs, stride,
        shuffle_windows=False, start_token=None,
    )

    split        = int(len(X_full) * (1 - val_split_ratio))
    X_train,    X_val    = X_full[:split],    X_full[split:]
    decX_train, decX_val = decX_full[:split], decX_full[split:]
    y_train,    y_val    = y_full[:split],    y_full[split:]

    # X_train / X_val are views of X_full, so the underlying memory is not
    # released until TF copies the data into the dataset. The first del removes
    # the Python reference to X_full so the GC can act after the second del.
    del X_full, decX_full, y_full
    gc.collect()

    # ── tf.data.Dataset pipeline ──────────────────────────────────────────────
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, decX_train, y_train))
    if shuffle:
        train_ds = train_ds.shuffle(
            buffer_size=min(8192, len(X_train)), seed=CONFIG["seed"]
        )
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds   = (tf.data.Dataset.from_tensor_slices((X_val, decX_val, y_val))
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

    # Free NumPy splits after TF has copied data into the datasets
    del X_train, decX_train, y_train, X_val, decX_val, y_val
    gc.collect()

    # ── Model, optimizer, and loss ────────────────────────────────────────────
    model   = Seq2Seq(enc_units, dec_units, output_dim=1,
                      n_outputs=n_outputs, encoder_batchnorm=internal_norm)
    opt_obj = _build_optimizer(optimizer, learning_rate)
    loss_fn = _build_loss(loss)

    # Resolve minimum improvement threshold (None maps to 0.0)
    _min_delta = min_delta if min_delta is not None else 0.0

    if monitor not in ("val_loss", "loss"):
        raise ValueError(f"monitor must be 'val_loss' or 'loss', got: '{monitor}'")

    # ── Graph-compiled train and validation steps ─────────────────────────────
    # @tf.function eliminates eager dispatch overhead per batch: the function is
    # traced once and executed as a native TF graph on subsequent calls.
    # forcing_ratio is passed as a tf.Tensor (not a Python float) to prevent
    # retracing of the graph each epoch when its value changes.
    @tf.function
    def train_step(Xb, decb, yb, forcing_ratio_tf):
        with tf.GradientTape() as tape:
            preds      = model(Xb, decb, training=True,
                               forcing_ratio=forcing_ratio_tf, per_sample=per_sample)
            preds_loss = tf.squeeze(preds, axis=-1) if preds.shape[-1] == 1 else preds
            loss_val   = loss_fn(yb, preds_loss)
        grads = tape.gradient(loss_val, model.trainable_variables)
        opt_obj.apply_gradients(zip(grads, model.trainable_variables))
        return loss_val

    @tf.function
    def val_step(Xb, decb, yb):
        preds      = model(Xb, decb, training=False)
        preds_loss = tf.squeeze(preds, axis=-1) if preds.shape[-1] == 1 else preds
        return loss_fn(yb, preds_loss)

    # ── Training loop ─────────────────────────────────────────────────────────
    history          = {"loss": [], "val_loss": []}
    best_monitored   = np.inf
    patience_counter = 0
    best_weights     = None

    for epoch in range(epochs):
        # Compute teacher forcing ratio according to the selected mode
        if mode == "forced":
            forcing_ratio = 1.0
        elif mode == "recursive":
            forcing_ratio = 0.0
        elif mode == "mixed":
            decay         = max(0.0, (decay_epochs - epoch) / max(1, decay_epochs))
            forcing_ratio = float(end_forcing + (start_forcing - end_forcing) * decay)
        else:
            raise ValueError(f"Invalid mode: '{mode}'. Use 'forced' | 'recursive' | 'mixed'.")

        # Wrap forcing_ratio in tf.constant to avoid graph retracing when its
        # float value changes between epochs (the TF function signature stays fixed).
        forcing_ratio_tf = tf.constant(forcing_ratio, dtype=tf.float32)
        train_losses = [
            float(train_step(Xb, decb, yb, forcing_ratio_tf).numpy())
            for Xb, decb, yb in train_ds
        ]

        # Validation step — graph-compiled
        val_losses = [
            float(val_step(Xb, decb, yb).numpy())
            for Xb, decb, yb in val_ds
        ]

        avg_train = np.mean(train_losses)
        avg_val   = np.mean(val_losses)
        history["loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        # Early stopping with best-weight restoration
        monitored_value = avg_val if monitor == "val_loss" else avg_train
        if monitored_value < best_monitored - _min_delta:
            best_monitored   = monitored_value
            patience_counter = 0
            best_weights     = model.get_weights()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if best_weights is not None:
                    model.set_weights(best_weights)
                print(f"Early stopping at epoch {epoch + 1} "
                      f"(monitor='{monitor}', min_delta={_min_delta})")
                break

    # ── Save weights and scalers (optional) ───────────────────────────────────
    if save_weights:
        if results_dir is None:
            raise ValueError("results_dir is required when save_weights=True.")
        xtal_str    = str(xtal_id) if xtal_id is not None else "xtal_unknown"
        norm_str    = ext_norm_method.lower()
        weights_dir = os.path.join(
            results_dir,
            f"{xtal_str}_seq2seq_arch1_{mode}_{n_steps}_{n_outputs}_{norm_str}"
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
    scaler_X, scaler_y, start_token=None, target_var="calibration",
    time_col="laser_datetime",
):
    """
    Internal helper: scale, window, predict, and align predictions on the
    time axis. Overlapping predictions at the same time index are averaged.

    Args:
        model      : trained Seq2Seq model.
        df_test    : test DataFrame with the same columns used during training.
        var        : list of predictor column names.
        n_steps    : look-back window size (encoder input length).
        n_outputs  : forecast horizon (decoder output length).
        stride     : step size between windows.
        scaler_X   : fitted scaler for input features (or None).
        scaler_y   : fitted scaler for the target column (or None).
        start_token: initial decoder token. None → last known target value.
        target_var : name of the target column.
        time_col   : column used for temporal sorting.

    Returns:
        Tuple (y_pred, y_true, valid_time) on the original (unscaled) scale.

    Raises:
        ValueError: If the test set is too short or no valid predictions are generated.
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
    y_col      = test_array.shape[1] - 1
    if len(test_array) < n_steps + n_outputs:
        raise ValueError("Test set is too short to generate any windows.")

    X_test, decX_test, _ = split_sequences(
        test_array, n_steps, n_outputs, stride, start_token=start_token
    )

    if X_test.size == 0:
        raise ValueError(
            f"No test windows were generated (X_test is empty). "
            f"Check that the test set has enough samples for "
            f"n_steps + n_outputs = {n_steps + n_outputs} with stride = {stride}."
        )

    yhat = model(X_test, decX_test, training=False).numpy()
    if yhat.shape[-1] == 1:
        yhat = yhat.squeeze(-1)

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
            y_true.append(test_array[i, y_col])
            valid_time.append(df_test[time_col].values[i])

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    if y_pred.size == 0:
        raise ValueError(
            "No valid predictions after reconstructing overlapping windows. "
            "Check that n_steps, n_outputs, and stride are consistent with "
            "the size of the test set."
        )

    # Inverse-transform both arrays back to the original calibration scale
    if scaler_y is not None:
        y_true = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    return y_pred, y_true, np.array(valid_time)


def evaluate_model(
    model, history, df_test, var, n_steps, n_outputs=1, stride=1,
    scaler_X=None, scaler_y=None, start_token=None,
    target_var="calibration", time_col="laser_datetime",
    metrics: list = None,
):
    """
    Evaluate the model on the test set and return numeric metrics. No plots.

    Args:
        model      : trained Seq2Seq model.
        history    : training history dict {"loss": [...], "val_loss": [...]}.
        df_test    : test DataFrame.
        var        : list of predictor column names.
        n_steps    : look-back window size.
        n_outputs  : forecast horizon.
        stride     : step between windows.
        scaler_X   : fitted input scaler (or None).
        scaler_y   : fitted target scaler (or None).
        start_token: initial decoder token (see _prepare_predictions).
        target_var : name of the target column.
        time_col   : column used for temporal sorting.
        metrics    : list of metrics to compute (see _compute_metrics).
                     None → only "mape" (backward-compatible default).

    Returns:
        Tuple (mape, y_true, y_pred, metrics_dict).
    """
    y_pred, y_true, _ = _prepare_predictions(
        model, df_test, var, n_steps, n_outputs, stride,
        scaler_X, scaler_y, start_token, target_var, time_col,
    )

    _metrics = metrics if metrics is not None else ["mape"]
    results  = _compute_metrics(y_true, y_pred, _metrics)

    for name, value in results.items():
        print(f"  {name.upper():6s}: {value:.4f}")

    mape = results.get("mape", 100 * mean_absolute_percentage_error(y_true, y_pred))
    return mape, y_true, y_pred, results


def evaluate_and_plot_model(
    model, history, df_test, var, n_steps, n_outputs=1, stride=1,
    scaler_X=None, scaler_y=None, crystal_id=None, start_token=None,
    target_var="calibration", plot_ratio=True,
    results_dir=None, time_col="laser_datetime",
    metrics: list = None,
    loss: str = None,
):
    """
    Evaluate the model, compute metrics, and generate diagnostic plots:
        Figure 1 — Training vs validation loss curve.
        Figure 2 — Predicted vs actual values (with optional ratio subplot).

    Note: history is expected to be a plain dict {"loss": [...], "val_loss": [...]}
    as returned by train_seq2seq (not a Keras History object).

    Args:
        model       : trained Seq2Seq model.
        history     : training history dict {"loss": [...], "val_loss": [...]}.
        df_test     : test DataFrame.
        var         : list of predictor column names.
        n_steps     : look-back window size.
        n_outputs   : forecast horizon.
        stride      : step between windows.
        scaler_X    : fitted input scaler (or None).
        scaler_y    : fitted target scaler (or None).
        crystal_id  : crystal identifier for plot titles and file names.
        start_token : initial decoder token (see _prepare_predictions).
        target_var  : name of the target column.
        plot_ratio  : if True, adds a lower subplot showing the true/pred ratio.
        results_dir : directory to save figures (None = do not save).
        time_col    : column used for temporal sorting.
        metrics     : list of metrics to compute (see _compute_metrics).
                      None → ["mape", "smape", "mae", "rmse", "maxae", "r2"].
        loss        : loss function name used during training; used only to label
                      the Y-axis of the loss curve. None → "mse".

    Returns:
        Tuple (mape, y_true, y_pred).
    """
    y_pred, y_true, valid_time = _prepare_predictions(
        model, df_test, var, n_steps, n_outputs, stride,
        scaler_X, scaler_y, start_token, target_var, time_col,
    )

    _metrics    = metrics if metrics is not None else ["mape", "smape", "mae", "rmse", "maxae", "r2"]
    results     = _compute_metrics(y_true, y_pred, _metrics)
    # Build a compact metric string for the plot title
    metrics_str = "  |  ".join(
        f"{name.upper()}: {value:.4f}" for name, value in results.items()
    )
    mape = results.get("mape", 100 * mean_absolute_percentage_error(y_true, y_pred))

    # ── Figure 1: loss curve ──────────────────────────────────────────────────
    loss_label = (loss or "mse").upper()
    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    ax_loss.plot(history["loss"],     label="Train")
    ax_loss.plot(history["val_loss"], label="Validation")
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
config_details_global      = {}


def _config_key(r: dict) -> tuple:
    """
    Generate a hashable key that uniquely identifies a model configuration.

    All hyperparameters that the grid can vary are included to avoid collisions:
    two runs differing only in optimizer, loss, monitor, or min_delta would
    otherwise accumulate into the same entry.

    Args:
        r: Result dictionary produced by a single grid run.

    Returns:
        Tuple used as dictionary key in the global accumulators.
    """
    return (
        tuple(r["var"]),
        tuple(r["enc_units"]),
        tuple(r["dec_units"]),
        r["mode"],
        r["start_forcing"],
        r["end_forcing"],
        r["decay_epochs"],
        r["per_sample"],
        r["learning_rate"],
        r["ext_norm_method"],
        r["internal_norm"],
        r["shuffle"],
        r.get("optimizer",  "adam"),      # prevents collision when optimizer varies
        r.get("loss",       "mse"),       # prevents collision when loss varies
        r.get("monitor",    "val_loss"),  # prevents collision when monitor varies
        r.get("min_delta",  0.0),         # prevents collision when min_delta varies
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
            (var, enc_units, dec_units, mode, start_forcing,
             end_forcing, decay_epochs, per_sample, learning_rate,
             ext_norm_method, internal_norm, shuffle,
             optimizer_k, loss_k, monitor_k, min_delta_k) = key

            det = config_details_global[key]
            f.write("Configuration:\n")
            f.write(f"  Variables:   {var}\n")
            f.write(f"  enc_units:   {enc_units}  |  dec_units: {dec_units}\n")
            f.write(f"  mode:        {mode}\n")
            f.write(f"  forcing:     start={start_forcing}, end={end_forcing}, "
                    f"decay_epochs={decay_epochs}\n")
            f.write(f"  per_sample:  {per_sample}  |  lr: {learning_rate}\n")
            f.write(f"  optimizer:   {optimizer_k}  |  loss: {loss_k}\n")
            f.write(f"  monitor:     {monitor_k}  |  min_delta: {min_delta_k}\n")
            f.write(f"  norm:        {ext_norm_method}  |  batchnorm: {internal_norm}"
                    f"  |  shuffle: {shuffle}\n")

            for metric_name in list(det["metrics"].keys()):
                cumulative = totals.get(metric_name, 0.0)
                f.write(f"  {metric_name.upper()} cumulative: {cumulative:.4f}\n")
                f.write("  Per-horizon detail:\n")

                values_list    = det["metrics"][metric_name]
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
        df_train    : Training DataFrame with engineered features.
        df_test     : Test DataFrame with engineered features.
        cfg         : Global configuration dictionary (see CONFIG).
        output_file : Path to the output results file. Defaults to
                      cfg['results_dir'] / cfg['results_file'].
    """
    output_file   = output_file or os.path.join(cfg["results_dir"], cfg["results_file"])
    g             = cfg["grid"]
    gf            = cfg["grid_fixed"]
    horizons_list = gf["horizons"]

    # Build the Cartesian product of all grid parameters
    combinations = list(product(
        g["rnn_units"],
        g["variables"],
        g["norm_method"],
        g["internal_norm"],
        g["shuffle"],
        g["mode"],
        g["per_sample"],
        g.get("optimizer",  [None]),
        g["learning_rate"],
        g.get("loss",       [None]),
        g.get("monitor",    ["val_loss"]),
        g.get("min_delta",  [None]),
    ))

    for a in horizons_list:
        n_steps   = a
        n_outputs = a
        stride    = a

        # batch_size: None → dynamic based on horizon size
        batch = gf["batch_size"]
        if batch is None:
            batch = 32 if n_steps >= 48 else 128

        for i, (rnn_units, cur_var, norm_key, batchnorm,
                shuffle_flag, mode, per_sample,
                optimizer_name, learning_rate, loss_name,
                monitor_metric, min_delta_val) in enumerate(combinations):
            try:
                reset_environment(seed=cfg["seed"], use_cpu=gf.get("use_cpu", False))

                # Fixed parameters from grid_fixed
                epochs        = gf["epochs"]
                val_split     = gf["val_split"]
                patience      = gf["patience"]
                start_forcing = gf["start_forcing"]
                end_forcing   = gf["end_forcing"]
                decay_epochs  = gf["decay_epochs"]

                model, history, scaler_X, scaler_y = train_seq2seq(
                    df_train=df_train,
                    var=cur_var,
                    target_var=cfg["target_var"],

                    n_steps=n_steps,
                    n_outputs=n_outputs,
                    stride=stride,

                    enc_units=rnn_units,
                    dec_units=rnn_units,

                    epochs=epochs,
                    batch_size=batch,
                    patience=patience,

                    mode=mode,
                    per_sample=per_sample,
                    optimizer=optimizer_name,
                    learning_rate=learning_rate,
                    loss=loss_name,

                    ext_norm_method=norm_key,
                    internal_norm=batchnorm,
                    shuffle=shuffle_flag,

                    val_split_ratio=val_split,
                    monitor=monitor_metric,
                    min_delta=min_delta_val,

                    start_forcing=start_forcing,
                    end_forcing=end_forcing,
                    decay_epochs=decay_epochs,

                    save_weights=gf["save_weights"],
                    results_dir=cfg["results_dir"],
                    xtal_id=cfg["xtal_id"],
                    time_col=cfg["time_col"],
                )

                mape, _, _, results_dict = evaluate_model(
                    model, history, df_test, cur_var,
                    n_steps=n_steps,
                    n_outputs=n_outputs,
                    stride=stride,
                    scaler_X=scaler_X,
                    scaler_y=scaler_y,
                    metrics=gf.get("test_metrics", ["mape"]),
                )

                accumulate_results([{
                    "var":            cur_var,
                    "n_steps":        n_steps,
                    "n_outputs":      n_outputs,
                    "stride":         stride,
                    "enc_units":      rnn_units,
                    "dec_units":      rnn_units,
                    "batch_size":     batch,
                    "mode":           mode,
                    "start_forcing":  start_forcing,
                    "end_forcing":    end_forcing,
                    "decay_epochs":   decay_epochs,
                    "per_sample":     per_sample,
                    "optimizer":      optimizer_name or "adam",
                    "learning_rate":  learning_rate,
                    "loss":           loss_name or "mse",
                    "monitor":        monitor_metric,
                    "min_delta":      min_delta_val if min_delta_val is not None else 0.0,
                    "ext_norm_method": norm_key,
                    "internal_norm":  batchnorm,
                    "shuffle":        shuffle_flag,
                    "evaluated_metrics": results_dict,
                }])

                print_results_txt(gf.get("reference_metrics", {}), output_file)

            except Exception:
                print(f"[ERROR] config {i} | horizon {a}:")
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

    model, history, scaler_X, scaler_y = train_seq2seq(
        df_train=df_train,
        var=d["variables"],
        target_var=cfg["target_var"],

        n_steps=d["n_steps"],
        n_outputs=d["n_outputs"],
        stride=d["stride"],

        enc_units=d["rnn_units"],
        dec_units=d["rnn_units"],

        epochs=d["epochs"],
        batch_size=batch,

        mode=d["mode"],
        per_sample=d["per_sample"],
        optimizer=d.get("optimizer", None),
        learning_rate=d["learning_rate"],
        loss=d.get("loss", None),

        ext_norm_method=d["norm_method"],
        internal_norm=d["internal_norm"],
        shuffle=d["shuffle"],

        val_split_ratio=d["val_split"],
        patience=d["patience"],
        monitor=d.get("monitor", "val_loss"),
        min_delta=d.get("min_delta", None),

        start_forcing=d.get("start_forcing", 1.0),
        end_forcing=d.get("end_forcing", 0.0),
        decay_epochs=d.get("decay_epochs", 30),

        save_weights=d.get("save_weights", False),
        results_dir=cfg["results_dir"],
        xtal_id=cfg["xtal_id"],
        time_col=cfg["time_col"],
    )

    evaluate_and_plot_model(
        model=model,
        history=history,
        df_test=df_test,
        var=d["variables"],

        n_steps=d["n_steps"],
        n_outputs=d["n_outputs"],
        stride=d["stride"],

        scaler_X=scaler_X,
        scaler_y=scaler_y,

        crystal_id=cfg["xtal_id"],
        plot_ratio=True,
        results_dir=cfg["results_dir"],
        time_col=cfg["time_col"],
        metrics=d.get("test_metrics", None),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 12. LOADING A SAVED MODEL
# ──────────────────────────────────────────────────────────────────────────────

def load_model(cfg=CONFIG):
    """
    Reconstruct a previously saved Seq2Seq model (save_weights=True).

    All configuration is taken from cfg['default']. Loads from:
        <results_dir>/<xtal_id>_seq2seq_arch1_<mode>_<n_steps>_<n_outputs>_<norm>/model_<n_steps>steps.weights.h5
        <results_dir>/<xtal_id>_seq2seq_arch1_<mode>_<n_steps>_<n_outputs>_<norm>/scaler_X_<n_steps>steps.pkl
        <results_dir>/<xtal_id>_seq2seq_arch1_<mode>_<n_steps>_<n_outputs>_<norm>/scaler_y_<n_steps>steps.pkl
        <results_dir>/<xtal_id>_seq2seq_arch1_<mode>_<n_steps>_<n_outputs>_<norm>/n_vars.txt

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
    tf.keras.backend.clear_session()
    d         = cfg["default"]
    n_steps   = d["n_steps"]
    n_outputs = d["n_outputs"]
    xtal_str  = str(cfg["xtal_id"])
    mode_str  = d["mode"]
    norm_str  = d["norm_method"].lower()
    weights_dir = os.path.join(
        cfg["results_dir"],
        f"{xtal_str}_seq2seq_arch1_{mode_str}_{n_steps}_{n_outputs}_{norm_str}"
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
    model = Seq2Seq(
        encoder_units=d["rnn_units"],
        decoder_units=d["rnn_units"],
        output_dim=1,
        n_outputs=n_outputs,
        encoder_batchnorm=d["internal_norm"],
    )

    # ── Encoder input column order for external inference ────────────────────
    # 1. Build the array in the original order [var1, ..., calibration]
    # 2. Apply scaler_X.transform() with that order
    # 3. Reorder to [calibration, var1, ...] before passing to the encoder
    # 4. The decoder receives [target, var1, ...] without reordering
    n_features_enc = len(d["variables"]) + 1   # [calibration, var1, var2, ...]
    n_features_dec = len(d["variables"]) + 1   # [target, var1, var2, ...]

    # Dummy forward pass to initialize all layer weights before loading saved values
    dummy_enc = tf.zeros((1, n_steps, n_features_enc))
    dummy_dec = tf.zeros((1, n_steps, n_features_dec))
    model(dummy_enc, dummy_dec)

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
     run_single(df_train, df_test, CONFIG)          # ← default mode

    # ── Option B: run a full grid search ──────────────────────────────────────
     #run_grid_search(df_train, df_test, CONFIG)     # ← uncomment for grid search

    # ── Option C: load a previously saved model and evaluate ──────────────────
    # Requires a prior run with save_weights=True in CONFIG['default'].
    # Expected folder example: <results_dir>/30600_seq2seq_arch1_forced_48_48_minmax/
'''
    _, df_test = prepare_splits(CONFIG)
    model, scaler_X, scaler_y = load_model(cfg=CONFIG)
    _d = CONFIG["default"]
    evaluate_and_plot_model(
        model=model, history={"loss": [], "val_loss": []},
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