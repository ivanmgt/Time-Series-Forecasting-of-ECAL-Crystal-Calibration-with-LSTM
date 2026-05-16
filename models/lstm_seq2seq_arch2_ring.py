# =============================================================================
# lstm_seq2seq_arch2_ring.py
# LSTM Seq2Seq neural network (decoder without exogenous variables) for
# multi-crystal forecasting over multiple rings.
# Adaptable to a different ring set by modifying CONFIG.
# Important note: due to the massive data volume and the architecture design,
# this model requires a professional GPU such as the A5000 or L40S.
# =============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# 0. CENTRAL CONFIGURATION — the only place to modify when adapting to a new dataset
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Paths (add/remove rings as needed) ───────────────────────────────────
    "data_sources": {
        "ring_1":   "/data/plus_z_1/ring_1.csv",
        # "ring_m1":  "/data/minus_z_1/ring_-1.csv",
        # "ring_50":  "/data/plus_z_6/ring_50.csv",
        # "ring_m36": "/data/minus_z_4/ring_-36.csv",
    },
    "active_ring":  "ring_1",          # which of the above sources to use
    "results_dir":  "/results",
    "results_file": "results_seq2seq_arch2_ring_v2.txt",

    # ── Data filtering ───────────────────────────────────────────────────────
    "target_var":   "calibration",   # target column to predict
    "time_col":     "laser_datetime",
    "calib_min":    0.7,             # minimum accepted target value
    "calib_max":    1.0,             # maximum accepted target value
    "train_year":   [2016],          # list of years used for training
    "test_year":    2017,            # year used for external evaluation

    # ── Columns to retain after loading ─────────────────────────────────────
    "keep_cols": ["xtal_id", "calibration", "int_deliv_inv_ub",
                  "laser_datetime", "time"],

    # ── Reproducibility ──────────────────────────────────────────────────────
    "seed": 1234,

    # =========================================================================
    # DEFAULT CONFIGURATION (run_single)
    # =========================================================================
    "default": {
        # ── Predictor variables ──────────────────────────────────────────────
        "variables":      ["delta_lumi"],   # var6
        # n_steps == n_outputs == stride for non-overlapping windows
        "n_steps":        48,
        "n_outputs":      48,
        "stride":         48,

        # ── Architecture ─────────────────────────────────────────────────────
        # enc_units and dec_units are derived from rnn_units in run_single and
        # in the grid search, but can be overridden separately if needed.
        "rnn_units":      [512, 512],  # shared units for encoder and decoder
        "internal_norm":  False,       # apply BatchNormalization after each LSTM layer

        # ── Training ─────────────────────────────────────────────────────────
        "epochs":         1,
        # batch_size: None → dynamic (32 if n_steps >= 48, 128 if n_steps < 48)
        "batch_size":     None,

        # ── Optimizer ────────────────────────────────────────────────────────
        # Options: "adam" | "adamw" | "sgd" | "rmsprop" | "adagrad"
        # None → use "adam" (library default)
        "optimizer":      None,
        # Optimizer learning rate
        "learning_rate":  1e-3,

        # ── Loss function ────────────────────────────────────────────────────
        # Options: "mse" | "mae" | "huber" | "logcosh"
        # None → use "mse"
        # mse     : strongly penalizes large errors; standard for regression.
        # mae     : more robust to outliers; converges more slowly.
        # huber   : hybrid mse/mae; mse for small errors, mae for large ones.
        # logcosh : similar to Huber but differentiable over the entire domain.
        "loss":           None,

        # ── Early stopping ────────────────────────────────────────────────────
        "patience":       20,   # number of epochs without improvement before stopping
        # Training history metric to monitor: "val_loss" or "loss"
        "monitor":        "val_loss",
        # Minimum improvement required to reset the patience counter.
        # None → 0.0 (any improvement counts, no minimum threshold)
        "min_delta":      None,

        # ── Data normalization ───────────────────────────────────────────────
        # Options: "MinMax" | "Standard" | "PowerTransformer" | "None"
        "norm_method":    "MinMax",

        # ── Train/validation split ───────────────────────────────────────────
        "val_split":      0.1,   # fraction of data used for validation (0–1)

        # ── Shuffle windows before training ──────────────────────────────────
        # True  → shuffles the tf.data.Dataset AND the pre-split permutation
        # False → no shuffling (preserves temporal order)
        "shuffle":        True,

        # ── Teacher forcing mode ──────────────────────────────────────────────
        # "forced"    → decoder always receives the ground truth
        # "recursive" → decoder always uses its own prediction
        # "mixed"     → gradual transition from forced to recursive
        "mode":           "forced",
        # The following three parameters only apply when mode=="mixed":
        "start_forcing":  1.0,
        "end_forcing":    0.0,
        "decay_epochs":   30,
        # Scheduled sampling per sample (True) vs. per sequence (False)
        "per_sample":     True,

        # ── Hardware ──────────────────────────────────────────────────────────
        # True  → force CPU usage (reproducible, slower)
        # False → use GPU if available
        "use_cpu":        False,

        # ── Per-crystal evaluation metrics (run_single → evaluate_and_plot_model)
        # Computed on the original scale for the crystal selected by plot_xtal_id.
        # Used only in run_single; the grid does not evaluate individual crystals.
        # Available: "mape" | "smape" | "mae" | "rmse" | "maxae" | "r2"
        # - mape  : Mean Absolute Percentage Error (×100). Main project metric.
        # - smape : Symmetric MAPE; more robust when target values approach zero.
        # - mae   : Mean Absolute Error; in the same units as the target.
        # - rmse  : Root Mean Squared Error; penalizes large errors.
        # - maxae : Maximum absolute error; indicates the worst individual case.
        # - r2    : Coefficient of determination (1=perfect, 0=mean, <0=worse).
        "test_metrics":   ["mape", "smape"],

        # ── Ring evaluation metrics (run_single → evaluate_all_xtals) ────
        # Computed by aggregating predictions across ALL crystals in the ring.
        # Unlike test_metrics (single crystal), these measure global model
        # behavior over the entire ring.
        # Available: "wmape_pond" | "mae_global" | "rmse_global"
        # - wmape_pond  : True per-crystal WMAPE, weighted by number of observations.
        #                 More robust than the average of individual MAPEs when
        #                 true values are small.
        # - mae_global  : MAE over all crystals concatenated.
        # - rmse_global : RMSE over all crystals concatenated.
        "ring_metrics":   ["wmape_pond", "mae_global", "rmse_global"],

        # ── Saving ───────────────────────────────────────────────────────────
        "save_weights":   True,
        "plot_xtal_id":   30600,
    },

    # =========================================================================
    # FIXED GRID PARAMETERS (not combined, applied to all runs)
    # =========================================================================
    "grid_fixed": {
        # ── Training epochs ──────────────────────────────────────────────────
        "epochs":         1,
        # ── Forecast horizons to evaluate ────────────────────────────────────
        "horizons":       [1, 12, 24, 36, 48, 60, 72, 84, 96],
        # ── ring evaluation metrics (fixed for the entire grid) ──────────
        # Available: "wmape_pond" | "mae_global" | "rmse_global"
        # - wmape_pond  : True per-crystal WMAPE, weighted by number of observations.
        # - mae_global  : MAE over all crystals concatenated.
        # - rmse_global : RMSE over all crystals concatenated.
        "ring_metrics":   ["wmape_pond", "mae_global"],
        # ── Reference metric values per horizon (baseline to outperform) ───────
        # Ring metrics (weighted WMAPE), not individual crystal metrics.
        "reference_metrics": {
            "wmape_pond": {
                1: 0.075, 12: 0.209, 24: 0.246, 36: 0.278,
                48: 0.319, 60: 0.345, 72: 0.377, 84: 0.402, 96: 0.427,
            },
            "mae_global": {
                # Replace with actual MAE reference values (these are placeholders)
                1: 1, 12: 1, 24: 1, 36: 1,
                48: 1, 60: 1, 72: 1, 84: 1, 96: 1,
            }
            
        },
        # batch_size: None → dynamic (32 if n_steps >= 48, 128 if n_steps < 48)
        "batch_size":     None,
        # ── Parámetros de entrenamiento fijos para el grid ────────────────────
        "val_split":      0.1,
        "patience":       20,
        "start_forcing":  1.0,
        "end_forcing":    0.0,
        "decay_epochs":   30,
        # ── Saving ───────────────────────────────────────────────────────────
        # True  → save model weights to results_dir for each horizon.
        # False → train without saving (faster for grid exploration).
        "save_weights":   False,
        # ── Hardware ─────────────────────────────────────────────────────────
        "use_cpu":        False,

    },

    # =========================================================================
    # GRID CONFIGURATIONS
    # Each entry is a dict with the specific parameters for that run.
    # _run_one_config iterates each entry over all horizons in grid_fixed.
    # Fields not specified inherit from CONFIG["default"].
    # =========================================================================
    "grid_configs": [
        # ── config 14 ─────────────────────────────────────────────────────────
        {
            "id":           14,
            "rnn_units":    [64, 64],   # units for encoder and decoder
            "variables":    "var6",     # key in VARIABLE_SETS
            "shuffle":      True,       # shuffle training windows
            "mode":         "forced",   # teacher forcing mode
            # The following fields are optional; if omitted they inherit from default
            # "per_sample":   True,
            # "learning_rate": 1e-3,
            # "optimizer":    None,     # None → "adam"
            # "loss":         None,     # None → "mse"
            # "norm_method":  "MinMax",
            # "internal_norm": False,
            # "monitor":      "val_loss",
            # "min_delta":    None,
        },
        # ── config 55 ─────────────────────────────────────────────────────────
        # {"id": 55, "rnn_units": [512, 512], "variables": "var5", "shuffle": True,  "mode": "forced"},
        # ── config 61 ─────────────────────────────────────────────────────────
        # {"id": 61, "rnn_units": [512, 512], "variables": "var6", "shuffle": False, "mode": "forced"},
        # ── config 26 ─────────────────────────────────────────────────────────
        # {"id": 26, "rnn_units": [128, 128], "variables": "var5", "shuffle": False, "mode": "recursive"},
        # ── config 27 ─────────────────────────────────────────────────────────
        # {"id": 27, "rnn_units": [128, 128], "variables": "var6", "shuffle": True, "mode": "forced"},
        # ── config 29 ─────────────────────────────────────────────────────────
        # {"id": 29, "rnn_units": [128, 128], "variables": "var6", "shuffle": False, "mode": "forced"},
        # ── config 40 ─────────────────────────────────────────────────────────
        # {"id": 40, "rnn_units": [256, 256], "variables": "var5", "shuffle": False, "mode": "forced"},
        # ── config 11 ─────────────────────────────────────────────────────────
        # {"id": 11, "rnn_units": [62, 64], "variables": "var5", "shuffle": True, "mode": "recursive"},
        # ── config 64 ─────────────────────────────────────────────────────────
        # {"id": 64, "rnn_units": [512, 512], "variables": "var5", "shuffle": True, "mode": "forced", "norm_method": "Standard"},
        # ── config 83 ─────────────────────────────────────────────────────────
        # {"id": 83, "rnn_units": [1024, 1024], "variables": "var5", "shuffle": True, "mode": "forced", "norm_method": "Standard"},
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
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, LSTM, LSTMCell
)
from tensorflow.keras.models import Model

# ── GPU memory growth (must be configured before any other TF operation) ──────
# Prevents TF from reserving all VRAM at startup → avoids OOM in long sessions
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[GPU] Memory growth enabled on {len(gpus)} GPU(s)")

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
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    _set_env_vars(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    print(f"Environment reset — seed: {seed}")


# ── Initialization upon import ───────────────────────────────────────────────
_seed = CONFIG["seed"]
_set_env_vars(_seed)
random.seed(_seed)
np.random.seed(_seed)
tf.random.set_seed(_seed)
tf.keras.utils.set_random_seed(_seed)
tf.config.experimental.enable_op_determinism()


# ──────────────────────────────────────────────────────────────────────────────
# 3. DATA LOADING AND PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def load_data(cfg: dict = CONFIG) -> pd.DataFrame:
    """
    Load the active ring CSV, filter by calibration range,
    parse date columns, and sort by (xtal_id, time_col).
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

    Applies .diff() independently for each xtal_id, preserving the original
    behavior and avoiding cross-crystal contamination.
    """
    fragments = []
    for xtal in df["xtal_id"].unique():
        i = df[df["xtal_id"] == xtal].copy().reset_index(drop=True)

        dt_sec = i["time"].diff().dt.total_seconds()
        d_lum  = i["int_deliv_inv_ub"].diff()

        i["dint_dtime"]   = d_lum / dt_sec
        i["delta_lumi"]   = d_lum
        i["delta_t"]      = i["laser_datetime"].diff().dt.total_seconds()
        i["days_no_int"]  = i["time"].diff().dt.days

        i.loc[0, "dint_dtime"] = i.loc[1, "dint_dtime"]
        i.loc[i["days_no_int"] > 0,  "dint_dtime"]               = 0
        i.loc[0, ["days_no_int", "delta_t", "delta_lumi"]]       = 0
        i.loc[i["days_no_int"] > 30, ["delta_lumi", "delta_t"]]  = 0
        i["dint_dtime"] = i["dint_dtime"].bfill().ffill()
        i.loc[~np.isfinite(i["dint_dtime"]), "dint_dtime"]       = 0

        fragments.append(i)

    return pd.concat(fragments, ignore_index=True)


def prepare_splits(cfg: dict = CONFIG):
    """
    Load data, apply feature engineering per year, and return
    (df_train, df_test) with all crystals in the ring.
    """
    df       = load_data(cfg)
    time_col = cfg["time_col"]
    yr       = df[time_col].dt.year

    # Accept train_year as a scalar or a list (guards against CONFIG typos)
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
            f"has data in CSV '{cfg['data_sources'][cfg['active_ring']]}'."
        )
    if len(df_test) == 0:
        raise ValueError(
            f"df_test is empty. Check that 'test_year'={cfg['test_year']} "
            f"has data in CSV '{cfg['data_sources'][cfg['active_ring']]}'."
        )

    return df_train, df_test


# ──────────────────────────────────────────────────────────────────────────────
# 4. VARIABLE GROUPS (adaptable)
# ──────────────────────────────────────────────────────────────────────────────
VARIABLE_SETS = {
    "var1": ["int_deliv_inv_ub", "dint_dtime"],
    "var2": ["int_deliv_inv_ub", "dint_dtime", "delta_lumi"],
    "var3": ["int_deliv_inv_ub", "dint_dtime", "delta_t"],
    "var4": ["int_deliv_inv_ub"],
    "var5": ["dint_dtime"],
    "var6": ["delta_lumi"],
}


# ──────────────────────────────────────────────────────────────────────────────
# 5. WINDOWING FOR SEQ2SEQ ARCHITECTURE 2 (decoder without exogenous variables)
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
    Generate sliding windows for Seq2Seq arch2 (decoder without exogenous variables)
    
    Expected column layout in `sequences`: [var1, ..., calibration, target]
        ─ calibration: second-to-last column by default (or pass calib_col)
        ─ target:      last column by default            (or pass y_col)

    Returns:
        X      (N, n_steps,   1+n_vars)       ← [calibration, vars...] for encoder
        dec_X  (N, n_outputs, 1)              ← [target]      for decoder
        y      (N, n_outputs)                 ← future target values
    """
    assert sequences.shape[1] >= 2, \
        "At least 2 columns are required (features + target)."

    X_list, dec_list, y_list = [], [], []
    T      = sequences.shape[0]
    n_cols = sequences.shape[1]

    y_col     = y_col     if y_col     is not None else n_cols - 1
    calib_col = calib_col if calib_col is not None else n_cols - 2

    assert y_col != calib_col,          "calibration and target cannot be the same column."
    assert 0 <= calib_col < n_cols,     f"calib_col={calib_col} out of range."
    assert 0 <= y_col     < n_cols,     f"y_col={y_col} out of range."


    for i in range(0, T - n_steps - n_outputs + 1, stride):
        end_ix  = i + n_steps
        out_end = end_ix + n_outputs

        # ── Encoder: no reordering, just exclude y_col ────────────────────────
        enc_cols = [c for c in range(n_cols) if c != y_col]
        seq_x    = sequences[i:end_ix, enc_cols]
        
        # ── Targets ───────────────────────────────────────────────────────────
        seq_y = sequences[end_ix:out_end, y_col]
        
        # ── Decoder input: ONLY the target ────────────────────────────────────
        dec_in = np.zeros((n_outputs, 1), dtype=sequences.dtype)
        last_row = sequences[end_ix - 1]
        
        if start_token is None:
            dec_in[0, 0] = last_row[y_col]
        else:
            dec_in[0, 0] = start_token
        
        if n_outputs > 1:
            dec_in[1:, 0] = sequences[end_ix:out_end - 1, y_col]

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
    Generate windows (X, dec_X, y) per crystal with independent normalization.

    Returns:
        X, dec_X, y : concatenated arrays across all crystals
        scalers_X   : dict {xtal_id: scaler_X}
        scalers_y   : dict {xtal_id: scaler_y}
    """
    X_all, dec_all, y_all = [], [], []
    scalers_X, scalers_y  = {}, {}

    for xtal in df["xtal_id"].unique():
        df_xtal = df[df["xtal_id"] == xtal].copy()
        df_xtal["target"] = df_xtal[target_var]

        scaler_X = _make_scaler(norm_method)
        scaler_y = _make_scaler(norm_method)

        if scaler_X is not None:
            # scaler_X is fitted with order [var1, ..., calibration]
            # (= var + [target_var]), which is the natural DataFrame order.
            # IMPORTANT for external inference: split_sequences reorders
            # encoder columns to [calibration, var1, ...] AFTER scaling.
            # When building inputs manually:
            #   1. Scale with scaler_X using order [var1, ..., calibration]
            #   2. Reorder to [calibration, var1, ...] before the encoder
            #   3. The decoder receives [target, var1, ...] without reordering
            df_xtal[var + [target_var]] = scaler_X.fit_transform(
                df_xtal[var + [target_var]]
            )
        if scaler_y is not None:
            df_xtal["target"] = scaler_y.fit_transform(df_xtal[["target"]])

        seq_array = df_xtal[var + [target_var, "target"]].values
        X_x, dec_x, y_x = split_sequences(seq_array, n_steps, n_outputs, stride)
        X_all.extend(X_x)
        dec_all.extend(dec_x)
        y_all.extend(y_x)

        scalers_X[xtal] = scaler_X
        scalers_y[xtal] = scaler_y

    return np.array(X_all), np.array(dec_all), np.array(y_all), scalers_X, scalers_y


# ──────────────────────────────────────────────────────────────────────────────
# 6. SEQ2SEQ ARCHITECTURE 2 MODEL
# ──────────────────────────────────────────────────────────────────────────────

class Seq2Seq(Model):
    """
    Stacked Encoder–Decoder LSTM with teacher forcing / scheduled sampling.
    The decoder receives ONLY the target variable at each step (no exogenous variables).
    Uses separate layer lists so Keras tracks weights correctly.
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

        # Separate lists so Keras tracks weights correctly
        self.encoder_lstms = [
            LSTM(u, return_sequences=True, return_state=True)
            for u in encoder_units
        ]
        self.encoder_bns = [
            BatchNormalization() if encoder_batchnorm else None
            for _ in encoder_units
        ]
        self.encoder_dos = [
            Dropout(encoder_dropout) if encoder_dropout > 0 else None
            for _ in encoder_units
        ]

        cells = [LSTMCell(u) for u in decoder_units]
        self.decoder_cell = (
            tf.keras.layers.StackedRNNCells(cells) if len(cells) > 1 else cells[0]
        )
        self.out_dense = Dense(output_dim)

    # ------------------------------------------------------------------
    def _encode(self, x, training):
        states = []
        for lstm, bn, do in zip(self.encoder_lstms, self.encoder_bns, self.encoder_dos):
            x, h, c = lstm(x, training=training)
            states += [h, c]
            if bn is not None:
                x = bn(x, training=training)
            if do is not None:
                x = do(x, training=training)
        return x, states

    def _init_dec_states(self, enc_states):
        if isinstance(self.decoder_cell, tf.keras.layers.StackedRNNCells):
            return [(enc_states[i], enc_states[i + 1])
                    for i in range(0, len(enc_states), 2)]
        return [enc_states[-2], enc_states[-1]]

    # ------------------------------------------------------------------
    def call(self, encoder_inputs, decoder_inputs,
             training=False, forcing_ratio=1.0, per_sample=False):
        """
        encoder_inputs : (batch, n_steps,   n_enc_features)
        decoder_inputs : (batch, n_outputs, target)  ← [target, vars...]
        """
        x = tf.cast(encoder_inputs, tf.float32)

        _, enc_states = self._encode(x, training)
        dec_states    = self._init_dec_states(enc_states)

        dec_inputs = tf.cast(decoder_inputs, tf.float32)
        if dec_inputs.shape.ndims == 2:
            dec_inputs = tf.expand_dims(dec_inputs, -1)

        batch_size = tf.shape(dec_inputs)[0]
        dec_input  = dec_inputs[:, 0, :]         # initial token (batch, 1)
        outputs    = []

        for t in range(self.n_outputs):
           out, dec_states = self.decoder_cell(dec_input, dec_states, training=training)
           pred = self.out_dense(out)           # (batch, output_dim)
           outputs.append(pred)     
           
           if training:
               if per_sample:
                   mask = tf.cast(tf.random.uniform([batch_size]) < forcing_ratio, dec_inputs.dtype)
                   mask = tf.expand_dims(mask, -1)
                   if (t + 1) < dec_inputs.shape[1]:
                       teacher = dec_inputs[:, t + 1, :]
                   else:
                       teacher = tf.zeros_like(pred)
                   dec_input = mask * teacher + (1.0 - mask) * pred
               else:
                   use_teacher = tf.random.uniform([]) < forcing_ratio
                   if use_teacher and (t + 1) < dec_inputs.shape[1]:
                       dec_input = dec_inputs[:, t + 1, :]
                   else:
                       dec_input = pred
           else:
               dec_input = pred

        return tf.stack(outputs, axis=1)          # (batch, n_outputs, output_dim)



# ──────────────────────────────────────────────────────────────────────────────
# 7. NORMALIZATION HELPER
# ──────────────────────────────────────────────────────────────────────────────

def _make_scaler(method: str):
    """Return a scaler instance for the requested method, or None if method='None'."""
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
        learning_rate : learning rate to apply.
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

    Args:
        name : "mse" | "mae" | "huber" | "logcosh"
               None  → defaults to "mse"
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


def _compute_metrics_xtal(y_true: np.ndarray, y_pred: np.ndarray,
                           metrics: list) -> dict:
    """
    Compute evaluation metrics for a SINGLE crystal (original scale).

    Used by: evaluate_model, evaluate_and_plot_model.

    Args:
        y_true  : ground-truth values (1-D array).
        y_pred  : model predictions (1-D array).
        metrics : list of metric names to compute.
                  Available: "mape" | "smape" | "mae" | "rmse" | "maxae" | "r2"

    Returns:
        dict {metric_name: value}.

    Interpretation guide for calibration series (typical range ~0.7–1.0):
        mape  : mean percentage error; directly comparable across horizons
                and crystals. This is the standard project metric.
        smape : more stable than mape when target approaches 0; penalizes
                positive and negative errors asymmetrically.
        mae   : error in calibration units; easy to interpret physically.
        rmse  : like mae but penalizes error spikes by squaring them;
                useful for detecting sporadic large failures.
        maxae : worst-case point error; relevant when a hard error bound
                is required for reconstruction guarantees.
        r2    : fraction of variance explained; 1=perfect, 0=constant mean,
                <0=worse than always predicting the mean.
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


# ── Metrics for ALL crystals in the ring ──────────────────────────────────────

def _compute_metrics_ring(all_y_true: np.ndarray, all_y_pred: np.ndarray,
                           df_wmape: "pd.DataFrame",
                           metrics: list) -> dict:
    """
    Compute global evaluation metrics over ALL crystals in the ring.

    Used by: evaluate_all_xtals.

    Args:
        all_y_true : concatenated 1-D array of all crystal ground truths.
        all_y_pred : concatenated 1-D array of all crystal predictions.
        df_wmape   : DataFrame with columns {"xtal_id", "wmape", "n"}, one row per crystal.
                     wmape_i = Σ|y_i - ŷ_i| / Σ|y_i| × 100  (true WMAPE per crystal)
                     n_i     = number of observations for crystal i.
        metrics    : list of metric names to compute.
                     Available: "wmape_pond" | "mae_global" | "rmse_global"

    Returns:
        dict with only the requested metrics.
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
    mode:            str   = "mixed",
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
    shuffle:         bool  = True,
    patience:        int   = 20,
    monitor:         str   = "val_loss", # "val_loss" | "loss"
    min_delta:       float = None,       # None → 0.0 (no minimum improvement threshold)
    save_weights:    bool  = False,
    results_dir:     str   = None,
    ring_name:       str   = None,
    time_col:        str   = "laser_datetime",
):
    """
    Train the Seq2Seq model over all crystals in df_train with independent
    per-crystal normalization. Returns (model, history, scalers_X, scalers_y).

    If save_weights=True, the following files are written:
        <results_dir>/<ring>_seq2seq_arch2_<mode>_<n_steps>_<n_outputs>_<norm>/model_<n_steps>steps.weights.h5
        <results_dir>/<ring>_seq2seq_arch2_<mode>_<n_steps>_<n_outputs>_<norm>/scaler_X_<xtal_id>.pkl
        <results_dir>/<ring>_seq2seq_arch2_<mode>_<n_steps>_<n_outputs>_<norm>/scaler_y_<xtal_id>.pkl
        <results_dir>/<ring>_seq2seq_arch2_<mode>_<n_steps>_<n_outputs>_<norm>/n_vars.txt

    If shuffle=False, data is split in temporal order without prior permutation.
    """
    # Windowing with per-crystal normalization
    X_full, decX_full, y_full, scalers_X, scalers_y = split_sequences_by_xtal(
        df_train, var, n_steps=n_steps, n_outputs=n_outputs,
        stride=stride, norm_method=ext_norm_method, target_var=target_var,
    )

    # If shuffle=True  → mix samples from all crystals before splitting,
    #                    ensuring the validation set contains varied crystals.
    # If shuffle=False → preserve the arrival order (temporal, per crystal).
    if shuffle:
        perm = np.random.permutation(len(X_full))
        X_full, decX_full, y_full = X_full[perm], decX_full[perm], y_full[perm]

    split        = int(len(X_full) * (1 - val_split_ratio))
    X_train,    X_val    = X_full[:split],    X_full[split:]
    decX_train, decX_val = decX_full[:split], decX_full[split:]
    y_train,    y_val    = y_full[:split],    y_full[split:]

    del X_full, decX_full, y_full
    gc.collect()

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, decX_train, y_train))
    if shuffle:
        train_ds = train_ds.shuffle(
            buffer_size=min(8192, len(X_train)), seed=CONFIG["seed"]
        )
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds   = (tf.data.Dataset.from_tensor_slices((X_val, decX_val, y_val))
                .batch(batch_size).prefetch(tf.data.AUTOTUNE))

    del X_train, decX_train, y_train, X_val, decX_val, y_val
    gc.collect()

    model         = Seq2Seq(enc_units, dec_units, output_dim=1,
                            n_outputs=n_outputs, encoder_batchnorm=internal_norm)
    optimizer_obj = _build_optimizer(optimizer, learning_rate)
    loss_fn       = _build_loss(loss)

    # Resolve minimum improvement threshold (None maps to 0.0)
    _min_delta = min_delta if min_delta is not None else 0.0

    if monitor not in ("val_loss", "loss"):
        raise ValueError(f"monitor must be 'val_loss' or 'loss', got: '{monitor}'")

    history          = {"loss": [], "val_loss": []}
    best_monitored   = np.inf
    patience_counter = 0
    best_weights     = None

    # @tf.function-compiled steps for GPU performance
    @tf.function
    def train_step(Xb, decb, yb, forcing_ratio_tf):
        with tf.GradientTape() as tape:
            preds      = model(Xb, decb, training=True,
                               forcing_ratio=forcing_ratio_tf, per_sample=per_sample)
            preds_loss = tf.squeeze(preds, axis=-1) if preds.shape[-1] == 1 else preds
            loss_val   = loss_fn(yb, preds_loss)
        grads = tape.gradient(loss_val, model.trainable_variables)
        optimizer_obj.apply_gradients(zip(grads, model.trainable_variables))
        return loss_val

    @tf.function
    def val_step(Xb, decb, yb):
        preds      = model(Xb, decb, training=False)
        preds_loss = tf.squeeze(preds, axis=-1) if preds.shape[-1] == 1 else preds
        return loss_fn(yb, preds_loss)

    for epoch in range(epochs):
        if mode == "forced":
            forcing_ratio = 1.0
        elif mode == "recursive":
            forcing_ratio = 0.0
        elif mode == "mixed":
            decay         = max(0.0, (decay_epochs - epoch) / max(1, decay_epochs))
            forcing_ratio = float(end_forcing + (start_forcing - end_forcing) * decay)
        else:
            raise ValueError(f"Invalid mode: '{mode}'. Use 'forced' | 'recursive' | 'mixed'.")

        forcing_ratio_tf = tf.constant(forcing_ratio, dtype=tf.float32)

        train_losses = [float(train_step(Xb, decb, yb, forcing_ratio_tf).numpy())
                        for Xb, decb, yb in train_ds]

        val_losses = [float(val_step(Xb, decb, yb).numpy())
                      for Xb, decb, yb in val_ds]

        avg_train = np.mean(train_losses)
        avg_val   = np.mean(val_losses)
        history["loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        # Early stopping: monitor the metric specified by `monitor`
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

    # Save weights and scalers (optional)
    if save_weights:
        if results_dir is None:
            raise ValueError("results_dir is required when save_weights=True.")
        ring_str    = ring_name or "ring_unknown"
        norm_str    = ext_norm_method.lower()
        weights_dir = os.path.join(
            results_dir, f"{ring_str}_seq2seq_arch2_{mode}_{n_steps}_{n_outputs}_{norm_str}"
        )
        os.makedirs(weights_dir, exist_ok=True)
        model.save_weights(os.path.join(weights_dir, f"model_{n_steps}steps.weights.h5"))
        for xtal, sx in scalers_X.items():
            joblib.dump(sx, os.path.join(weights_dir, f"scaler_X_{xtal}.pkl"))
        for xtal, sy in scalers_y.items():
            joblib.dump(sy, os.path.join(weights_dir, f"scaler_y_{xtal}.pkl"))
        with open(os.path.join(weights_dir, "n_vars.txt"), "w") as f:
            f.write(",".join(var) if var else "__no_variables__")
        print(f"Weights saved to: {weights_dir}")

    return model, history, scalers_X, scalers_y


# ──────────────────────────────────────────────────────────────────────────────
# 9. EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_predictions_xtal(
    model, df_xtal, var, n_steps, n_outputs, stride,
    scaler_X, scaler_y, start_token=None,
    target_var="calibration", time_col="laser_datetime",
):
    """
    Internal helper: scale, window, predict, and align predictions on the
    time axis for a SINGLE crystal.

    Args:
        model     : trained Seq2Seq model.
        df_xtal   : test DataFrame for one crystal only.
        var       : list of predictor column names.
        n_steps   : look-back window size.
        n_outputs : forecast horizon.
        stride    : step size between windows.
        scaler_X  : fitted input scaler for this crystal (or None).
        scaler_y  : fitted target scaler for this crystal (or None).
        start_token: initial decoder token. None → last known target value.
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

    if scaler_X is not None:
        data_test.loc[:, var + [target_var]] = scaler_X.transform(
            data_test[var + [target_var]]
        )
    if scaler_y is not None:
        data_test.loc[:, "target"] = scaler_y.transform(data_test[["target"]])

    test_array = data_test.values
    if len(test_array) < n_steps + n_outputs:
        raise ValueError("Test segment is too short to generate any windows.")

    X_test, decX_test, _ = split_sequences(
        test_array, n_steps, n_outputs, stride, start_token=start_token
    )

    # Guard: windowing can return empty arrays if the crystal has very few samples
    # relative to n_steps + n_outputs with the given stride. This is distinct from
    # the length check above: with stride > 1 the effective threshold is higher.
    if X_test.size == 0:
        raise ValueError(
            f"No windows were generated for this crystal (X_test is empty). "
            f"Check that it has enough samples for "
            f"n_steps + n_outputs = {n_steps + n_outputs} with stride = {stride}."
        )

    yhat = model(X_test, decX_test, training=False).numpy()
    if yhat.shape[-1] == 1:
        yhat = yhat.squeeze(-1)

    # Temporal alignment: average overlapping predictions per time index
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

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Guard: if no index fell within the valid range the arrays remain empty.
    # This indicates a misalignment between n_steps, n_outputs, stride, and crystal size.
    if y_pred.size == 0:
        raise ValueError(
            f"No valid predictions after reconstructing overlapping windows "
            f"for this crystal. Check that n_steps={n_steps}, "
            f"n_outputs={n_outputs}, and stride={stride} are consistent with "
            f"the size of the test segment for this crystal."
        )

    # Inverse-transform to the original calibration scale.
    # If scaler_y is None (norm_method="None"), values are already on the original
    # scale and no inverse transform is needed.
    if scaler_y is not None:
        y_true = scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    return y_pred, y_true, np.array(valid_time)


def evaluate_model(
    model, df_test, var, n_steps, n_outputs=1, stride=1,
    scaler_X=None, scaler_y=None, start_token=None,
    target_var="calibration", time_col="laser_datetime",
    metrics: list = None,
):
    """
    Evaluate a SINGLE crystal. No plots generated.

    (history is only needed in evaluate_and_plot_model for the loss curve plot.)

    Args:
        metrics : list of metrics for the individual crystal.
                  None → only "mape" (backward-compatible default).

    Returns:
        Tuple (mape, y_true, y_pred, metrics_dict).
    """
    y_pred, y_true, _ = _prepare_predictions_xtal(
        model, df_test, var, n_steps, n_outputs, stride,
        scaler_X, scaler_y, start_token, target_var, time_col,
    )

    _metrics = metrics if metrics is not None else ["mape"]
    results  = _compute_metrics_xtal(y_true, y_pred, _metrics)

    for name, value in results.items():
        print(f"  {name.upper():6s}: {value:.4f}")

    mape = results.get(
        "mape",
        100 * mean_absolute_percentage_error(y_true, y_pred)
    )
    return mape, y_true, y_pred, results


def evaluate_all_xtals(
    model, df_test, var, n_steps, n_outputs=1, stride=1,
    scalers_X=None, scalers_y=None, start_token=None,
    target_var="calibration", time_col="laser_datetime",
    ring_metrics: list = None,
):
    """
    Evaluate the model over ALL crystals in the ring.

    Calls _prepare_predictions_xtal directly (no individual crystal metrics,
    which are not used in this flow).

    Crystals without a matching scaler (not in training set) are skipped with
    an explicit warning.

    Args:
        ring_metrics : list of ring-level metrics to compute.
                       Available: "wmape_pond" | "mae_global" | "rmse_global"
                       None → all three.

    Returns:
        Tuple (metrics_dict, df_per_crystal).
        The primary scalar for grid accumulation is metrics_dict["wmape_pond"].
    """
    _ring_metrics = ring_metrics if ring_metrics is not None else [
        "wmape_pond", "mae_global", "rmse_global"
    ]

    results_per_xtal       = []
    all_y_true, all_y_pred = [], []

    for xtal in df_test["xtal_id"].unique():
        df_xtal  = df_test[df_test["xtal_id"] == xtal]
        scaler_X = scalers_X.get(xtal) if scalers_X else None
        scaler_y = scalers_y.get(xtal) if scalers_y else None

        if scalers_X and scaler_X is None:
            print(f"[WARNING] Crystal {xtal} has no scaler_X (not in training set). Skipped.")
            continue
        if scalers_y and scaler_y is None:
            print(f"[WARNING] Crystal {xtal} has no scaler_y (not in training set). Skipped.")
            continue

        try:
            y_pred, y_true, _ = _prepare_predictions_xtal(
                model, df_xtal, var, n_steps, n_outputs, stride,
                scaler_X, scaler_y, start_token, target_var, time_col,
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

        except Exception as e:
            print(f"[ERROR] Crystal {xtal}: {e}")
            continue

    if not results_per_xtal:
        return {m: np.nan for m in _ring_metrics}, pd.DataFrame()

    df_res     = pd.DataFrame(results_per_xtal)
    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    final_metrics = _compute_metrics_ring(y_true_all, y_pred_all, df_res, _ring_metrics)
    return final_metrics, df_res


def evaluate_and_plot_model(
    model, history, df_test, var, n_steps, n_outputs=1, stride=1,
    scalers_X=None, scalers_y=None, xtal_id=None, start_token=None,
    target_var="calibration", plot_ratio=True,
    results_dir=None, time_col="laser_datetime",
    metrics: list = None,
    loss: str = None,
):
    """
    Evaluate a specific crystal (or the first one with an available scaler),
    compute individual metrics, and generate diagnostic plots:
        Figure 1 — Training vs validation loss curve.
        Figure 2 — Predicted vs actual values (with optional ratio subplot).

    Uses _compute_metrics_xtal → mape, smape, mae, rmse, maxae, r2.

    Args:
        metrics : list of individual-crystal metrics to compute.
                  None → ["mape", "smape", "mae", "rmse", "maxae", "r2"].
        loss    : loss function name used during training
                  ("mse" | "mae" | "huber" | "logcosh" | None → "mse").
                  Used only to label the Y-axis of the loss curve.

    Returns:
        Tuple (mape, y_true, y_pred).
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

    df_xtal  = df_test[df_test["xtal_id"] == xtal]
    scaler_X = scalers_X.get(xtal) if scalers_X else None
    scaler_y = scalers_y.get(xtal) if scalers_y else None

    if scalers_X and scaler_X is None:
        raise ValueError(
            f"Crystal {xtal} has no scaler. "
            f"Select another with xtal_id=<id>."
        )

    y_pred, y_true, valid_time = _prepare_predictions_xtal(
        model, df_xtal, var, n_steps, n_outputs, stride,
        scaler_X, scaler_y, start_token, target_var, time_col,
    )

    # Individual crystal metrics
    _metrics = metrics if metrics is not None else [
        "mape", "smape", "mae", "rmse", "maxae", "r2"
    ]
    results     = _compute_metrics_xtal(y_true, y_pred, _metrics)
    metrics_str = "  |  ".join(
        f"{name.upper()}: {value:.4f}" for name, value in results.items()
    )
    mape = results.get(
        "mape",
        100 * mean_absolute_percentage_error(y_true, y_pred)
    )

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
    if results_dir:
        fig_loss.savefig(
            os.path.join(results_dir, f"loss_xtal_{xtal}_{n_steps}.png"), dpi=300
        )
    plt.show()

    # ── Figure 2: predicted vs actual (+ optional ratio subplot) ─────────────
    # If xtal_id was None the crystal was auto-selected; indicate this with "(auto)".
    suffix = "" if xtal_id is not None else " (auto)"
    title  = (
        f"Actual vs predicted — horizon {n_steps} "
        f"— crystal {xtal}{suffix}\n{metrics_str}"
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
    return mape, y_true, y_pred


# ──────────────────────────────────────────────────────────────────────────────
# 10. RESULTS ACCUMULATION AND REPORTING
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
        tuple(r["enc_units"]),
        tuple(r["dec_units"]),
        r["mode"],
        r["start_forcing"],
        r["end_forcing"],
        r["decay_epochs"],
        r["per_sample"],
        r.get("optimizer", "adam") or "adam",
        r["learning_rate"],
        r.get("loss",      "mse")  or "mse",
        r["ext_norm_method"],
        r["internal_norm"],
        r.get("monitor",   "val_loss"),
        r.get("min_delta", 0.0)    or 0.0,
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
    for result in results_list:
        key = _config_key(result)
        if key not in config_details_global:
            config_details_global[key] = {
                "cfg_ids": [],
                "n_steps": [], "n_outputs": [], "stride": [],
                "metrics": defaultdict(list),
            }
        config_details_global[key]["cfg_ids"] .append(result.get("cfg_id", None))
        config_details_global[key]["n_steps"] .append(result["n_steps"])
        config_details_global[key]["n_outputs"].append(result["n_outputs"])
        config_details_global[key]["stride"]  .append(result["stride"])

        for metric_name, value in result["evaluated_metrics"].items():
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
            (var, enc_units, dec_units, mode, start_forcing,
             end_forcing, decay_epochs, per_sample,
             optimizer_k, learning_rate, loss_k,
             ext_norm_method, internal_norm,
             monitor_k, min_delta_k, shuffle) = key

            det = config_details_global[key]

            f.write("Configuration:\n")
            f.write(f"  cfg_ids:    {det['cfg_ids']}\n")
            f.write(f"  Variables:  {var}\n")
            f.write(f"  enc_units:  {enc_units}  |  dec_units: {dec_units}\n")
            f.write(f"  mode:       {mode}\n")
            f.write(f"  forcing:    start={start_forcing}, end={end_forcing}, "
                    f"decay_epochs={decay_epochs}\n")
            f.write(f"  per_sample: {per_sample}  |  lr: {learning_rate}\n")
            f.write(f"  optimizer:  {optimizer_k}  |  loss: {loss_k}\n")
            f.write(f"  monitor:    {monitor_k}  |  min_delta: {min_delta_k}\n")
            f.write(f"  norm:       {ext_norm_method}  |  batchnorm: {internal_norm}"
                    f"  |  shuffle: {shuffle}\n")

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
                    if ns == no == st and ns in ref_dict and val < ref_dict[ns]:
                        better_horizons.append(ns)
                f.write(f"  Horizons beating reference: {better_horizons}\n")

            f.write("-" * 80 + "\n\n")


# ──────────────────────────────────────────────────────────────────────────────
# 11. MAIN EXECUTION
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

    Parameters are read from cfg_grid; unspecified ones inherit from cfg["default"],
    so each grid entry only needs to declare the values that differ from the default.

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

    cur_var      = VARIABLE_SETS[cfg_grid["variables"]]
    rnn_units    = cfg_grid["rnn_units"]
    shuffle_flag = cfg_grid["shuffle"]
    mode         = cfg_grid["mode"]
    cfg_id       = cfg_grid["id"]

    # Optional parameters in cfg_grid → fall back to default if not specified
    per_sample     = cfg_grid.get("per_sample",    d["per_sample"])
    optimizer_name = cfg_grid.get("optimizer",     d.get("optimizer",    None))
    learning_rate  = cfg_grid.get("learning_rate", d["learning_rate"])
    loss_name      = cfg_grid.get("loss",          d.get("loss",         None))
    norm_key       = cfg_grid.get("norm_method",   d["norm_method"])
    batchnorm      = cfg_grid.get("internal_norm", d["internal_norm"])
    monitor_metric = cfg_grid.get("monitor",       d.get("monitor",      "val_loss"))
    min_delta_val  = cfg_grid.get("min_delta",     d.get("min_delta",    None))
    use_cpu        = cfg_grid.get("use_cpu",        gf.get("use_cpu",     False))
    save_w         = gf["save_weights"]

    start_forcing = gf.get("start_forcing", d.get("start_forcing", 1.0))
    end_forcing   = gf.get("end_forcing",   d.get("end_forcing",   0.0))
    decay_epochs  = gf.get("decay_epochs",  d.get("decay_epochs",  30))

    for a in horizons_list:
        # batch_size: None → dynamic based on the current horizon `a`, not the default
        batch = gf["batch_size"]
        if batch is None:
            batch = 32 if a >= 48 else 128
        try:
            reset_environment(seed=cfg["seed"], use_cpu=use_cpu)

            model, history, scalers_X, scalers_y = train_seq2seq(
                df_train=df_train, var=cur_var,
                target_var=cfg["target_var"],
                n_steps=a, n_outputs=a, stride=a,
                enc_units=rnn_units, dec_units=rnn_units,
                epochs=gf["epochs"],
                batch_size=batch,
                mode=mode,
                start_forcing=start_forcing,
                end_forcing=end_forcing,
                decay_epochs=decay_epochs,
                per_sample=per_sample,
                optimizer=optimizer_name,
                learning_rate=learning_rate,
                loss=loss_name,
                ext_norm_method=norm_key,
                internal_norm=batchnorm,
                shuffle=shuffle_flag,
                val_split_ratio=gf["val_split"],
                patience=gf["patience"],
                monitor=monitor_metric,
                min_delta=min_delta_val,
                save_weights=save_w,
                results_dir=cfg["results_dir"],
                ring_name=cfg["active_ring"],
                time_col=cfg["time_col"],
            )

            ring_metrics_dict, _ = evaluate_all_xtals(
                model, df_test, cur_var,
                n_steps=a, n_outputs=a, stride=a,
                scalers_X=scalers_X, scalers_y=scalers_y,
                time_col=cfg["time_col"],
                ring_metrics=gf["ring_metrics"],  # Pass the full list defined in config
            )

            metrics_str = "  |  ".join(
                f"{k.upper()}: {v:.4f}" for k, v in ring_metrics_dict.items()
            )
            print(f"  {metrics_str}")

            accumulate_results([{
                "cfg_id":         cfg_id,
                "var":            cur_var,
                "n_steps":        a,
                "n_outputs":      a,
                "stride":         a,
                "enc_units":      rnn_units,
                "dec_units":      rnn_units,
                "batch_size":     batch,
                "mode":           mode,
                "start_forcing":  start_forcing,
                "end_forcing":    end_forcing,
                "decay_epochs":   decay_epochs,
                "per_sample":     per_sample,
                "optimizer":      optimizer_name,
                "learning_rate":  learning_rate,
                "loss":           loss_name,
                "monitor":        monitor_metric,
                "min_delta":      min_delta_val if min_delta_val is not None else 0.0,
                "ext_norm_method": norm_key,
                "internal_norm":  batchnorm,
                "shuffle":        shuffle_flag,
                "evaluated_metrics": ring_metrics_dict,  # Full metrics dict
            }])
            print_results_txt(gf.get("reference_metrics", {}), output_file)

        except Exception:
            print(f"[ERROR] cfg {cfg_id} | n_steps={a} | enc={rnn_units}:")
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

    rnn_units = d["rnn_units"]

    model, history, scalers_X, scalers_y = train_seq2seq(
        df_train=df_train,
        var=d["variables"],
        target_var=cfg["target_var"],
        n_steps=d["n_steps"],
        n_outputs=d["n_outputs"],
        stride=d["stride"],
        enc_units=rnn_units,
        dec_units=rnn_units,
        epochs=d["epochs"],
        batch_size=batch,
        mode=d["mode"],
        start_forcing=d.get("start_forcing", 1.0),
        end_forcing=d.get("end_forcing",     0.0),
        decay_epochs=d.get("decay_epochs",   30),
        per_sample=d["per_sample"],
        optimizer=d.get("optimizer",         None),
        learning_rate=d["learning_rate"],
        loss=d.get("loss",                   None),
        ext_norm_method=d["norm_method"],
        internal_norm=d["internal_norm"],
        shuffle=d["shuffle"],
        val_split_ratio=d["val_split"],
        patience=d["patience"],
        monitor=d.get("monitor",             "val_loss"),
        min_delta=d.get("min_delta",         None),
        save_weights=d.get("save_weights",   False),
        results_dir=cfg["results_dir"],
        ring_name=cfg["active_ring"],
        time_col=cfg["time_col"],
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
# 12. LOADING A SAVED MODEL
# ──────────────────────────────────────────────────────────────────────────────

def load_model(cfg=CONFIG):
    """
    Reconstruct a previously saved multi-crystal Seq2Seq model.

    Scalers are dicts {xtal_id: scaler} because each crystal has its own
    independent normalization.

    Loads from:
        <results_dir>/<ring>_seq2seq_arch2_<mode>_<n_steps>_<n_outputs>_<norm>/model_<n_steps>steps.weights.h5
        <results_dir>/<ring>_seq2seq_arch2_<mode>_<n_steps>_<n_outputs>_<norm>/scaler_X_<xtal_id>.pkl
        <results_dir>/<ring>_seq2seq_arch2_<mode>_<n_steps>_<n_outputs>_<norm>/scaler_y_<xtal_id>.pkl
        <results_dir>/<ring>_seq2seq_arch2_<mode>_<n_steps>_<n_outputs>_<norm>/n_vars.txt

    Args:
        cfg: Global configuration dictionary (see CONFIG).

    Returns:
        Tuple (model, scalers_X, scalers_y) ready for evaluation.

    Raises:
        FileNotFoundError : If n_vars.txt or scaler files are missing.
        ValueError        : If the saved variable list differs from CONFIG,
                            or if scalers_X and scalers_y have mismatched keys.
    """
    tf.keras.backend.clear_session()
    d           = cfg["default"]
    n_steps     = d["n_steps"]
    n_outputs   = d["n_outputs"]
    ring_str    = cfg["active_ring"]
    mode_str    = d["mode"]
    norm_str    = d["norm_method"].lower()
    weights_dir = os.path.join(
        cfg["results_dir"],
        f"{ring_str}_seq2seq_arch2_{mode_str}_{n_steps}_{n_outputs}_{norm_str}"
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
    rnn_units = d["rnn_units"]
    model = Seq2Seq(
        encoder_units=rnn_units,
        decoder_units=rnn_units,
        output_dim=1,
        n_outputs=n_outputs,
        encoder_batchnorm=d["internal_norm"],
    )

    # Dummy forward pass to initialize all layer weights before loading saved values.
    # ── Encoder column order for external inference ───────────────────────────
    # 1. Build the array in order [var1, ..., calibration] (= d["variables"] + [target_var])
    # 2. Apply scaler_X.transform() with that order
    # 3. Reorder to [calibration, var1, ...] before passing to the encoder
    # 4. The decoder receives [target, var1, ...] without additional reordering
    n_enc_features = len(d["variables"]) + 1   # vars + calibration
    n_dec_features = 1                         # Only target
    dummy_X   = np.zeros((1, n_steps,   n_enc_features), dtype=np.float32)
    dummy_dec = np.zeros((1, n_outputs, n_dec_features), dtype=np.float32)
    model(dummy_X, dummy_dec, training=False)

    # ── Step 3: Load model weights ────────────────────────────────────────────
    model.load_weights(
        os.path.join(weights_dir, f"model_{n_steps}steps.weights.h5")
    )

    # ── Step 4: Load per-crystal scalers (xtal_id may be int or string) ───────
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
    #run_grid_search(df_train, df_test, CONFIG)   # ← uncomment for grid search

    # ── Option B: load a previously saved model and evaluate ──────────────────
    # Requires a prior run with save_weights=True in CONFIG['default'].
    # Expected folder: <results_dir>/ring_1_seq2seq_arch2_forced_48_48_minmax/
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
     )'''