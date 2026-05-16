# =============================================================================
# naive_baseline.py
# Naive window baseline for calibration forecasting.
# Predicts the next n_outputs steps by repeating the last n_outputs values
# of the current window (persistence / last-value baseline).
#
# Supports evaluation at two levels:
#   - Single crystal  : individual metrics + pred vs real plot
#   - Full ring       : ring-level metrics (wmape_pond, mae_global, rmse_global)
#                       computed the same way as the neural network programs,
#                       plus an optional plot for one chosen crystal.
# =============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # ── Data sources ──────────────────────────────────────────────────────────
    "data_sources": {
        "ring_1":   "/data/plus_z_1/ring_1.csv",
        # "ring_m1":  "/data/minus_z_1/ring_-1.csv",
        # "ring_50":  "/data/plus_z_6/ring_50.csv",
        # "ring_m36": "/data/minus_z_4/ring_-36.csv"
    },
    "active_ring":  "ring_1",
    "results_dir":  "/results",

    # ── Data filters ──────────────────────────────────────────────────────────
    "target_var":   "calibration",
    "time_col":     "laser_datetime",
    "calib_min":    0.7,
    "calib_max":    1.0,
    # test_year: year used for evaluation (same convention as NN programs)
    "test_year":    2017,

    # ── Windowing parameters ──────────────────────────────────────────────────
    # n_steps == n_outputs == stride → non-overlapping windows (standard setup)
    "n_steps":      36,
    "n_outputs":    36,
    "stride":       36,

    # ── Output control ────────────────────────────────────────────────────────
    # xtal_id of the crystal to plot. None → first available crystal or False → Do not generate plots
    "plot_xtal_id": False,

    # ── Individual crystal metrics (evaluate_single_crystal) ──────────────────
    # Computed in original scale for the crystal chosen with plot_xtal_id.
    # Available: "mape" | "smape" | "mae" | "rmse" | "maxae" | "r2"
    # - mape  : Mean absolute percentage error (×100). Project standard.
    # - smape : Symmetric MAPE; more stable when target values are near 0.
    # - mae   : Mean absolute error; same units as the target.
    # - rmse  : Root mean squared error; penalises large errors more.
    # - maxae : Maximum absolute error; worst-case point error.
    # - r2    : Coefficient of determination (1=perfect, 0=mean, <0=worse).
    #
    # Interpretation guide for calibration series (typical range ~0.7–1.0):
    #   mape  : percentage error; directly comparable across horizons.
    #   smape : more stable than mape when true values are very small.
    #   mae   : error in calibration units; easy to interpret physically.
    #   rmse  : like mae but penalises error spikes; useful for fault detection.
    #   maxae : maximum point error; relevant for reconstruction guarantees.
    #   r2    : fraction of variance explained; 1=perfect, 0=mean, <0=worse.
    "metrics_single": ["mape", "smape", "mae", "rmse", "maxae", "r2"],

    # ── Ring-level metrics (evaluate_ring) ────────────────────────────────────
    # Computed by aggregating predictions over ALL crystals in the ring.
    # Unlike metrics_single (one crystal), these measure the global behaviour
    # of the baseline over the entire ring.
    # Available: "wmape_pond" | "mae_global" | "rmse_global"
    # - wmape_pond  : True per-crystal WMAPE weighted by number of observations.
    #                 More robust than averaging individual MAPEs when true
    #                 values are small.
    # - mae_global  : MAE computed over all crystals concatenated.
    # - rmse_global : RMSE computed over all crystals concatenated.
    "metrics_ring":   ["wmape_pond", "mae_global", "rmse_global"],
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_test_data(cfg: dict = CONFIG) -> pd.DataFrame:
    """
    Loads the active ring CSV, filters by calibration range, parses dates,
    sorts by (xtal_id, time_col), and returns only the test year rows.

    Validates that the resulting DataFrame is not empty.
    """
    ruta = cfg["data_sources"][cfg["active_ring"]]
    df   = pd.read_csv(ruta)[["xtal_id", cfg["target_var"], cfg["time_col"]]].copy()
    df   = df[(df[cfg["target_var"]] >= cfg["calib_min"]) &
              (df[cfg["target_var"]] <= cfg["calib_max"])]
    df[cfg["time_col"]] = pd.to_datetime(
        df[cfg["time_col"]], format="%Y-%m-%d %H:%M:%S"
    )
    df = df.sort_values(["xtal_id", cfg["time_col"]]).reset_index(drop=True)

    df_test = df[
        df[cfg["time_col"]].dt.year == cfg["test_year"]
    ].copy().reset_index(drop=True)

    if len(df_test) == 0:
        raise ValueError(
            f"Test set is empty. Check that 'test_year'={cfg['test_year']} "
            f"has data in '{ruta}' and that calib_min/calib_max filters "
            f"({cfg['calib_min']}–{cfg['calib_max']}) do not remove all rows."
        )
    return df_test


# ──────────────────────────────────────────────────────────────────────────────
# 3. NAIVE PREDICTION (single crystal)
# ──────────────────────────────────────────────────────────────────────────────

def _naive_predictions_xtal(
    df_xtal:    pd.DataFrame,
    n_steps:    int,
    n_outputs:  int  = 1,
    stride:     int  = 1,
    target_var: str  = "calibration",
    time_col:   str  = "laser_datetime",
) -> tuple:
    """
    Computes naive window predictions for a SINGLE crystal.

    The naive strategy predicts the next n_outputs values by repeating the
    last n_outputs observations of each window (persistence baseline).

    Overlapping windows are handled by averaging predictions at each index,
    exactly mirroring the reconstruction logic in the neural network programs.

    Args:
        df_xtal    : DataFrame for one crystal, unsorted is fine.
        n_steps    : look-back window length.
        n_outputs  : forecast horizon (number of steps ahead).
        stride     : step between consecutive windows.
        target_var : name of the target column.
        time_col   : name of the datetime column.

    Returns:
        y_pred     : array of naive predictions (original scale).
        y_true     : array of true values (original scale).
        valid_time : array of datetime values aligned with y_pred / y_true.

    Raises:
        ValueError if the crystal has too few observations to build any window,
        or if no valid predictions remain after reconstruction.
    """
    df_xtal = df_xtal.sort_values(time_col).copy()
    y_full  = df_xtal[target_var].values
    dates   = df_xtal[time_col].values
    max_idx = len(y_full)

    # Guard: minimum length
    if max_idx < n_steps + n_outputs:
        raise ValueError(
            f"Crystal has {max_idx} observations, but needs at least "
            f"n_steps + n_outputs = {n_steps + n_outputs}."
        )

    pred_list = [[] for _ in range(max_idx)]

    for i in range(0, max_idx - n_steps - n_outputs + 1, stride):
        window = y_full[i : i + n_steps]
        # Naive prediction: repeat the last n_outputs values of the window
        pred = window[-n_outputs:]
        for j in range(n_outputs):
            idx = i + n_steps + j
            if idx < max_idx:
                pred_list[idx].append(pred[j])

    y_pred, y_true, valid_time = [], [], []
    for i in range(n_steps, max_idx):
        if pred_list[i]:
            y_pred.append(np.mean(pred_list[i]))
            y_true.append(y_full[i])
            valid_time.append(dates[i])

    # Guard: empty reconstruction
    if not y_pred:
        raise ValueError(
            f"No valid predictions after window reconstruction. "
            f"Check that n_steps={n_steps}, n_outputs={n_outputs}, "
            f"stride={stride} are consistent with the crystal's test size."
        )

    return np.array(y_pred), np.array(y_true), np.array(valid_time)


# ──────────────────────────────────────────────────────────────────────────────
# 4. METRICS
# ──────────────────────────────────────────────────────────────────────────────

# Metrics where higher is better (used in reference comparison)
METRIC_HIGHER_IS_BETTER = {"r2"}


def _compute_metrics_single(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    metrics: list,
) -> dict:
    """
    Computes individual crystal metrics (original scale).

    Args:
        y_true  : true values (1-D array).
        y_pred  : predicted values (1-D array).
        metrics : list of metric names. Available:
                  "mape" | "smape" | "mae" | "rmse" | "maxae" | "r2"

    Returns dict {metric_name: value}.
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


def _compute_metrics_ring(
    all_y_true: np.ndarray,
    all_y_pred: np.ndarray,
    df_wmape:   pd.DataFrame,
    metrics:    list,
) -> dict:
    """
    Computes ring-level metrics over ALL crystals.

    Identical logic to the neural network programs:
        wmape_i    = Σ|y_i - ŷ_i| / Σ|y_i| × 100   (true WMAPE per crystal)
        wmape_pond = Σ(wmape_i × n_i) / Σn_i         (weighted by n observations)

    Args:
        all_y_true : 1-D array, concatenation of y_true for all crystals.
        all_y_pred : 1-D array, concatenation of y_pred for all crystals.
        df_wmape   : DataFrame with columns {"xtal_id", "wmape", "n"}.
        metrics    : list of names. Available:
                     "wmape_pond" | "mae_global" | "rmse_global"

    Returns dict with only the requested metrics.
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
# 5. EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_single_crystal(
    df_test:    pd.DataFrame,
    n_steps:    int,
    n_outputs:  int  = 1,
    stride:     int  = 1,
    xtal_id             = None,
    target_var: str  = "calibration",
    time_col:   str  = "laser_datetime",
    metrics:    list = None,
    plot_ratio: bool = True,
    results_dir: str = None,
) -> dict:
    """
    Evaluates the naive baseline for ONE crystal, prints metrics,
    and generates a pred vs real plot (with optional ratio panel).

    Args:
        df_test     : test DataFrame containing all crystals.
        n_steps     : look-back window length.
        n_outputs   : forecast horizon.
        stride      : step between windows.
        xtal_id     : crystal to evaluate. None → first available crystal.
        target_var  : name of the target column.
        time_col    : name of the datetime column.
        metrics     : list of metric names to compute.
                      None → CONFIG["metrics_single"].
        plot_ratio  : if True, adds a True/Pred ratio panel below the main plot.
        results_dir : directory to save the plot. None → do not save.

    Returns dict {metric_name: value}.
    """
    _metrics = metrics if metrics is not None else CONFIG["metrics_single"]

    # Resolve crystal
    available = df_test["xtal_id"].unique().tolist()
    if xtal_id is not None:
        xtal = xtal_id
        if xtal not in available:
            raise ValueError(
                f"Crystal {xtal} not found in df_test. "
                f"Available: {available[:10]}{'...' if len(available) > 10 else ''}"
            )
    else:
        xtal = available[0]

    df_xtal = df_test[df_test["xtal_id"] == xtal]
    y_pred, y_true, valid_time = _naive_predictions_xtal(
        df_xtal, n_steps, n_outputs, stride, target_var, time_col
    )

    results = _compute_metrics_single(y_true, y_pred, _metrics)

    print(f"\n── Naive baseline metrics — crystal {xtal} "
          f"(n_steps={n_steps}, n_outputs={n_outputs}, stride={stride}) ──")
    for name, value in results.items():
        print(f"  {name.upper():6s}: {value:.4f}")
    print("─" * 60)

    # ── Plot ──────────────────────────────────────────────────────────────────
    suffix      = "" if xtal_id is not None else " (auto)"
    metrics_str = "  |  ".join(
        f"{name.upper()}: {value:.4f}" for name, value in results.items()
    )
    title = (
        f"Naive baseline — horizon {n_steps} — crystal {xtal}{suffix}\n"
        f"{metrics_str}"
    )

    if plot_ratio:
        ratio = y_true / y_pred
        fig, axes = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        axes[0].plot(valid_time, y_true, "-", color="blue",   label="True")
        axes[0].plot(valid_time, y_pred, "-", color="orange", label="Naive prediction")
        axes[0].set_ylabel(target_var.capitalize())
        axes[0].set_title(title)
        axes[0].legend(fontsize=12)
        axes[1].plot(valid_time, ratio, "-", color="black")
        axes[1].axhline(1.0, color="red", linestyle="--", linewidth=1.5)
        axes[1].set_ylabel("True / Pred")
        axes[1].set_xlabel(time_col)
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(valid_time, y_true, "-", color="blue",   label="True")
        ax.plot(valid_time, y_pred, "-", color="orange", label="Naive prediction")
        ax.set_ylabel(target_var.capitalize())
        ax.set_xlabel(time_col)
        ax.set_title(title)
        ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fname = (
            f"naive_baseline_xtal_{xtal}_{n_steps}_{n_outputs}_{stride}.png"
        )
        fig.savefig(os.path.join(results_dir, fname), dpi=300)
        print(f"  Plot saved → {os.path.join(results_dir, fname)}")

    plt.show()
    return results


def evaluate_ring(
    df_test:     pd.DataFrame,
    n_steps:     int,
    n_outputs:   int  = 1,
    stride:      int  = 1,
    target_var:  str  = "calibration",
    time_col:    str  = "laser_datetime",
    metrics_ring: list = None,
    plot_xtal_id       = None,
    metrics_single: list = None,
    plot_ratio:  bool = True,
    results_dir: str  = None,
) -> tuple:
    """
    Evaluates the naive baseline over ALL crystals in the ring.

    Ring-level metrics (wmape_pond, mae_global, rmse_global) are computed
    with the same weighting logic as the neural network programs:
        wmape_i    = Σ|y_i - ŷ_i| / Σ|y_i| × 100   per crystal
        wmape_pond = Σ(wmape_i × n_i) / Σn_i

    Optionally plots one crystal (plot_xtal_id) with individual metrics.
    Crystals with too few observations are skipped with an explicit warning.

    Args:
        df_test        : test DataFrame containing all crystals.
        n_steps        : look-back window length.
        n_outputs      : forecast horizon.
        stride         : step between windows.
        target_var     : name of the target column.
        time_col       : name of the datetime column.
        metrics_ring   : ring-level metrics to compute.
                         None → CONFIG["metrics_ring"].
        plot_xtal_id   : crystal to plot individually.
                         None → plot the first available crystal.
                         False → skip individual plot entirely.
        metrics_single : individual metrics for the plotted crystal.
                         None → CONFIG["metrics_single"].
        plot_ratio     : if True, adds a True/Pred ratio panel.
        results_dir    : directory to save plots. None → do not save.

    Returns (metrics_dict, df_per_crystal):
        metrics_dict   : dict {metric_name: value} for ring-level metrics.
        df_per_crystal : DataFrame with per-crystal wmape and n.
    """
    _metrics_ring   = metrics_ring   if metrics_ring   is not None else CONFIG["metrics_ring"]
    _metrics_single = metrics_single if metrics_single is not None else CONFIG["metrics_single"]

    results_per_xtal = []
    all_y_true, all_y_pred = [], []

    for xtal in df_test["xtal_id"].unique():
        df_xtal = df_test[df_test["xtal_id"] == xtal]
        try:
            y_pred, y_true, _ = _naive_predictions_xtal(
                df_xtal, n_steps, n_outputs, stride, target_var, time_col
            )
            # True per-crystal WMAPE: Σ|y_i - ŷ_i| / Σ|y_i| × 100
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

        except ValueError as e:
            print(f"[WARNING] Crystal {xtal} skipped: {e}")
            continue

    if not results_per_xtal:
        print("[ERROR] No crystals could be evaluated.")
        return {m: np.nan for m in _metrics_ring}, pd.DataFrame()

    df_res     = pd.DataFrame(results_per_xtal)
    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)

    metrics_dict = _compute_metrics_ring(y_true_all, y_pred_all, df_res, _metrics_ring)

    print(f"\n── Naive baseline — ring metrics "
          f"(n_steps={n_steps}, n_outputs={n_outputs}, stride={stride}) ──")
    for name, value in metrics_dict.items():
        print(f"  {name.upper():12s}: {value:.4f}")
    print("─" * 60)
    print(df_res.to_string(index=False))

    # ── Individual plot for one crystal ───────────────────────────────────────
    if plot_xtal_id is not False:
        xtal_to_plot = (
            plot_xtal_id
            if plot_xtal_id is not None
            else df_res["xtal_id"].iloc[0]
        )
        try:
            evaluate_single_crystal(
                df_test=df_test,
                n_steps=n_steps,
                n_outputs=n_outputs,
                stride=stride,
                xtal_id=xtal_to_plot,
                target_var=target_var,
                time_col=time_col,
                metrics=_metrics_single,
                plot_ratio=plot_ratio,
                results_dir=results_dir,
            )
        except ValueError as e:
            print(f"[WARNING] Could not plot crystal {xtal_to_plot}: {e}")

    return metrics_dict, df_res


# ──────────────────────────────────────────────────────────────────────────────
# 6. BATCH EVALUATION OVER MULTIPLE HORIZONS
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_horizons(
    df_test:      pd.DataFrame,
    horizons:     list,
    target_var:   str  = "calibration",
    time_col:     str  = "laser_datetime",
    metrics_ring: list = None,
    results_dir:  str  = None,
    print_summary: bool = True,
) -> pd.DataFrame:
    """
    Runs evaluate_ring over a list of horizons (no plots) and returns
    a summary DataFrame with one row per horizon.

    Useful for generating the reference table used by the NN programs.

    Args:
        df_test       : test DataFrame with all crystals.
        horizons      : list of horizon values, e.g. [1, 12, 24, 36, 48].
                        Each horizon is used as n_steps = n_outputs = stride.
        target_var    : name of the target column.
        time_col      : name of the datetime column.
        metrics_ring  : ring-level metrics to compute.
                        None → CONFIG["metrics_ring"].
        results_dir   : if provided, saves a CSV summary there.
        print_summary : if True, prints the summary table.

    Returns:
        DataFrame with columns: ["horizon"] + metric names.
    """
    _metrics_ring = metrics_ring if metrics_ring is not None else CONFIG["metrics_ring"]
    rows = []

    for h in horizons:
        print(f"\n{'='*50}")
        print(f"Horizon {h}")
        print(f"{'='*50}")
        # plot_xtal_id=False → skip individual plots in batch mode
        metrics_dict, _ = evaluate_ring(
            df_test=df_test,
            n_steps=h, n_outputs=h, stride=h,
            target_var=target_var,
            time_col=time_col,
            metrics_ring=_metrics_ring,
            plot_xtal_id=False,
        )
        row = {"horizon": h}
        row.update(metrics_dict)
        rows.append(row)

    df_summary = pd.DataFrame(rows)

    if print_summary:
        print("\n── Naive baseline summary across horizons ──────────────")
        print(df_summary.to_string(index=False))
        print("─" * 60)

    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        fpath = os.path.join(results_dir, "naive_baseline_summary.csv")
        df_summary.to_csv(fpath, index=False)
        print(f"Summary saved → {fpath}")

    return df_summary


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg     = CONFIG
    df_test = load_test_data(cfg)
    '''
    # ── Option A: evaluate a single crystal ───────────────────────────────────
    evaluate_single_crystal(
        df_test=df_test,
        n_steps=cfg["n_steps"],
        n_outputs=cfg["n_outputs"],
        stride=cfg["stride"],
        xtal_id=cfg["plot_xtal_id"],
        target_var=cfg["target_var"],
        time_col=cfg["time_col"],
        metrics=cfg["metrics_single"],
        plot_ratio=True,
        results_dir=cfg["results_dir"],
    )
    
    # ── Option B: evaluate the full ring (+ plot one crystal) ─────────────────
    metrics_dict, df_per_crystal = evaluate_ring(
         df_test=df_test,
         n_steps=cfg["n_steps"],
         n_outputs=cfg["n_outputs"],
         stride=cfg["stride"],
         target_var=cfg["target_var"],
         time_col=cfg["time_col"],
         metrics_ring=cfg["metrics_ring"],
         plot_xtal_id=cfg["plot_xtal_id"],
         metrics_single=cfg["metrics_single"],
         plot_ratio=True,
         results_dir=cfg["results_dir"],
     )
'''
    # ── Option C: evaluate over multiple horizons (reference table) ───────────
    df_summary = evaluate_horizons(
         df_test=df_test,
         horizons=[1, 12, 24, 36, 48, 60, 72, 84, 96],
         target_var=cfg["target_var"],
         time_col=cfg["time_col"],
         metrics_ring=cfg["metrics_ring"],
         results_dir=cfg["results_dir"],
     )