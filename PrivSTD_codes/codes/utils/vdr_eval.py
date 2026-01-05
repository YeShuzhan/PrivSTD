
"""
vdr_eval.py — Evaluation utilities for spatio-temporal DP releases.

Implements the three evaluation families used in the VDR paper:
  (1) Range-count Relative Error (RE@sm).
  (2) Hotspot search: distance MAE (meters) & regret (absolute and relative).
  (3) Short-term forecasting (Theta-like SES+drift) scored with sMAPE.

All functions operate on 3D numpy arrays shaped (X, Y, T).
"""

from typing import List, Tuple, Optional, Sequence, Dict, Any
import numpy as np
import math

# -----------------------------
# Helpers
# -----------------------------

def _ensure_3d(a: np.ndarray) -> np.ndarray:
    if a.ndim != 3:
        raise ValueError(f"Expected a 3D array (X,Y,T), got shape {a.shape}")
    return a

def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float((2.0 * np.abs(y_pred - y_true) / denom).mean())

def _relative_error(true_vals: np.ndarray, pred_vals: np.ndarray, sm: float = 20.0) -> float:
    true_vals = np.asarray(true_vals, dtype=float)
    pred_vals = np.asarray(pred_vals, dtype=float)
    denom = np.maximum(true_vals, sm)
    return float((np.abs(pred_vals - true_vals) / denom).mean())

def _topk_indices(a2d: np.ndarray, k: int) -> np.ndarray:
    """Return k indices (row-major flattened) of the largest values in a 2D array."""
    flat = a2d.ravel()
    if k >= flat.size:
        return np.argsort(-flat)  # all
    # Use argpartition for efficiency, then sort the top-k
    topk_part = np.argpartition(-flat, kth=k-1)[:k]
    order = np.argsort(-flat[topk_part])
    return topk_part[order]

def _flat_to_xy(idx: np.ndarray, shape_xy: Tuple[int, int]) -> np.ndarray:
    X, Y = shape_xy
    x = idx // Y
    y = idx % Y
    return np.stack([x, y], axis=1)  # (k, 2)

def _greedy_match_distance(A: np.ndarray, B: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Greedy one-to-one matching by nearest neighbor (not globally optimal but fast and SciPy-free).
    Returns (mean_distance, pairing_indices) where pairing_indices is an array of length len(A)
    with indices into B for the assigned partner (or -1 if B gets exhausted).
    """
    if len(A) == 0 or len(B) == 0:
        return float('nan'), np.array([], dtype=int)
    unused_B = set(range(len(B)))
    pair = np.full(len(A), -1, dtype=int)
    dists = []
    for i, a in enumerate(A):
        # find nearest unused B
        best_j = -1
        best_d2 = None
        for j in unused_B:
            d2 = (A[i,0]-B[j,0])**2 + (A[i,1]-B[j,1])**2
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_j = j
        if best_j >= 0:
            unused_B.remove(best_j)
            pair[i] = best_j
            dists.append(math.sqrt(best_d2))
    if len(dists) == 0:
        return float('inf'), pair
    return float(np.mean(dists)), pair

# -----------------------------
# (1) Range-count Relative Error
# -----------------------------

def sample_range_queries(
    shape: Tuple[int, int, int],
    windows: Sequence[Tuple[int, int, int]],
    n_per_size: int = 1000,
    seed: Optional[int] = 0,
) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Randomly sample rectangular range queries across space-time.

    Args:
        shape: (X, Y, T)
        windows: a list of (wx, wy, wt) window sizes.
        n_per_size: number of queries to sample per window size.
        seed: RNG seed.

    Returns:
        List of queries, each as (x0, x1, y0, y1, t0, t1) with x1,y1,t1 exclusive.
    """
    X, Y, T = shape
    rng = np.random.default_rng(seed)
    queries = []
    for (wx, wy, wt) in windows:
        if wx < 1 or wy < 1 or wt < 1:
            raise ValueError("Window sizes must be >= 1")
        if wx > X or wy > Y or wt > T:
            continue  # skip impossible windows
        xs = rng.integers(0, X - wx + 1, size=n_per_size)
        ys = rng.integers(0, Y - wy + 1, size=n_per_size)
        ts = rng.integers(0, T - wt + 1, size=n_per_size)
        for i in range(n_per_size):
            x0, y0, t0 = int(xs[i]), int(ys[i]), int(ts[i])
            queries.append((x0, x0+wx, y0, y0+wy, t0, t0+wt))
    return queries

def eval_range_relative_error(
    counts_true: np.ndarray,
    counts_pred: np.ndarray,
    queries: Sequence[Tuple[int,int,int,int,int,int]],
    sm: float = 20.0,
) -> Dict[str, Any]:
    """
    Compute RE@sm over a list of range queries.

    Returns a dict with overall mean, per-query errors, and (optionally) per-window-size group means
    if queries contain a 'window_size' tag in an accompanying array.
    """
    counts_true = _ensure_3d(counts_true)
    counts_pred  = _ensure_3d(counts_pred)
    errs = []
    trues = []
    preds = []
    for (x0,x1,y0,y1,t0,t1) in queries:
        true_sum = counts_true[x0:x1, y0:y1, t0:t1].sum()
        pred_sum = counts_pred[x0:x1, y0:y1, t0:t1].sum()
        err = abs(pred_sum - true_sum) / max(true_sum, sm)
        errs.append(err); trues.append(true_sum); preds.append(pred_sum)
    errs = np.array(errs, dtype=float)
    return {
        "RE@sm_mean": float(errs.mean()) if len(errs) else float("nan"),
        "RE@sm_per_query": errs,
        "true_sums": np.array(trues, dtype=float),
        "pred_sums": np.array(preds, dtype=float),
    }

# -----------------------------
# (2) Hotspot Search
# -----------------------------

def eval_hotspots(
    counts_true: np.ndarray,
    counts_pred: np.ndarray,
    topk: int = 50,
    times: Optional[Sequence[int]] = None,
    meters_per_cell: float = 30.0,
    match_mode: str = "greedy",
) -> Dict[str, Any]:
    """
    Evaluate hotspot recovery per time slice:
      - distance MAE (meters) between predicted and true hotspots (one-to-one matched)
      - regret (absolute): sum_true(topk) - sum_true(at predicted topk)
      - regret (relative): regret_abs / max(sum_true(topk), 1e-8)

    Args:
        topk: number of hotspots per time slice.
        times: which t indices to evaluate. Default: all.
        meters_per_cell: spatial resolution to translate grid distance to meters.
        match_mode: 'greedy' (SciPy-free) or 'hungarian' (requires SciPy). If 'hungarian'
                    is requested but SciPy isn't available, it falls back to greedy.

    Returns:
        Dict with per-time metrics and overall means.
    """
    counts_true = _ensure_3d(counts_true)
    counts_pred = _ensure_3d(counts_pred)
    X, Y, T = counts_true.shape
    if times is None:
        times = list(range(T))

    dists = []
    regrets_abs = []
    regrets_rel = []

    # Optional Hungarian support
    can_hungarian = False
    if match_mode == "hungarian":
        try:
            from scipy.optimize import linear_sum_assignment  # type: ignore
            can_hungarian = True
        except Exception:
            can_hungarian = False

    per_t = []
    for t in times:
        true2d = counts_true[:, :, t]
        pred2d = counts_pred[:, :, t]

        idx_true = _topk_indices(true2d, topk)
        idx_pred = _topk_indices(pred2d, topk)

        xy_true = _flat_to_xy(idx_true, (X, Y))
        xy_pred = _flat_to_xy(idx_pred, (X, Y))

        if len(xy_true) == 0 or len(xy_pred) == 0:
            per_t.append({"t": t, "dist_mae_m": float("nan"), "regret_abs": float("nan"), "regret_rel": float("nan")})
            continue

        # Distance MAE via matching
        if can_hungarian and match_mode == "hungarian":
            from scipy.optimize import linear_sum_assignment  # type: ignore
            # Build cost matrix of Euclidean distances
            A = xy_pred.astype(float)
            B = xy_true.astype(float)
            cost = np.sqrt(
                (A[:, None, 0] - B[None, :, 0])**2 +
                (A[:, None, 1] - B[None, :, 1])**2
            )
            k_eff = min(len(A), len(B))
            row_ind, col_ind = linear_sum_assignment(cost[:k_eff, :k_eff])
            mean_grid_dist = float(cost[row_ind, col_ind].mean())
        else:
            mean_grid_dist, _ = _greedy_match_distance(xy_pred, xy_true)

        dist_mae_m = mean_grid_dist * meters_per_cell

        # Regret
        true_topk_vals = np.sort(true2d.ravel())[::-1][:len(idx_true)]
        sum_true_topk = float(true_topk_vals.sum())

        # True counts at predicted top-k locations
        xy_pred_x = xy_pred[:, 0]
        xy_pred_y = xy_pred[:, 1]
        true_at_pred = true2d[xy_pred_x, xy_pred_y]
        sum_true_at_pred = float(true_at_pred.sum())

        regret_abs = max(sum_true_topk - sum_true_at_pred, 0.0)
        regret_rel = regret_abs / (sum_true_topk + 1e-8)

        dists.append(dist_mae_m)
        regrets_abs.append(regret_abs)
        regrets_rel.append(regret_rel)

        per_t.append({
            "t": t,
            "dist_mae_m": dist_mae_m,
            "regret_abs": regret_abs,
            "regret_rel": regret_rel,
            "sum_true_topk": sum_true_topk,
            "sum_true_at_pred": sum_true_at_pred,
        })

    return {
        "dist_mae_m_mean": float(np.mean(dists)) if len(dists) else float("nan"),
        "regret_abs_mean": float(np.mean(regrets_abs)) if len(regrets_abs) else float("nan"),
        "regret_rel_mean": float(np.mean(regrets_rel)) if len(regrets_rel) else float("nan"),
        "per_time": per_t,
        "config": {"topk": topk, "times": list(times), "meters_per_cell": meters_per_cell, "match_mode": ("hungarian" if can_hungarian and match_mode == "hungarian" else "greedy")}
    }

# -----------------------------
# (3) Short-term Forecasting (Theta-like SES+drift) with sMAPE
# -----------------------------

def _ses_forecast(y: np.ndarray, h: int, alpha: float, l0: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Simple Exponential Smoothing (SES) one-step updates; returns h-step-ahead constant forecast and final level.
    """
    if l0 is None:
        l = float(y[0])
    else:
        l = float(l0)
    for t in range(1, len(y)):
        l = alpha * float(y[t]) + (1.0 - alpha) * l
    f = np.full(h, l, dtype=float)  # SES forecast is flat
    return f, l

def _theta_like_forecast(y: np.ndarray, h: int, alphas: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    Lightweight Theta-style forecast: SES with tuned alpha + drift (estimated by simple linear slope).
    This is a practical approximation to the classical Theta method.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 3:
        # too short: fallback to naive last-value
        return np.full(h, y[-1] if n else 0.0, dtype=float)

    if alphas is None:
        alphas = np.linspace(0.05, 0.95, 19)

    # Tune alpha by one-step-ahead SSE on the training series
    best_alpha = None
    best_sse = None
    for a in alphas:
        # simulate SES one-step forecasts
        l = float(y[0])
        sse = 0.0
        for t in range(1, n):
            # one-step forecast is l (prev level)
            yhat = l
            err = y[t] - yhat
            sse += err * err
            l = a * y[t] + (1.0 - a) * l
        if best_sse is None or sse < best_sse:
            best_sse = sse
            best_alpha = a

    # Get SES forecast baseline
    f_ses, lT = _ses_forecast(y, h, alpha=best_alpha)

    # Add a simple drift estimated by OLS slope on the full series
    # slope = cov(t, y) / var(t), with t = 0..n-1
    t = np.arange(n, dtype=float)
    t_mean = (n - 1) / 2.0
    y_mean = float(y.mean())
    cov_ty = float(((t - t_mean) * (y - y_mean)).sum())
    var_t = float(((t - t_mean) ** 2).sum()) + 1e-12
    slope = cov_ty / var_t

    # Theta-like forecast: flat SES baseline plus linear drift
    h_idx = np.arange(1, h + 1, dtype=float)
    f = f_ses + slope * h_idx
    return f

def eval_theta_forecast_smape(
    counts_train_released: np.ndarray,
    counts_test_true: np.ndarray,
    cells: Optional[Sequence[Tuple[int,int]]] = None,
    horizon: int = 1,
    max_cells: int = 1000,
    seed: Optional[int] = 0,
) -> Dict[str, Any]:
    """
    Train per-cell Theta-like model on released (post-processed) training data,
    forecast 'horizon' steps, and score sMAPE against the ground-truth test series.

    Args:
        counts_train_released: (X, Y, T_train)
        counts_test_true:      (X, Y, T_test) -- ground-truth future horizon >= 'horizon'
        cells: list of (x, y) cells to evaluate. If None, sample up to max_cells at random.
        horizon: forecast horizon (e.g., 1 for next slice).
        max_cells: maximum number of cells to evaluate if cells is None.
        seed: RNG seed for random cell sampling.

    Returns:
        Dict with mean sMAPE and per-cell details.
    """
    counts_train_released = _ensure_3d(counts_train_released)
    counts_test_true = _ensure_3d(counts_test_true)
    X, Y, T_tr = counts_train_released.shape
    Xt, Yt, T_te = counts_test_true.shape
    if (X, Y) != (Xt, Yt):
        raise ValueError("Train and test grids must have the same spatial shape.")
    if T_te < horizon:
        raise ValueError("Test horizon is shorter than 'horizon'.")

    # Choose cells
    if cells is None:
        rng = np.random.default_rng(seed)
        all_cells = [(i, j) for i in range(X) for j in range(Y)]
        if len(all_cells) > max_cells:
            cells = rng.choice(len(all_cells), size=max_cells, replace=False)
            cells = [all_cells[int(k)] for k in cells]
        else:
            cells = all_cells

    smapes = []
    per_cell = []
    for (i, j) in cells:
        y_train = counts_train_released[i, j, :]
        y_test = counts_test_true[i, j, :horizon]
        f = _theta_like_forecast(y_train, h=horizon)
        s = _smape(y_test, f)
        smapes.append(s)
        per_cell.append({"cell": (i, j), "sMAPE": float(s)})

    return {
        "sMAPE_mean": float(np.mean(smapes)) if len(smapes) else float("nan"),
        "per_cell": per_cell,
        "config": {"horizon": horizon, "num_cells": len(cells)}
    }

# -----------------------------
# Convenience wrapper to run all three families
# -----------------------------

def eval_all(
    counts_true_full: np.ndarray,
    counts_pred_full: np.ndarray,
    windows: Sequence[Tuple[int,int,int]] = ((1,1,1),(2,2,1),(4,4,1),(8,8,1),(16,16,1)),
    n_per_size: int = 1000,
    sm: float = 20.0,
    topk: int = 50,
    meters_per_cell: float = 30.0,
    forecast_horizon: int = 1,
    forecast_train_T: Optional[int] = None,
    max_cells_forecast: int = 1000,
    seed: Optional[int] = 0,
) -> Dict[str, Any]:
    """
    Run range RE@sm, hotspot metrics, and Theta-like sMAPE in one go.

    Args:
        counts_true_full: (X,Y,T) ground-truth counts.
        counts_pred_full: (X,Y,T) released/postprocessed counts.
        windows / n_per_size / sm: settings for range RE.
        topk / meters_per_cell: settings for hotspot metrics.
        forecast_horizon / forecast_train_T / max_cells_forecast: settings for forecasting.
            - forecast_train_T: if None, split 70/30 for train/test along T.
            - Otherwise, first 'forecast_train_T' steps are training; next 'forecast_horizon' steps must be available.
    Returns:
        Dict with keys: 'range_re', 'hotspots', 'forecast'.
    """
    counts_true_full = _ensure_3d(counts_true_full)
    counts_pred_full = _ensure_3d(counts_pred_full)
    X, Y, T = counts_true_full.shape
    if counts_pred_full.shape != (X, Y, T):
        raise ValueError("counts_true_full and counts_pred_full must have the same shape.")

    # (1) Range RE
    queries = sample_range_queries((X,Y,T), windows=windows, n_per_size=n_per_size, seed=seed)
    range_re = eval_range_relative_error(counts_true_full, counts_pred_full, queries, sm=sm)

    # (2) Hotspots (on all time slices by default)
    hotspots = eval_hotspots(counts_true_full, counts_pred_full, topk=topk, times=None, meters_per_cell=meters_per_cell, match_mode="greedy")

    # (3) Forecast (Theta-like)
    if forecast_train_T is None:
        # 70/30 split with at least 'forecast_horizon' test steps
        train_T = max(int(T * 0.7), 1)
        test_T = T - train_T
        if test_T < forecast_horizon:
            # fallback: shrink horizon to available length
            forecast_horizon = test_T
            if forecast_horizon <= 0:
                raise ValueError("Time axis too short for forecasting.")
        counts_train = counts_pred_full[:, :, :train_T]
        counts_test  = counts_true_full[:, :, train_T:train_T+forecast_horizon]
    else:
        train_T = forecast_train_T
        if train_T <= 0 or train_T >= T:
            raise ValueError("Invalid forecast_train_T given current T.")
        if T - train_T < forecast_horizon:
            raise ValueError("Not enough future steps after forecast_train_T for the requested horizon.")
        counts_train = counts_pred_full[:, :, :train_T]
        counts_test  = counts_true_full[:, :, train_T:train_T+forecast_horizon]

    forecast = eval_theta_forecast_smape(
        counts_train_released=counts_train,
        counts_test_true=counts_test,
        horizon=forecast_horizon,
        max_cells=max_cells_forecast,
        seed=seed,
    )

    return {
        "range_re": range_re,
        "hotspots": hotspots,
        "forecast": forecast,
        "config": {
            "windows": list(windows),
            "n_per_size": n_per_size,
            "sm": sm,
            "topk": topk,
            "meters_per_cell": meters_per_cell,
            "forecast_horizon": forecast_horizon,
            "train_T": train_T,
        }
    }
