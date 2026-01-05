import numpy as np 
from typing import List, Tuple, Sequence, Optional, Dict, Any
import warnings
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==== Configurable evaluation switches (modify as needed) ====
SFORECAST_SMAPE_MODE = "strict"   # "strict" or "robust"
CLIP_NEG_BEFORE_EVAL = False      # Whether to clip predictions to non-negative before evaluation
THETA_HIST_LEN = 100              # Maximum history length used by ThetaModel

# ==== Optional dependencies: fallback if unavailable ====
try:
    from statsmodels.tsa.forecasting.theta import ThetaModel
    from statsmodels.tsa.stattools import acf
    from sklearn.metrics import mean_absolute_error
    _HAVE_TS = True
except Exception:
    _HAVE_TS = False

# =========================
# Existing: point query evaluation
# =========================
def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def RE(y_true, y_pred, sm):
    return np.mean([np.abs(y_true[i] - y_pred[i]) / max(y_true[i], sm) for i in range(len(y_true))])

def get_eval_results(counts, data_rec, test_samples: List[Tuple[int, int, int]], sm=None):
    """
    Point (cell) query evaluation: MAE and RE@sm list
    counts/data_rec shape: [H, W, T]
    """
    if sm is None:
        sm = [5, 20]
    y_true = np.array([counts[tuple(sample)] for sample in test_samples])
    y_pred = np.array([data_rec[tuple(sample)] for sample in test_samples])
    mae = MAE(y_true, y_pred)
    re = []
    for sm_ in sm:
        re.append(RE(y_true, y_pred, sm_))
    return mae, re

# ====== Two sMAPE conventions (strict / robust) ======
def smape_strict(A, F):
    """
    STRICT: do not add 1 to the denominator and do not force 0/0 to 0;
    for denom==0 terms use NaN and take nanmean.
    This typically yields a higher sMAPE and aligns with many external implementations.
    """
    A = np.asarray(A, dtype=float); F = np.asarray(F, dtype=float)
    denom = np.abs(A) + np.abs(F)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = 2.0 * np.abs(F - A) / denom
    return float(np.nanmean(r))

def smape_robust(A, F, eps=0.0, zero_as_zero=True):
    """
    ROBUST: treat 0/0 as 0; otherwise compute with the standard denominator (or add eps).
    Usually lower and more stable than strict.
    """
    A = np.asarray(A, dtype=float); F = np.asarray(F, dtype=float)
    denom = np.abs(A) + np.abs(F)
    num = 2.0 * np.abs(F - A)
    if zero_as_zero:
        mask = (denom == 0.0)
        num = num.copy()
        num[mask] = 0.0
        denom = np.where(denom == 0.0, 1.0, denom)
        return float(np.mean(num / denom))
    else:
        return float(np.mean(num / np.maximum(denom, eps)))

def _smape_dispatch(A, F, mode: str):
    if mode == "strict":
        return smape_strict(A, F)
    elif mode == "robust":
        return smape_robust(A, F, eps=1e-12, zero_as_zero=True)
    else:
        raise ValueError(f"Unknown sMAPE mode: {mode}")

# ====== Anisotropic grid mapping: x/y use rho_xy, z(time) uses rho_t ======
def convert_db_to_ijk_aniso(db: np.ndarray, rho_xy: float, rho_t: float, max_val: float):
    N, D = db.shape
    assert D == 3, "expect 3D points"
    bins_x = np.arange(start=-max_val/2, stop=max_val/2, step=rho_xy)
    bins_y = np.arange(start=-max_val/2, stop=max_val/2, step=rho_xy)
    bins_t = np.arange(start=-max_val/2, stop=max_val/2, step=rho_t)
    idx_x = np.searchsorted(bins_x, db[:, 0], side='right') - 1
    idx_y = np.searchsorted(bins_y, db[:, 1], side='right') - 1
    idx_t = np.searchsorted(bins_t, db[:, 2], side='right') - 1
    return (idx_x, idx_y, idx_t)

def autocorrelation_seasonality_test(y, sp):
    if not _HAVE_TS:
        return True
    n = len(y)
    if n < 3 * sp: 
        return False
    coefs = acf(y, nlags=sp, fft=False)
    coef  = coefs[sp]
    tcrit = 1.645
    limits = tcrit/np.sqrt(n) * np.sqrt(np.cumsum(np.append(1, 2 * coefs[1:]**2)))
    return abs(coef) > limits[sp-1]

# ====== Unified forecast single-query evaluation (strictly on-grid) ======
def evaluate_forecasting_query(ts, ts_true, pt, fh, period,
                               smape_mode: str = SFORECAST_SMAPE_MODE,
                               clip_neg: bool = CLIP_NEG_BEFORE_EVAL,
                               hist_len: int = THETA_HIST_LEN):
    """
    On-grid evaluation:
      - ts/ts_true are the full time series for the same (i,j) (prediction / ground truth),
        pt is the start index, fh is the horizon
      - predictor: prefer ThetaModel(hist), fallback to naive (repeat last)
      - no scaling, no aggregation, no resampling; optionally clip negatives
      - sMAPE supports strict/robust
    """
    if pt >= len(ts_true) - fh or pt < 0:
        return float("inf"), float("inf")

    hist = ts[max(0, pt - hist_len):pt].astype(float)
    y_true = ts_true[pt:pt+fh].astype(float)

    # Forecast
    if _HAVE_TS and len(hist) > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                y_pred = ThetaModel(hist, period=period).fit().forecast(fh)
            except Exception:
                last = float(hist[-1]) if len(hist) else 0.0
                y_pred = np.full((fh,), last, dtype=float)
    else:
        last = float(hist[-1]) if len(hist) else 0.0
        y_pred = np.full((fh,), last, dtype=float)

    # Optional clip before evaluation
    if clip_neg:
        y_pred = np.clip(y_pred, 0.0, None)

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mape = _smape_dispatch(y_true, y_pred, smape_mode)
    return mae, mape

def compute_data_filled_slices_from_counts(H3):
    nz = np.where(H3.sum(axis=(0,1)) > 0)[0]
    return int(nz[-1]) if nz.size else 0

def get_queries_for_forecasting_vdr_exact(
    datafile: str, max_val: float, min_vals: np.ndarray, max_vals: np.ndarray,
    rho_xy: float, rho_t: float, test_size: int,
    H: np.ndarray, data_filled_slices: int, fh: int = 3,
    smape_mode: str = SFORECAST_SMAPE_MODE
):
    Hs, Ws, Ts = H.shape
    raw = np.load(datafile)
    test_loc = ((raw - min_vals)/(max_vals - min_vals) - 0.5) * max_val
    rng = np.random.default_rng(2024); rng.shuffle(test_loc)

    # Time window constraint (using rho_t)
    keep_low  = -max_val/2 + (data_filled_slices - 3*fh) * rho_t
    keep_high = -max_val/2 + (data_filled_slices - fh)   * rho_t
    test_loc = test_loc[(test_loc[:,2] > keep_low) & (test_loc[:,2] < keep_high)]
    test_loc = test_loc[:50 * test_size]

    # Continuous -> grid: x/y use rho_xy, t uses rho_t
    ijk = convert_db_to_ijk_aniso(test_loc, rho_xy=rho_xy, rho_t=rho_t, max_val=max_val)
    ijk_arr = np.asarray(list(zip(*ijk)), dtype=int)

    fcast_qs, H_mae, H_mape = [], [], []
    trial = 0
    for (i, j, t) in ijk_arr:
        trial += 1
        if not (0 <= i < Hs and 0 <= j < Ws and 0 <= t < Ts - fh):
            continue
        ts = H[i, j].astype(float)
        seg = ts[:t+fh]
        # Guard 1: segment length must be sufficient
        if seg.size < (2 * fh + 1):
            continue

        # Guard 2: zero (or near-zero) variance -> skip (would make ACF divide by zero)
        if not np.isfinite(seg).all():
            continue
        if np.var(seg) <= 1e-12:
            continue

        # Guard 3 (optional but useful): too few non-zero points is not meaningful
        # For count data, many all-zero series will be filtered here
        if np.count_nonzero(seg) < (fh + 1):
            continue
        if not autocorrelation_seasonality_test(ts[:t+fh], fh):
            continue
        mae, mape = evaluate_forecasting_query(ts, ts, pt=t, fh=fh, period=fh,
                                               smape_mode=smape_mode,
                                               clip_neg=CLIP_NEG_BEFORE_EVAL)
        if not np.isfinite(mape) or mape > 0.6:
            continue
        fcast_qs.append((int(i), int(j), int(t)))
        H_mae.append(mae); H_mape.append(mape)
        if len(fcast_qs) == 100:
            break

    mae_mean  = float(np.mean(H_mae))  if H_mae else float("nan")
    mape_mean = float(np.mean(H_mape)) if H_mape else float("nan")
    return fcast_qs, mae_mean, mape_mean

# ====== Hotspot evaluation (unchanged) ======
def fast_answer_hostspot_queries(_hot, lb, ru, x_hat, _loc):
    _grid = x_hat[lb[0]:ru[0], lb[1]:ru[1], lb[2]:ru[2]]
    _mask = (_grid >= _hot)
    if np.sum(_mask) == 0:
        _idx = np.unravel_index(np.argmax(_grid, axis=None), _grid.shape)
        return tuple((_idx + lb).tolist())
    _mask = np.transpose(np.nonzero(_mask)) + lb
    dist = np.linalg.norm(_mask - _loc, axis=-1)
    _ans_c = _mask[np.argmin(dist)]
    return tuple(_ans_c.tolist())

def get_hotspot_queries(_hot, _grid, loc_ijk_arr, limit=500, radius=50):
    hotspot_queries, H_c, H_ress = [], [], []
    shape = np.array(_grid.shape)
    for _loc in loc_ijk_arr:
        _loc = np.asarray(_loc, dtype=int)
        lb = np.minimum(np.maximum(0, _loc - radius), shape)
        ru = np.minimum(np.maximum(0, _loc + radius + 1), shape)
        sub = _grid[lb[0]:ru[0], lb[1]:ru[1], lb[2]:ru[2]]
        _mask = (sub >= _hot)
        if np.sum(_mask) == 0:
            continue
        idxs = np.transpose(np.nonzero(_mask)) + lb
        dist = np.linalg.norm(idxs - _loc, axis=-1)
        pick = idxs[np.argmin(dist)]
        H_c.append(tuple(pick.tolist()))
        H_ress.append(float(np.min(dist)))
        hotspot_queries.append(_loc)
        if len(hotspot_queries) == limit:
            break
    return hotspot_queries, H_c, H_ress

def gather_hotspot_results(reco_grid, H, hot_levels, H_slow_qs, Hress_slow_payload, radius=50):
    out = {'mae':{}, 'regret':{}, 'nq':{}}
    shape = np.array(reco_grid.shape)
    for _hot in hot_levels:
        _qs = H_slow_qs[_hot]
        if len(_qs) == 0:
            out['mae'][_hot] = float('nan'); out['regret'][_hot] = float('nan'); out['nq'][_hot] = 0
            continue
        _c_er = 0.0
        ID_c = []
        for _loc in _qs:
            _loc = np.asarray(_loc, dtype=int)
            lb = np.minimum(np.maximum(0, _loc - radius), shape)
            ru = np.minimum(np.maximum(0, _loc + radius + 1), shape)
            _ans_c = fast_answer_hostspot_queries(_hot, lb, ru, reco_grid, _loc)
            ID_c.append(_ans_c)
            if H[_ans_c] < _hot:
                _c_er += (_hot - float(H[_ans_c]))
        ID_c = np.asarray(ID_c, dtype=int)
        IDQ_res = np.linalg.norm(ID_c - _qs, axis=-1).reshape(-1,1)
        HQ_res = Hress_slow_payload[_hot]
        hot_mae = float(np.average(np.abs(IDQ_res - HQ_res), axis=0)[0])
        hot_reg = float(_c_er / max(len(ID_c), 1))
        out['mae'][_hot] = hot_mae; out['regret'][_hot] = hot_reg; out['nq'][_hot] = int(len(ID_c))
    return out

# ====== Unified on-grid forecast aggregate evaluation ======
def gather_forecasting_results(reco_grid, H, fcast_qs, fh,
                               smape_mode: str = SFORECAST_SMAPE_MODE,
                               clip_neg: bool = CLIP_NEG_BEFORE_EVAL):
    maes, smapes = [], []
    for (i, j, t) in fcast_qs:
        ts = reco_grid[i, j].astype(float)
        ts_true = H[i, j].astype(float)
        mae, mape = evaluate_forecasting_query(ts, ts_true, pt=t, fh=fh, period=fh,
                                               smape_mode=smape_mode,
                                               clip_neg=clip_neg)
        if np.isfinite(mae) and np.isfinite(mape):
            maes.append(mae); smapes.append(mape)
    if len(maes) == 0:
        return float('nan'), float('nan'), 0
    return float(np.mean(maes)), float(np.mean(smapes)), len(maes)

# ====== (Optional) Audit function: helps align conventions ======
def audit_forecast_metrics(H, reco_grid, fcast_qs, fh,
                           smape_mode: str = SFORECAST_SMAPE_MODE,
                           clip_neg: bool = CLIP_NEG_BEFORE_EVAL):
    rows = []
    for (i, j, t) in fcast_qs:
        y_true = H[i, j][t:t+fh].astype(float)
        y_pred = reco_grid[i, j][t:t+fh].astype(float)
        if clip_neg:
            y_pred = np.clip(y_pred, 0.0, None)
        mae  = float(np.mean(np.abs(y_true - y_pred)))
        smap = _smape_dispatch(y_true, y_pred, smape_mode)
        scale = float(np.mean(np.abs(y_true)))
        rows.append((scale, mae, smap))
    A = np.array(rows) if rows else np.zeros((0,3))
    if len(A):
        print(f"[audit] N={len(A)}  mean|A|={A[:,0].mean():.3f }  "
              f"corr(|A|, MAE)={np.corrcoef(A[:,0],A[:,1])[0,1]:.3f}  "
              f"corr(|A|, sMAPE)={np.corrcoef(A[:,0],A[:,2])[0,1]:.3f}")
        print(f"[audit] MAE mean={A[:,1].mean():.4f}  sMAPE mean={A[:,2].mean():.4f}")
    else:
        print("[audit] no effective queries")





# ====== Range Count Query evaluation (added) ======

def get_all_cells(lb_idx, ru_idx):
    """
    Get all cell coordinates within the query range.
    lb_idx: lower-bound corner index (i_min, j_min, t_min)
    ru_idx: upper-bound corner index (i_max, j_max, t_max)
    Returns: list of all cell coordinates
    """
    _D = len(lb_idx)
    
    def cartesianProduct(one, two):
        result = []
        for v1 in one:
            for v2 in two:
                result.append([v1, v2])
        return result
    
    def flatten2list(obj):
        gather = []
        for item in obj:
            if isinstance(item, (list, tuple, set)):
                gather.extend(flatten2list(item))
            else:
                gather.append(item)
        return gather
    
    result = range(lb_idx[0], ru_idx[0])
    for i in range(1, _D):
        result = cartesianProduct(result, range(lb_idx[i], ru_idx[i]))
    
    result = [tuple(flatten2list(x)) for x in result]
    return result


def answer_queries_from_grid(grid_pred, test_loc_ijk, test_loc_ijk_ru):
    """
    Answer range queries from the predicted grid.
    grid_pred: [H, W, T] predicted count grid
    test_loc_ijk: list of lower-bound coordinates for queries
    test_loc_ijk_ru: list of upper-bound coordinates for queries
    Returns: query results [num_queries, 1]
    """
    query_results = []
    for lb, ru in zip(test_loc_ijk, test_loc_ijk_ru):
        cell_list = get_all_cells(lb, ru)
        count_sum = 0
        for cell in cell_list:
            count_sum += grid_pred[cell]
        query_results.append(count_sum)
    
    return np.array(query_results).reshape(-1, 1)


def generate_range_queries_vdr_style(
    counts: np.ndarray,
    noisy_counts: np.ndarray,
    datafile: str,
    max_val: float,
    min_vals: np.ndarray,
    max_vals: np.ndarray,
    rho_xy: float,
    rho_t: float,
    augmented_query_size: np.ndarray,  # [size_x, size_y, size_t]
    test_size: int = 2000,
    num_snaps: int = -1,
    random_seed: int = 2024
):
    H, W, T = counts.shape
    
    # Compute query size (number of cells)
    answer_len = np.rint(augmented_query_size / np.array([rho_xy, rho_xy, rho_t])).astype(int)
    
    # Load and normalize data points
    raw = np.load(datafile)
    test_loc = ((raw - min_vals) / (max_vals - min_vals) - 0.5) * max_val
    
    # Shuffle order
    rng = np.random.default_rng(random_seed)
    rng.shuffle(test_loc)
    
    # Filter: ensure the query does not go out of bounds
    test_loc = test_loc[
        np.logical_and.reduce([
            test_loc[:, 0] < (max_val/2 - augmented_query_size[0] - rho_xy),
            test_loc[:, 1] < (max_val/2 - augmented_query_size[1] - rho_xy),
            test_loc[:, 2] < (max_val/2 - augmented_query_size[2]),
        ])
    ]
    
    # If the number of time slices is specified, further filter
    if num_snaps != -1:
        test_loc = test_loc[
            test_loc[:, 2] < (max_val/2 - augmented_query_size[2] - (T - num_snaps) * rho_t)
        ]
    
    # Oversample
    test_loc = test_loc[:test_size * 100]
    
    # Convert to grid coordinates
    test_loc_ijk = convert_db_to_ijk_aniso(test_loc, rho_xy=rho_xy, rho_t=rho_t, max_val=max_val)
    test_loc_ijk_arr = np.array(list(zip(*test_loc_ijk)))
    
    # Compute upper-right coordinates
    test_loc_ijk_ru_arr = test_loc_ijk_arr + answer_len
    
    # Answer queries
    identity_test_ress = []
    test_res = []
    valid_lb = []
    valid_ru = []
    
    for lb, ru in zip(test_loc_ijk_arr, test_loc_ijk_ru_arr):
        # Get all cells within the query range
        cell_list = get_all_cells(tuple(lb), tuple(ru))
        
        # Sum over noisy_counts and counts
        noisy_sum = 0
        true_sum = 0
        for cell in cell_list:
            try:
                noisy_sum += noisy_counts[cell]
                true_sum += counts[cell]
            except IndexError:
                continue
        
        identity_test_ress.append(noisy_sum)
        test_res.append(true_sum)
        valid_lb.append(tuple(lb))
        valid_ru.append(tuple(ru))
        
        if len(valid_lb) >= test_size:
            break
    
    identity_test_ress = np.array(identity_test_ress).reshape(-1, 1)
    test_res = np.array(test_res).reshape(-1, 1)
    
    print(f'[Range Query] Generated {len(valid_lb)} queries')
    print(f'[Range Query] Answer stats - min: {test_res.min()}, max: {test_res.max()}, '
          f'mean: {test_res.mean():.2f}, median: {np.median(test_res):.2f}')
    
    return valid_lb, valid_ru, test_res, identity_test_ress


def evaluate_range_queries(
    counts: np.ndarray,
    data_rec: np.ndarray,
    test_loc_ijk,
    test_loc_ijk_ru,
    test_res: np.ndarray,
    sm=None
):
    """
    Evaluate range queries using MAE and RE.
    
    Args:
        counts: ground-truth counts [H, W, T]
        data_rec: predicted/reconstructed counts [H, W, T]
        test_loc_ijk: list of lower-bound query coordinates
        test_loc_ijk_ru: list of upper-bound query coordinates
        test_res: ground-truth query answers [num_queries, 1]
        sm: list of smoothing constants
    
    Returns:
        mae: mean absolute error
        re: list of relative errors (corresponding to each smoothing constant)
    """
    if sm is None:
        sm = [5, 10, 20]
    
    # Answer queries from the predicted grid
    pred_res = answer_queries_from_grid(data_rec, test_loc_ijk, test_loc_ijk_ru)
    
    # Compute MAE
    mae = float(np.mean(np.abs(pred_res - test_res)))
    
    # Compute RE (multiple smoothing constants)
    re = []
    for sm_val in sm:
        re_val = float(np.mean(np.abs(pred_res - test_res) / np.maximum(test_res, sm_val)))
        re.append(re_val)
    
    return mae, re
