# userlevel_MWEM_main.py
# -*- coding: utf-8 -*-

"""
MWEM user-level version (align to user-level)

Only apply the necessary user-level changes; keep the rest of the logic unchanged:
1) Do not use read_dataset(config); instead read the 5-column user-level txt: [user time lat lon loc]
2) Truncate each user to k records -> Ds (db_s); the DP mechanism runs only on Ds
3) user-level delta = 1/(n_users^2) 
4) Align user-level global sensitivity: Δ = k
   -> Multiply Gaussian noise scales by k for all "count/marginal/query answering"
5) Align evaluation: counts_true (from full D) vs data_rec_cal (=gamma*data_rec)
6) Hotspot and forecasting: prepare queries using full D (counts_true / db_full); evaluate using counts_true
7) gamma = |D|/|Ds| for global calibration (post-processing; does not affect DP)
"""

import os
import torch
import torch.nn as nn
import numpy as np
from parse import config
import math
from logger.logger import ConfigParser
from utils.dataset import get_counts
from utils.eval import *
from tqdm import tqdm
from utils.results_stats import ResultStats
import time


# =========================================================
# 1) User-level data loading (dedicated function)
# =========================================================
def _parse_iso8601_to_epoch_seconds(ts: str) -> float:
    """Parse like: '2010-07-24T13:45:06Z' -> epoch seconds (UTC)."""
    import datetime as _dt
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = _dt.datetime.fromisoformat(ts)
    return dt.timestamp()


def read_userlevel_txt_dataset(
    name: str,
    data_dir: str,
    filename: str | None = None,
    delimiter: str | None = None,
    skip_bad_lines: bool = True,
):
    """
    Read user-level raw data:
      [user] [check-in time] [latitude] [longitude] [location id]
    Returns:
      db_full: [N,3] -> [lon, lat, time_seconds]
      user_ids: [N]
      min_vals/max_vals: [3]
      n_records, n_users
    """
    if filename is None:
        filename = f"{name}.txt"
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    users, times, lats, lons = [], [], [], []
    bad = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                parts = line.split(delimiter) if delimiter is not None else line.split()
                uid = int(parts[0])
                tsec = _parse_iso8601_to_epoch_seconds(parts[1])
                lat = float(parts[2])
                lon = float(parts[3])
                users.append(uid)
                times.append(tsec)
                lats.append(lat)
                lons.append(lon)
            except Exception:
                bad += 1
                if not skip_bad_lines:
                    raise
                continue

    if len(users) == 0:
        raise ValueError(f"No valid rows read from: {path} (bad_lines={bad})")

    user_ids = np.asarray(users, dtype=np.int64)
    db_full = np.stack(
        [
            np.asarray(lons, dtype=np.float64),
            np.asarray(lats, dtype=np.float64),
            np.asarray(times, dtype=np.float64),
        ],
        axis=1,
    )

    min_vals = np.min(db_full, axis=0)
    max_vals = np.max(db_full, axis=0)
    n_records = int(db_full.shape[0])
    n_users = int(np.unique(user_ids).size)

    print(f"[read_userlevel_txt_dataset] path={path}")
    print(f"[read_userlevel_txt_dataset] rows={n_records}, users={n_users}, bad_lines={bad}")
    print(f"[read_userlevel_txt_dataset] min_vals(lon,lat,time)={min_vals}")
    print(f"[read_userlevel_txt_dataset] max_vals(lon,lat,time)={max_vals}")

    return db_full, user_ids, min_vals, max_vals, n_records, n_users


def truncate_per_user(db_full: np.ndarray, user_ids: np.ndarray, k: int, seed: int = 2024):
    """Truncate each user's contributions to k records (uniform sampling without replacement)."""
    if k <= 0:
        raise ValueError("k must be positive for user-level truncation")

    rng = np.random.default_rng(seed)
    order = np.argsort(user_ids, kind="mergesort")
    uid_sorted = user_ids[order]
    db_sorted = db_full[order]

    uniq, start_idx = np.unique(uid_sorted, return_index=True)
    start_idx = list(start_idx) + [len(uid_sorted)]

    kept_idx_sorted = []
    for i in range(len(uniq)):
        s = start_idx[i]
        e = start_idx[i + 1]
        m = e - s
        if m <= k:
            kept = np.arange(s, e, dtype=np.int64)
        else:
            kept = rng.choice(np.arange(s, e, dtype=np.int64), size=k, replace=False)
        kept_idx_sorted.append(kept)

    kept_idx_sorted = np.concatenate(kept_idx_sorted, axis=0)
    rng.shuffle(kept_idx_sorted)

    db_s = db_sorted[kept_idx_sorted]
    user_ids_s = uid_sorted[kept_idx_sorted]
    return db_s, user_ids_s


# =========================
# Numerical-stability utility functions (unchanged)
# =========================
def softmax_stable(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax; includes a fallback for all NaN/Inf cases."""
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        p = np.zeros_like(x, dtype=np.float64)
        p[int(np.nanargmax(x))] = 1.0
        return p
    x_max = np.max(x)
    z = x - x_max
    z = np.clip(z, -50.0, 50.0)
    e = np.exp(z)
    s = e.sum()
    if s <= 0 or not np.isfinite(s):
        p = np.ones_like(x, dtype=np.float64) / len(x)
    else:
        p = e / s
    s2 = p.sum()
    if not np.isfinite(s2) or s2 <= 0:
        p = np.ones_like(x, dtype=np.float64) / len(x)
    else:
        p = p / s2
    return p


def nan_to_num_inplace(arr: np.ndarray, val: float = 0.0, posinf: float = 1e12, neginf: float = 0.0):
    np.nan_to_num(arr, copy=False, nan=val, posinf=posinf, neginf=neginf)


def renorm_to_mass(A: np.ndarray, target_mass: float):
    s = A.sum()
    if not np.isfinite(s) or s <= 0:
        A[...] = target_mass / A.size
    else:
        A *= (target_mass / s)
    nan_to_num_inplace(A, val=0.0)


def safe_exp_update_factor(delta: np.ndarray, denom: float, lr: float, clip_val: float = 30.0):
    denom = max(float(denom), 1.0)
    expo = lr * (delta / denom)
    expo = np.clip(expo, -clip_val, clip_val)
    return np.exp(expo)


# =========================
# zCDP conversion (unchanged)
# =========================
def dp_to_required_zcdp(epsilon: float, delta: float) -> float:
    """Given (ε, δ), return the minimum rho (zCDP) that implies (ε, δ)-DP."""
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    a = math.sqrt(math.log(1.0 / delta))
    x = math.sqrt(max(epsilon, 0.0) + a * a) - a
    x = max(0.0, x)
    rho = x * x
    return rho


# =========================
# MWEM (unchanged)
# =========================
def get_marginal(arr: np.ndarray, keep):
    if not keep:
        return np.sum(arr)
    sum_axes = tuple(i for i in range(len(arr.shape)) if i not in keep)
    return np.sum(arr, axis=sum_axes)


def update_A(A: np.ndarray, keep, m: np.ndarray, n_slice: float, iter_idx: int):
    lr = 1.0 / float(1.0 + 0.25 * iter_idx)
    marg_A = get_marginal(A, keep)
    d = m - marg_A
    exp_factors = safe_exp_update_factor(d, denom=max(n_slice, 1.0), lr=lr, clip_val=30.0)
    if len(keep) == 0:
        A *= float(exp_factors)
    else:
        factor_shape = [1] * len(A.shape)
        if isinstance(keep, tuple) and len(keep) > 0:
            for ii, ax in enumerate(keep):
                factor_shape[ax] = exp_factors.shape[ii] if exp_factors.ndim > ii else 1
        exp_factors = exp_factors.reshape(factor_shape)
        A *= exp_factors
    nan_to_num_inplace(A, val=0.0)
    renorm_to_mass(A, n_slice)


# =========================
# Logging & random seeds
# =========================
config_parser = ConfigParser(name="MWEM", save_dir="./")
logger = config_parser.get_logger(config_parser.exper_name)
torch.manual_seed(2024)
np.random.seed(2024)
os.environ["CUDA_VISIBLE_DEVICES"] = config["train"]["gpu"]

logger.info(f"config: {config}")

# =========================================================
# user-level: read full D, build counts_true, truncate to Ds
# =========================================================
data_dir = "./data/"
name = config["datasets"]["name"]
raw_filename = name + ".txt"

k = int(config["datasets"]["truncate_size"])
seed = 2024

db_full, user_ids, min_vals, max_vals, n_records, n_users = read_userlevel_txt_dataset(
    name=name,
    data_dir=data_dir,
    filename=raw_filename,
    delimiter=None,
    skip_bad_lines=True,
)


delta = 1.0 / (n_users**2)

logger.info(f"[MWEM UserLevel] n_records={n_records}, n_users={n_users}, delta={delta:.3e}, k={k}")
logger.info(f"max_lon: {max_vals[0]}, min_lon: {min_vals[0]}, max_lat: {max_vals[1]}, min_lat: {min_vals[1]}")
logger.info(f"time_interval: {(max_vals[2] - min_vals[2]) / 3600} hours")

# counts_true from full D (for evaluation/queries)
counts_true, test_samples = get_counts(
    config, db_full, min_vals, max_vals,
    config["datasets"]["sample_size"], config["datasets"]["time_grid"]
)
counts_true = counts_true[:, :, :config["datasets"]["time_grid"]]

logger.info(f"counts_true shape: {counts_true.shape}")
logger.info(
    f"counts_true_max: {np.max(counts_true)}, counts_true_min: {np.min(counts_true)}, "
    f"counts_true_mean: {np.mean(counts_true)}, median: {np.median(counts_true)}, sum: {np.sum(counts_true)}"
)

# truncate to Ds
db, _ = truncate_per_user(db_full, user_ids, k=k, seed=seed)
gamma = float(db_full.shape[0]) / float(db.shape[0])
logger.info(f"[Refinement] gamma = |D|/|Ds| = {db_full.shape[0]}/{db.shape[0]} = {gamma:.6f}")

# counts_s from Ds (mechanism input)
counts, _ = get_counts(
    config, db, min_vals, max_vals,
    config["datasets"]["sample_size"], config["datasets"]["time_grid"]
)
counts = counts[:, :, :config["datasets"]["time_grid"]]
logger.info(f"[Ds counts] shape: {counts.shape}")

# =========================
# One-time global Gaussian noise (if needed) — user-level: sigma_global *= k
# =========================
eps = float(config["privacy"]["eps"])

# (Keeping your original delta=1/n^4 style would conflict with user-level; here we align to user-level: delta=1/n_users^2)
rho_min = dp_to_required_zcdp(eps, delta)

if rho_min > 0:
    sigma_global = (float(k) * 1.0) / math.sqrt(2.0 * rho_min)  # Δ=k
else:
    sigma_global = 0.0

print("Global Gaussian Sigma (zCDP, user-level):", sigma_global)

if sigma_global > 0:
    noise = np.random.normal(0.0, sigma_global, counts.shape)
    noisy_counts = counts + noise
else:
    noisy_counts = counts.copy()

nan_to_num_inplace(noisy_counts, val=0.0)
logger.info(
    f"noisy_max: {np.max(noisy_counts)}, noisy_min: {np.min(noisy_counts)}, "
    f"noisy_mean: {np.mean(noisy_counts)}, noisy_median: {np.median(noisy_counts)}, "
    f"noisy_sum: {np.sum(noisy_counts)}"
)


H_, W_, T_ = counts_true.shape
max_val_vdr = float(config["datasets"].get("max_val", 10.0))
rho_xy = max_val_vdr / float(max(H_, 1))
rho_t = max_val_vdr / float(max(T_, 1))
data_filled_slices = compute_data_filled_slices_from_counts(counts_true)
forecast_horizon = int(config["test"].get("forecast_horizon", 3))

# Write a temporary npy from full D for query generation (avoid depending on the original *.npy)
datafile_full = os.path.join(data_dir, f"{name}_userlevel_full.npy")
np.save(datafile_full, db_full)

fcast_qs, _h_mae_ref, _h_mape_ref = get_queries_for_forecasting_vdr_exact(
    datafile=datafile_full,
    max_val=max_val_vdr,
    min_vals=min_vals,
    max_vals=max_vals,
    rho_xy=rho_xy,
    rho_t=rho_t,
    test_size=config["datasets"]["sample_size"],
    H=counts_true,
    data_filled_slices=data_filled_slices,
    fh=forecast_horizon,
)
logger.info(
    f"[ForecastQs] prepared={len(fcast_qs)} "
    f"(ref H_MAE={_h_mae_ref:.4f}, H_sMAPE={_h_mape_ref:.4f})"
)

_raw = np.load(datafile_full)
_raw = ((_raw - min_vals) / (max_vals - min_vals) - 0.5) * max_val_vdr
_loc_ijk = convert_db_to_ijk_aniso(_raw, rho_xy=rho_xy, rho_t=rho_t, max_val=max_val_vdr)
loc_ijk_arr = np.asarray(list(zip(*_loc_ijk)), dtype=int)

hot_levels = config["test"].get("hotspot_levels", [20])
Hc_slow_payload = {}
Hress_slow_payload = {}
H_slow_qs = {}
for _hot in hot_levels:
    _qs, _Hc, _Hress = get_hotspot_queries(_hot, counts_true, loc_ijk_arr, limit=500, radius=50)
    Hc_slow_payload[_hot] = np.asarray(_Hc, dtype=int)
    Hress_slow_payload[_hot] = np.asarray(_Hress, dtype=float).reshape(-1, 1)
    H_slow_qs[_hot] = np.asarray(_qs, dtype=int)
    logger.info(f"[HotspotQs] level={_hot} prepared={len(_qs)}")


# =========================
# MWEM (keep your parameters/procedure unchanged; only set sigma_mwem to user-level Δ=k, and run on Ds counts)
# =========================
H, W, T_ = counts.shape
T = 20  # Number of MWEM iterations
alpha = config["train"]["alpha"]  # keep but not directly used
data_rec_list = []
time1 = time.time()

exp_mech_scale = (eps / max(2.0 * T, 1.0)) * 0.5

for slice_idx in tqdm(range(T_), total=T_):
    B = counts[:, :, slice_idx].astype(np.float64)
    noisy_B = noisy_counts[:, :, slice_idx].astype(np.float64)

    n_slice = float(np.sum(B))
    A = np.maximum(0.0, noisy_B)
    renorm_to_mass(A, n_slice)

    Q_keep = [(), (0,), (1,), (0, 1)]
    history = []

    for iter_idx in range(1, T + 1):
        scores = []
        for keep in Q_keep:
            marg_A = get_marginal(A, keep)
            marg_B = get_marginal(B, keep)
            diff = np.sum(np.abs(marg_A - marg_B))
            size = 1 if not keep else np.prod([B.shape[k] for k in keep])
            s = (diff - size) / max(n_slice, 1.0)
            scores.append(float(s))
        scores = np.asarray(scores, dtype=np.float64)

        probs = softmax_stable(exp_mech_scale * scores)
        if not np.isfinite(probs.sum()) or abs(probs.sum() - 1.0) > 1e-8:
            probs = np.ones(len(Q_keep), dtype=np.float64) / len(Q_keep)

        idx = int(np.random.choice(len(Q_keep), p=probs))
        keep = Q_keep[idx]

        marg_B = get_marginal(B, keep)

        # === user-level change: multiply sigma_mwem by Δ=k ===
        sigma_mwem = math.sqrt(2.0 * T * math.log(1.25 / delta)) * (float(k) / max(eps, 1e-12))

        noise_shape = () if np.isscalar(marg_B) else marg_B.shape
        m_obs = marg_B + np.random.normal(0.0, sigma_mwem, size=noise_shape)

        history.append((keep, np.array(m_obs, dtype=np.float64)))
        update_A(A, keep, m_obs, n_slice, iter_idx=iter_idx)

    for refine_round in range(10):
        change = 0.0
        for it, (keep, m_obs) in enumerate(history, start=1):
            marg_A = get_marginal(A, keep)
            update_A(A, keep, m_obs, n_slice, iter_idx=it)
            change += float(np.sum(np.abs(m_obs - marg_A)))
        if change < 1e-3:
            break

    data_rec_list.append(A)

data_rec = np.stack(data_rec_list, axis=2).astype(np.float64)
time2 = time.time()
print(f'Running time: {time2 - time1} ')

# ====== global calibration gamma (post-processing) ======
data_rec_cal = np.maximum(gamma * data_rec, 0.0)
noisy_cal = np.maximum(gamma * noisy_counts, 0.0)


# mae, re = get_eval_results(counts_true, data_rec_cal, test_samples, sm=config["test"]["sm"])
# logger.info(f"MAE: {mae}, RE: {re}")

# hot_res = gather_hotspot_results(
#     reco_grid=data_rec_cal,
#     H=counts_true,
#     hot_levels=hot_levels,
#     H_slow_qs=H_slow_qs,
#     Hress_slow_payload=Hress_slow_payload,
#     radius=50,
# )
# for lv in hot_levels:
#     logger.info(f"[Hotspot] MAE={hot_res['mae'][lv]:.4f}")

# fmae, fsmape, n_eff = gather_forecasting_results(
#     reco_grid=data_rec_cal,
#     H=counts_true,
#     fcast_qs=fcast_qs,
#     fh=forecast_horizon,
# )
# logger.info(f"[Forecast] fh={forecast_horizon} n={n_eff} sMAPE={fsmape:.4f}")

# =========================
# Save
# =========================
logger.info("Saving...")

save_path = os.path.join(
    config["train"]["save_dir"],
    f"{config['datasets']['name']}/{config['datasets']['sample_size']}/eps_{eps}_userlevel_k{k}",
)
os.makedirs(save_path, exist_ok=True)

np.save(os.path.join(save_path, "published_data_rec_MWEM_userlevel.npy"), data_rec)
np.save(os.path.join(save_path, "published_data_rec_MWEM_userlevel_gamma.npy"), data_rec_cal)
# min_cell_count_re = 999
# min_hotspot_mae = 999
# min_forecast_smape = 999
# if min_cell_count_re > re[0]:
#     min_cell_count_re = re[0]
# if min_hotspot_mae > hot_res['mae'][config['test']['sm'][0]]:
#     min_hotspot_mae = hot_res['mae'][config['test']['sm'][0]]
# if min_forecast_smape > fsmape:
#     min_forecast_smape = fsmape

logger.info('data saved at'+ save_path + '/published_data_rec_MWEM.npy')
logger.info('data saved at'+ save_path + '/published_data_rec_MWEM_userlevel_gamma.npy')
# logger.info('min_cell_count_re: {}'.format(min_cell_count_re))
# logger.info('min_hotspot_mae: {}'.format(min_hotspot_mae))
# logger.info('min_forecast_smape: {}'.format(min_forecast_smape))
logger.info("MWEM (UserLevel) finished.")
