# userlevel_AG_main.py
# -*- coding: utf-8 -*-

"""
AG (AdaptiveGrid) user-level version (align to  user-level)

Only make the necessary user-level changes; keep the rest of the logic as unchanged as possible:
1) No longer use read_dataset(config); instead read the 5-column user-level raw txt
2) Truncate each user to k records to obtain Ds (db_s); the DP mechanism runs only on Ds
3) user-level delta = 1/(n_users^2)
4) User-level global sensitivity aligned: Δ = k
   => All Gaussian noise sigmas must be multiplied by k (coarse / fine / total baseline)
5) Evaluation aligned: use counts_true (from full D) for evaluation; apply global calibration gamma=|D|/|Ds| to the released result
6) Hotspot and forecasting use counts_true and db_full to construct queries (consistent with userlevel_PrivSDT/UG)
"""

import os
import torch
import torch.nn as nn
import numpy as np
from parse import config
import math
from logger.logger import ConfigParser
from utils.dataset import get_counts, MVDataset, pad_data
from torch.utils.data import DataLoader
from model.MultiView import MultiView
from model.DnCNN import DnCNN
from utils.eval import *
from tqdm import tqdm
from utils.results_stats import ResultStats
import time


# =========================================================
# 1) User-level data loading (dedicated function, do not reuse read_dataset(config))
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
    Return:
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


# ---------------------------
# DP helpers
# ---------------------------
def gaussian_sigma(eps: float, delta: float, sensitivity: float = 1.0) -> float:
    """(ε, δ)-DP Gaussian mechanism standard deviation; default L2 sensitivity = 1."""
    if eps <= 0:
        raise ValueError("epsilon must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return math.sqrt(2.0 * math.log(1.25 / delta)) * (sensitivity / eps)


def dp_to_required_zcdp(epsilon: float, delta: float) -> float:
    """ Kept only (useful if you need a zCDP derivation); the main pipeline uses (ε,δ)-DP Gaussian noise. """
    if epsilon < 0:
        raise ValueError("epsilon must be non-negative")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    a = math.sqrt(math.log(1.0 / delta))
    x = math.sqrt(epsilon + a * a) - a
    x = max(0.0, x)
    rho = x * x
    return rho


# ---------------------------
# vectorized geometry helpers
# ---------------------------
def overlapping_lengths_1d(lefts1, rights1, lefts2, rights2):
    """
    Compute overlap lengths between groups of 1D intervals.
    lefts1, rights1: original-grid boundaries of shape (I,1)
    lefts2, rights2: sub-grid boundaries of shape (1,M)
    Return an (I, M) overlap-length matrix.
    """
    return np.clip(np.minimum(rights1, rights2) - np.maximum(lefts1, lefts2), 0.0, None)


def build_block_weights(lon_bins, lat_bins, sub_lon_bins, sub_lat_bins, p_range, q_range):
    """
    For a coarse cell (p,q), construct 1D overlap weight matrices A_w (I_p x m2) and B_w (J_p x m2).
    Weights equal (overlap length / sub-cell length), used for an area-normalized separable approximation:
      A_w @ U @ B_w^T  == sum_{sub_i,sub_j} (overlap_lon/Δx_sub)*(overlap_lat/Δy_sub)*U
    """
    p, p1 = p_range
    q, q1 = q_range
    I_idx = np.arange(p, p1)   # longitude-direction cell index range
    J_idx = np.arange(q, q1)   # latitude-direction cell index range

    L_left = lon_bins[I_idx][:, None]
    L_right = lon_bins[I_idx + 1][:, None]
    T_left = lat_bins[J_idx][:, None]
    T_right = lat_bins[J_idx + 1][:, None]

    S_left = sub_lon_bins[None, :-1]
    S_right = sub_lon_bins[None, 1:]
    U_left = sub_lat_bins[None, :-1]
    U_right = sub_lat_bins[None, 1:]

    A = overlapping_lengths_1d(L_left, L_right, S_left, S_right)  # (I_p, m2)
    B = overlapping_lengths_1d(T_left, T_right, U_left, U_right)  # (J_p, m2)

    sub_lon_len = (S_right - S_left)
    sub_lat_len = (U_right - U_left)
    sub_lon_len = np.clip(sub_lon_len, 1e-12, None)
    sub_lat_len = np.clip(sub_lat_len, 1e-12, None)

    A_w = A / sub_lon_len
    B_w = B / sub_lat_len
    return I_idx, J_idx, A_w.astype(np.float32), B_w.astype(np.float32)


def find_covering_cell_indices(bin_edges, left, right):
    """
    Given interval [left, right], find the covered original-grid cell index range [i0, i1).
    """
    i0 = np.searchsorted(bin_edges, left, side="right") - 1
    i0 = max(i0, 0)
    i1 = np.searchsorted(bin_edges, right, side="left")
    i1 = min(i1, len(bin_edges) - 1)
    if i1 <= i0:
        i1 = i0
    return i0, i1


# ---------------------------
# main
# ---------------------------
config_parser = ConfigParser(name="MultiView", save_dir="./")
logger = config_parser.get_logger(config_parser.exper_name)
torch.manual_seed(2024)
np.random.seed(2024)
os.environ["CUDA_VISIBLE_DEVICES"] = config["train"]["gpu"]
logger.info(f"config: {config}")

# ========== user-level: read full D ==========
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

# ========== user-level delta ==========
eps = float(config["privacy"]["eps"])
delta_total = 1.0 / (n_users ** 2)

logger.info(
    f"[AG UserLevel] n_records={n_records}, n_users={n_users}, eps={eps}, delta_total={delta_total:.3e}, k={k}"
)
logger.info(
    f"max_lon: {max_vals[0]}, min_lon: {min_vals[0]}, max_lat: {max_vals[1]}, min_lat: {min_vals[1]}"
)
logger.info(f"time_interval: {(max_vals[2] - min_vals[2]) / 3600} hours")

# ========== counts_true from D (for evaluation) ==========
counts_true, test_samples = get_counts(
    config, db_full, min_vals, max_vals, config["datasets"]["sample_size"], config["datasets"]["time_grid"]
)
counts_true = counts_true[:, :, :config["datasets"]["time_grid"]]
H_, W_, T_ = counts_true.shape
logger.info(f"[TrueCounts] shape: {counts_true.shape}")
logger.info(
    f"counts_max: {np.max(counts_true)}, counts_min: {np.min(counts_true)}, counts_mean: {np.mean(counts_true)}, "
    f"median: {np.median(counts_true)}, counts_sum: {np.sum(counts_true)}"
)

# ========== truncate to Ds (DP only sees Ds) ==========
db, _user_ids_s = truncate_per_user(db_full, user_ids, k=k, seed=seed)
gamma = float(db_full.shape[0]) / float(db.shape[0])
logger.info(f"[ Refinement] gamma = |D|/|Ds| = {db_full.shape[0]}/{db.shape[0]} = {gamma:.6f}")

# ---------------------------
# AdapativeGrid with Gaussian noise
# ---------------------------
time1 = time.time()
alpha = 0.5
c = 10.0
c2 = c / 2.0

# Compose the budget as (αε, δ/2) + ((1-α)ε, δ/2) -> (ε, δ)
eps_coarse = alpha * eps
eps_fine = (1.0 - alpha) * eps
delta_coarse = delta_total / 2
delta_fine = delta_total / 2

# User-level sensitivity aligned: Δ = k (all layers must multiply by k)
sigma_coarse = gaussian_sigma(eps_coarse, delta_coarse, sensitivity=float(k))
sigma_fine = gaussian_sigma(eps_fine, delta_fine, sensitivity=float(k))

# Uniform bins (aligned with counts_true shape)
lon_bins = np.linspace(min_vals[0], max_vals[0], H_ + 1)
lat_bins = np.linspace(min_vals[1], max_vals[1], W_ + 1)
time_bins = np.linspace(min_vals[2], max_vals[2], T_ + 1)

# Empirical m1 formula (keep your implementation style, but using user-level n_users is more reasonable)
m_ug = int(np.ceil(np.sqrt(n_users * eps / c)))
m1 = max(10, int(np.ceil(m_ug / 4.0)))
logger.info(f"AdaptiveGrid m1 (coarse grid): {m1}")

# Coarse-grid boundaries
lon_bins_coarse = np.linspace(min_vals[0], max_vals[0], m1 + 1)
lat_bins_coarse = np.linspace(min_vals[1], max_vals[1], m1 + 1)

# 1) Get coarse counts once (no noise) — on Ds
counts_coarse = np.histogramdd(db, bins=(lon_bins_coarse, lat_bins_coarse, time_bins))[0].astype(
    np.float32
)  # (m1, m1, T_)

# 2) Add Gaussian noise to coarse counts
noisy_coarse = counts_coarse + np.random.normal(
    loc=0.0, scale=sigma_coarse, size=counts_coarse.shape
).astype(np.float32)

# 3) Pre-assign each point to a coarse cell (p,q), for fine partitioning
p_idx = np.clip(np.searchsorted(lon_bins_coarse, db[:, 0], side="right") - 1, 0, m1 - 1)
q_idx = np.clip(np.searchsorted(lat_bins_coarse, db[:, 1], side="right") - 1, 0, m1 - 1)
pq_flat = p_idx * m1 + q_idx
order = np.argsort(pq_flat)
db_sorted = db[order]
pq_sorted = pq_flat[order]
starts = np.searchsorted(pq_sorted, np.arange(m1 * m1), side="left")
ends = np.searchsorted(pq_sorted, np.arange(m1 * m1), side="right")

adaptive_grids = []
# 4) For each coarse cell: adaptive refinement + fine counts + Gaussian noise + constrained inference (vectorized over T)
for p in tqdm(range(m1), desc="Fine partition per coarse cell (Gaussian DP)"):
    for q in range(m1):
        N_prime = float(np.sum(noisy_coarse[p, q, :]))
        m2 = max(1, int(np.ceil(np.sqrt(max(0.0, N_prime * (1.0 - alpha) * eps / c2)))))
        sub_lon_bins = np.linspace(lon_bins_coarse[p], lon_bins_coarse[p + 1], m2 + 1)
        sub_lat_bins = np.linspace(lat_bins_coarse[q], lat_bins_coarse[q + 1], m2 + 1)

        idx_flat = p * m1 + q
        s, e = starts[idx_flat], ends[idx_flat]
        db_sub = db_sorted[s:e]

        if e > s:
            counts_fine = np.histogramdd(db_sub, bins=(sub_lon_bins, sub_lat_bins, time_bins))[0].astype(
                np.float32
            )
        else:
            counts_fine = np.zeros((m2, m2, T_), dtype=np.float32)

        noisy_fine = counts_fine + np.random.normal(loc=0.0, scale=sigma_fine, size=counts_fine.shape).astype(
            np.float32
        )

        denom = (1.0 - alpha) ** 2 + (alpha**2) * (m2**2)
        coeff_v = (alpha**2) * (m2**2) / denom
        coeff_sum = (1.0 - alpha) ** 2 / denom
        sum_u = noisy_fine.sum(axis=(0, 1), keepdims=True)  # (1,1,T_)
        v = noisy_coarse[p, q, :][None, None, :]  # (1,1,T_)
        v_prime = coeff_v * v + coeff_sum * sum_u
        noisy_fine += (v_prime - sum_u) / (m2**2)

        adaptive_grids.append(
            {
                "p": p,
                "q": q,
                "m2": m2,
                "sub_lon_bins": sub_lon_bins.astype(np.float64),
                "sub_lat_bins": sub_lat_bins.astype(np.float64),
                "noisy_fine": noisy_fine.astype(np.float32),
            }
        )

# ---------------------------
# Reconstruct data_rec (vectorized blocks: A_w @ U @ B_w^T)
# ---------------------------
data_rec = np.zeros_like(counts_true, dtype=np.float32)

for grid in tqdm(adaptive_grids, desc="Reconstructing full grid from fine leaves"):
    p = grid["p"]
    q = grid["q"]
    m2 = grid["m2"]
    sub_lon_bins = grid["sub_lon_bins"]
    sub_lat_bins = grid["sub_lat_bins"]
    noisy_fine = grid["noisy_fine"]  # (m2, m2, T_)

    coarse_lon_left, coarse_lon_right = lon_bins_coarse[p], lon_bins_coarse[p + 1]
    coarse_lat_left, coarse_lat_right = lat_bins_coarse[q], lat_bins_coarse[q + 1]

    i0, i1 = find_covering_cell_indices(lon_bins, coarse_lon_left, coarse_lon_right)
    j0, j1 = find_covering_cell_indices(lat_bins, coarse_lat_left, coarse_lat_right)
    if i1 <= i0 or j1 <= j0:
        continue

    I_idx, J_idx, A_w, B_w = build_block_weights(
        lon_bins, lat_bins, sub_lon_bins, sub_lat_bins, (i0, i1), (j0, j1)
    )

    block = np.einsum("im,mnt,jn->ijt", A_w, noisy_fine, B_w, optimize=True)
    data_rec[I_idx[:, None], J_idx[None, :], :] += block.astype(np.float32)

time2 = time.time()
traintime = time2 - time1

# ==========  global calibration gamma ==========
data_rec_cal = np.maximum(gamma * data_rec, 0.0)

# ---------------------------
# Statistics and evaluation
# ---------------------------
noisy_values = np.concatenate([g["noisy_fine"].ravel() for g in adaptive_grids])
logger.info(
    f"noisy_max: {np.max(noisy_values)}, noisy_min: {np.min(noisy_values)}, "
    f"noisy_mean: {np.mean(noisy_values)}, noisy_median: {np.median(noisy_values)}, "
    f"noisy_sum: {np.sum(noisy_values)}"
)

# As the ID baseline: add noise with the same total budget to Ds fine-grid counts_s directly (sensitivity=k),
# then multiply by gamma, and compare with counts_true
counts_s_fine, _ = get_counts(
    config, db, min_vals, max_vals, config["datasets"]["sample_size"], config["datasets"]["time_grid"]
)
counts_s_fine = counts_s_fine[:, :, :config["datasets"]["time_grid"]]
sigma_total = gaussian_sigma(eps, delta_total, sensitivity=float(k))
noisy_counts_fine = counts_s_fine + np.random.normal(0.0, sigma_total, counts_s_fine.shape).astype(np.float32)
noisy_counts_fine_cal = np.maximum(gamma * noisy_counts_fine, 0.0)

id_mae, id_re = get_eval_results(counts_true, noisy_counts_fine_cal, test_samples, sm=config["test"]["sm"])

# ====== one-time preparation (user-level: generate queries using full D; evaluate using counts_true) ======
max_val_vdr = float(config["datasets"].get("max_val", 10.0))
rho_xy = max_val_vdr / float(H_)
rho_t = max_val_vdr / float(T_)
data_filled_slices = compute_data_filled_slices_from_counts(counts_true)
forecast_horizon = int(config["test"].get("forecast_horizon", 3))

# Save a temporary file from full D for query generation
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
    fh=3,
)
logger.info(f'Forecast queries completed')
_raw = np.load(datafile_full)
_raw = ((_raw - min_vals) / (max_vals - min_vals) - 0.5) * max_val_vdr
_loc_ijk = convert_db_to_ijk_aniso(_raw, rho_xy=rho_xy, rho_t=rho_t, max_val=max_val_vdr)
loc_ijk_arr = np.asarray(list(zip(*_loc_ijk)), dtype=int)

hot_levels = config["test"].get("hotspot_levels", [20])
Hc_slow_payload, Hress_slow_payload, H_slow_qs = {}, {}, {}
for _hot in hot_levels:
    _qs, _Hc, _Hress = get_hotspot_queries(_hot, counts_true, loc_ijk_arr, limit=500, radius=50)
    Hc_slow_payload[_hot] = np.asarray(_Hc, dtype=int)
    Hress_slow_payload[_hot] = np.asarray(_Hress, dtype=float).reshape(-1, 1)
    H_slow_qs[_hot] = np.asarray(_qs, dtype=int)
logger.info(f'Hotspot queries completed')
# Evaluation
# mae, re = get_eval_results(counts_true, data_rec_cal, test_samples, sm=config["test"]["sm"])
# logger.info(f"[AG UserLevel] MAE(gamma): {mae}, RE(gamma): {re}")

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
#     reco_grid=data_rec_cal, H=counts_true, fcast_qs=fcast_qs, fh=forecast_horizon
# )
# logger.info(f"[Forecast] fh={forecast_horizon} n={n_eff} sMAPE={fsmape:.4f}")

# Save
save_path = config["train"]["save_dir"] + "/{}/{}/eps_{}_userlevel_k{}".format(
    config["datasets"]["name"], config["datasets"]["sample_size"], eps, k
)
os.makedirs(save_path, exist_ok=True)
np.save(save_path + "/published_data_rec_AG_userlevel.npy", data_rec)
np.save(save_path + "/published_data_rec_AG_userlevel_gamma.npy", data_rec_cal)

# min_cell_count_re = 999
# min_hotspot_mae = 999
# min_forecast_smape = 999
# if min_cell_count_re > re[0]:
#     min_cell_count_re = re[0]
# if min_hotspot_mae > hot_res['mae'][config['test']['sm'][0]]:
#     min_hotspot_mae = hot_res['mae'][config['test']['sm'][0]]
# if min_forecast_smape > fsmape:
#     min_forecast_smape = fsmape
logger.info("Saved:")
logger.info(save_path + "/published_data_rec_AG_userlevel.npy")
logger.info(save_path + "/published_data_rec_AG_userlevel_gamma.npy")

# logger.info('min_cell_count_re: {}'.format(min_cell_count_re))
# logger.info('min_hotspot_mae: {}'.format(min_hotspot_mae))
# logger.info('min_forecast_smape: {}'.format(min_forecast_smape))
logger.info("AG (user-level, overlap-conserving, gamma-calibrated) finished.")
