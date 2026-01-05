# userlevel_AHP_main.py
# -*- coding: utf-8 -*-

"""
AHP user-level version (align to  user-level)

Only make the necessary user-level changes; keep the rest of the logic as unchanged as possible:
1) No longer use read_dataset(config); instead read the 5-column user-level raw txt
2) Truncate each user to k records to obtain Ds (db_s); the DP mechanism runs only on Ds
3) user-level delta = 1/(n_users^2)
4) User-level global sensitivity aligned: Δ = k
   => All Gaussian noise sigmas must be multiplied by k (sigma1 / sigma2 / baseline)
5) Evaluation aligned: use counts_true (from full D) for evaluation; apply global calibration gamma=|D|/|Ds| to the released result
6) Hotspot and forecasting: use full D (db_full, counts_true) to prepare queries; evaluation uses counts_true
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


# === Noise calibration for approximate DP (ε,δ) Gaussian mechanism ===
def gaussian_sigma(epsilon: float, delta: float, sensitivity: float = 1.0) -> float:
    """
    Standard Gaussian mechanism calibration for approximate DP (ε, δ):
        sigma = S * sqrt(2 ln(1.25/delta)) / epsilon
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


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


delta_total = 1.0 / (n_users**2)

logger.info(f"[AHP UserLevel] n_records={n_records}, n_users={n_users}, delta_total={delta_total:.3e}, k={k}")
logger.info(f"max_lon: {max_vals[0]}, min_lon: {min_vals[0]}, max_lat: {max_vals[1]}, min_lat: {min_vals[1]}")
logger.info(f"time_interval: {(max_vals[2] - min_vals[2]) / 3600} hours")

# ========== counts_true from D (for evaluation) ==========
counts_true, test_samples = get_counts(
    config, db_full, min_vals, max_vals,
    config["datasets"]["sample_size"], config["datasets"]["time_grid"]
)
counts_true = counts_true[:, :, :config["datasets"]["time_grid"]]

logger.info(f"counts shape: {counts_true.shape}")
logger.info(f"counts_max: {np.max(counts_true)}, counts_min: {np.min(counts_true)}, counts_mean: {np.mean(counts_true)}, "
            f"median: {np.median(counts_true)}, counts_sum: {np.sum(counts_true)}")

# ========== truncate to Ds (DP only sees Ds) ==========
db, _user_ids_s = truncate_per_user(db_full, user_ids, k=k, seed=seed)
gamma = float(db_full.shape[0]) / float(db.shape[0])
logger.info(f"[ Refinement] gamma = |D|/|Ds| = {db_full.shape[0]}/{db.shape[0]} = {gamma:.6f}")

# ========== counts_s from Ds (mechanism runs on Ds) ==========
counts_s, _ = get_counts(
    config, db, min_vals, max_vals,
    config["datasets"]["sample_size"], config["datasets"]["time_grid"]
)
counts_s = counts_s[:, :, :config["datasets"]["time_grid"]]

# =========================
#     AHP - Gaussian version (User-level)
# =========================
time1 = time.time()
eps_total = float(config["privacy"]["eps"])

# Budget split: (ε,δ) = (ε1,δ1) + (ε2,δ2)
eps1 = eps_total / 2.0
eps2 = eps_total / 2.0
delta1 = delta_total / 2.0
delta2 = delta_total / 2.0

# User-level sensitivity aligned with the number of users: Δ = k  -> sigma *= k
sigma1 = gaussian_sigma(eps1, delta1, sensitivity=float(k))  # Stage 1: private view
sigma2 = gaussian_sigma(eps2, delta2, sensitivity=float(k))  # Stage 2: noisy cluster sums

logger.info(f"[Gaussian DP UserLevel] eps_total={eps_total}, delta_total={delta_total} | "
            f"eps1={eps1}, delta1={delta1}, sigma1={sigma1:.6f} | "
            f"eps2={eps2}, delta2={delta2}, sigma2={sigma2:.6f}")

# ======  and evaluation prep (use counts_true / db_full) ======
H_, W_, T_ = counts_true.shape
max_val_vdr = float(config["datasets"].get("max_val", 10.0))
rho_xy = max_val_vdr / float(H_)
rho_t = max_val_vdr / float(T_)
data_filled_slices = compute_data_filled_slices_from_counts(counts_true)
forecast_horizon = int(config["test"].get("forecast_horizon", 3))

# Write a temporary npy from full D to generate queries
datafile_full = os.path.join(data_dir, f"{name}_userlevel_full.npy")
np.save(datafile_full, db_full)

fcast_qs, _h_mae_ref, _h_mape_ref = get_queries_for_forecasting_vdr_exact(
    datafile=datafile_full, max_val=max_val_vdr, min_vals=min_vals, max_vals=max_vals,
    rho_xy=rho_xy, rho_t=rho_t, test_size=config["datasets"]["sample_size"],
    H=counts_true, data_filled_slices=data_filled_slices, fh=3,
)
logger.info(f"[ ForecastQs] prepared={len(fcast_qs)} "
            f"(ref H_MAE={_h_mae_ref:.4f}, H_sMAPE={_h_mape_ref:.4f})")

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
    logger.info(f"[ HotspotQs] level={_hot} prepared={len(_qs)}")


# ====== AHP Stage 1: generate "private view" then threshold and sort (on Ds counts_s) ======
H_hat = counts_s + np.random.normal(0.0, sigma1, size=counts_s.shape)

H_hat_flat = H_hat.flatten().copy()
m = H_hat_flat.size

eta = 1.0  # Keep the tuning term
theta = eta * math.sqrt(2.0 * math.log(m)) * sigma1
H_hat_flat[H_hat_flat < theta] = 0.0

sorted_indices = np.argsort(H_hat_flat)

# ====== AHP greedy clustering (performed on private data) ======
clusters = []
i = 0
while i < m:
    cluster_start = i
    v = H_hat_flat[sorted_indices[i]]
    curr_size = 1
    curr_sum = float(v)
    curr_sq = float(v) ** 2
    curr_ae = 0.0
    curr_err = curr_ae + (sigma2 ** 2) / curr_size
    i += 1

    while i < m:
        v_next = float(H_hat_flat[sorted_indices[i]])
        new_size = curr_size + 1
        new_sum = curr_sum + v_next
        new_sq = curr_sq + (v_next ** 2)
        new_ae = new_sq - (new_sum ** 2) / new_size
        new_err = new_ae + (sigma2 ** 2) / new_size

        no_merge_err_upper = curr_err + (sigma2 ** 2)

        if new_err < no_merge_err_upper:
            curr_size = new_size
            curr_sum = new_sum
            curr_sq = new_sq
            curr_ae = new_ae
            curr_err = new_err
            i += 1
        else:
            break

    clusters.append((cluster_start, i))

# ====== AHP Stage 2: add Gaussian noise to cluster sums and then average within clusters (on Ds) ======
counts_s_flat = counts_s.flatten().copy()
data_rec_flat = np.zeros_like(counts_s_flat)

for start, end in clusters:
    if start == end:
        continue
    cl_indices = sorted_indices[start:end]
    cl_size = end - start

    true_sum = float(np.sum(counts_s_flat[cl_indices]))
    noisy_sum = true_sum + np.random.normal(0.0, sigma2)
    data_rec_flat[cl_indices] = noisy_sum / cl_size

data_rec = data_rec_flat.reshape(counts_s.shape)

# ======  global calibration gamma (both release and evaluation use gamma*) ======
data_rec_cal = np.maximum(gamma * data_rec, 0.0)

time2 = time.time()
traintime = time2 - time1
print("Running time:", traintime)
# ====== Evaluation: compare against counts_true ======
# mae, re = get_eval_results(counts_true, data_rec_cal, test_samples, sm=config["test"]["sm"])
# logger.info(f"[AHP UserLevel] MAE(gamma): {mae}, RE(gamma): {re}")


# hot_res = gather_hotspot_results(
#     reco_grid=data_rec_cal, H=counts_true, hot_levels=hot_levels,
#     H_slow_qs=H_slow_qs, Hress_slow_payload=Hress_slow_payload, radius=50
# )
# for lv in hot_levels:
#     logger.info(f"[Hotspot] MAE={hot_res['mae'][lv]:.4f}")

# fmae, fsmape, n_eff = gather_forecasting_results(
#     reco_grid=data_rec_cal, H=counts_true, fcast_qs=fcast_qs, fh=forecast_horizon
# )
# logger.info(f"[Forecast] fh={forecast_horizon} n={n_eff} sMAPE={fsmape:.4f}")

logger.info("Saving released npy...")

save_path = config["train"]["save_dir"] + "/{}/{}/eps_{}_userlevel_k{}".format(
    config["datasets"]["name"], config["datasets"]["sample_size"], eps_total, k
)
os.makedirs(save_path, exist_ok=True)
# min_cell_count_re = 999
# min_hotspot_mae = 999
# min_forecast_smape = 999
# if min_cell_count_re > re[0]:
#     min_cell_count_re = re[0]
# if min_hotspot_mae > hot_res['mae'][config['test']['sm'][0]]:
#     min_hotspot_mae = hot_res['mae'][config['test']['sm'][0]]
# if min_forecast_smape > fsmape:
#     min_forecast_smape = fsmape

np.save(save_path + "/published_data_rec_AHP_userlevel.npy", data_rec)
np.save(save_path + "/published_data_rec_AHP_userlevel_gamma.npy", data_rec_cal)

logger.info("Saved:")
logger.info(save_path + "/published_data_rec_AHP_userlevel.npy")
logger.info(save_path + "/published_data_rec_AHP_userlevel_gamma.npy")

# logger.info('min_cell_count_re: {}'.format(min_cell_count_re))
# logger.info('min_hotspot_mae: {}'.format(min_hotspot_mae))
# logger.info('min_forecast_smape: {}'.format(min_forecast_smape))
logger.info("AHP (user-level, overlap-conserving, gamma-calibrated) finished.")