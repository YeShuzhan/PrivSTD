# userlevel_HB_Striped_main.py
# -*- coding: utf-8 -*-

"""
HB-Striped user-level version (align to user-level)

Only make the necessary user-level changes; keep the rest of the logic unchanged:
1) Do not use read_dataset(config); instead read 5-column user-level txt: [user time lat lon loc]
2) Truncate each user to k -> Ds (db_s); the DP mechanism runs only on Ds
3) user-level delta = 1/(n_users^2)
4) Align user-level global sensitivity: Δ = k
   => Gaussian sigma *= k (here originally sigma = sqrt(2 ln(1.25/delta))/eps)
5) Evaluation: evaluate data_rec_cal (=gamma*data_rec) against counts_true (from full D)
6) Hotspot and forecasting: prepare queries using full D (db_full, counts_true); evaluate using counts_true
7) gamma = |D|/|Ds| for global calibration (post-processing, does not affect DP)
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
# 1) User-level data loading (dedicated functions)
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


# =========================================================
# 2) main
# =========================================================
config_parser = ConfigParser(name="HB-Striped_User", save_dir="./")
logger = config_parser.get_logger(config_parser.exper_name)

torch.manual_seed(2024)
np.random.seed(2024)
os.environ["CUDA_VISIBLE_DEVICES"] = config["train"]["gpu"]
logger.info(f"config: {config}")

# ---------- user-level: read full D ----------
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

logger.info(f"[HB-Striped UserLevel] n_records={n_records}, n_users={n_users}, delta={delta:.3e}, k={k}")
logger.info(f"max_lon: {max_vals[0]}, min_lon: {min_vals[0]}, max_lat: {max_vals[1]}, min_lat: {min_vals[1]}")
logger.info(f"time_interval: {(max_vals[2] - min_vals[2]) / 3600} hours")

# ---------- counts_true from full D (for evaluation/queries) ----------
counts_true, test_samples = get_counts(
    config, db_full, min_vals, max_vals,
    config["datasets"]["cell_size"], config["datasets"]["time_grid"]
)
counts_true = counts_true[:, :, :config["datasets"]["time_grid"]]

logger.info(f"counts shape: {counts_true.shape}")
logger.info(
    f"counts_max: {np.max(counts_true)}, counts_min: {np.min(counts_true)}, counts_mean: {np.mean(counts_true)}, "
    f"median: {np.median(counts_true)}, counts_sum: {np.sum(counts_true)}"
)

# ---------- truncate to Ds (mechanism only sees Ds) ----------
db, _user_ids_s = truncate_per_user(db_full, user_ids, k=k, seed=seed)
gamma = float(db_full.shape[0]) / float(db.shape[0])
logger.info(f"[Refinement] gamma = |D|/|Ds| = {db_full.shape[0]}/{db.shape[0]} = {gamma:.6f}")

# ---------- counts_s from Ds (HB-Striped runs on Ds) ----------
counts, _ = get_counts(
    config, db, min_vals, max_vals,
    config["datasets"]["cell_size"], config["datasets"]["time_grid"]
)
counts = counts[:, :, :config["datasets"]["time_grid"]]

logger.info(f"[Ds counts] shape={counts.shape}")
logger.info(
    f"[Ds counts] max: {np.max(counts)}, min: {np.min(counts)}, mean: {np.mean(counts)}, "
    f"median: {np.median(counts)}, sum: {np.sum(counts)}"
)

# add noise (user-level: sensitivity Δ = k)
eps = float(config["privacy"]["eps"])
sigma = math.sqrt(2 * math.log(1.25 / delta)) * (float(k) / eps)

noise = np.random.normal(0, sigma, counts.shape)
noisy_counts = counts + noise

logger.info(
    f"noisy_max: {np.max(noisy_counts)}, noisy_min: {np.min(noisy_counts)}, "
    f"noisy_mean: {np.mean(noisy_counts)}, noisy_median: {np.median(noisy_counts)}, "
    f"noisy_sum: {np.sum(noisy_counts)}"
)

# noisy_counts: [H, W, N]
train_data = np.transpose(noisy_counts.copy(), (2, 0, 1))  # [N, H, W]
train_data = np.expand_dims(train_data, axis=1)  # [N, C, H, W]
train_dataset = MVDataset(train_data, img_size=config["net"]["img_size"], is_train=True)
train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=False, num_workers=8)


H_, W_, T_ = counts_true.shape
max_val_vdr = float(config["datasets"].get("max_val", 10.0))
rho_xy = max_val_vdr / float(H_)
rho_t = max_val_vdr / float(T_)
data_filled_slices = compute_data_filled_slices_from_counts(counts_true)
forecast_horizon = int(config["test"].get("forecast_horizon", 3))

# Write a temporary npy from full D for forecasting/hotspot queries (avoid depending on the original *.npy)
datafile_full = os.path.join(data_dir, f"{name}_userlevel_full.npy")
np.save(datafile_full, db_full)

fcast_qs, _h_mae_ref, _h_mape_ref = get_queries_for_forecasting_vdr_exact(
    datafile=datafile_full, max_val=max_val_vdr, min_vals=min_vals, max_vals=max_vals,
    rho_xy=rho_xy,
    rho_t=rho_t,
    test_size=config["datasets"]["cell_size"],
    H=counts_true, data_filled_slices=data_filled_slices, fh=3,
)
logger.info(f'Forecast queries completed')

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
logger.info(f'Hotspot queries completed')


time1 = time.time()

# Set branching factor per dimension (using 2 for 3D as per analysis for higher dimensions)
b = 2
d = 3
c = b ** d  # number of children per node

# Pad to next power of 2 in each dimension
def next_power_of_2(x):
    if x == 0:
        return 1
    return 2 ** math.ceil(math.log2(x))

H_p = next_power_of_2(H_)
W_p = next_power_of_2(W_)
T_p = next_power_of_2(T_)

padded_counts = np.pad(counts, ((0, H_p - H_), (0, W_p - W_), (0, T_p - T_)), mode="constant").astype(float)
padded_noisy_counts = np.pad(noisy_counts, ((0, H_p - H_), (0, W_p - W_), (0, T_p - T_)), mode="constant").astype(float)

# Build levels bottom-up for true sums (for structure, but not used for DP)
levels = []
current = padded_counts.copy()
levels.append(current)

while max(current.shape) > 1:
    new_shape = tuple(math.ceil(s / b) for s in current.shape)
    new = np.zeros(new_shape)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            for k_ in range(new_shape[2]):
                sub = current[i*b:(i+1)*b, j*b:(j+1)*b, k_*b:(k_+1)*b]
                new[i, j, k_] = sub.sum()
    levels.append(new)
    current = new

num_levels = len(levels)  # h+1 levels, index 0: leaf (level 1), index h: root (level h+1)
h = num_levels - 1  # number of noised levels

# Compute sigma_level = sigma * sqrt(h) to match variance scaling
sigma_level = sigma * math.sqrt(h)

# Create noisy levels
noisy_levels = [levels[l].copy() for l in range(num_levels)]
for l in range(h):  # noise levels 0 to h-1
    noise_l = np.random.normal(0, sigma_level, noisy_levels[l].shape)
    noisy_levels[l] += noise_l

# Weighted averaging (bottom-up)
z_levels = [noisy_levels[0].copy()]
sum_child_list = []
for l in range(1, num_levels):
    new_shape = levels[l].shape
    sum_child = np.zeros(new_shape)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            for k_ in range(new_shape[2]):
                sub = z_levels[l-1][i*b:(i+1)*b, j*b:(j+1)*b, k_*b:(k_+1)*b]
                sum_child[i, j, k_] = sub.sum()
    sum_child_list.append(sum_child)

    i_level = l  # level index starting from 1 for leaf
    alpha = (c ** (i_level + 1) - c ** i_level) / (c ** (i_level + 1) - 1)
    beta = (c ** i_level - 1) / (c ** (i_level + 1) - 1)
    z = alpha * noisy_levels[l] + beta * sum_child
    z_levels.append(z)

# Mean consistency (top-down)
n_bar_levels = [z_levels[h].copy()]
for l in range(h - 1, -1, -1):
    sum_child = sum_child_list[l]
    new_shape = levels[l].shape
    adjustment = np.zeros(new_shape)
    parent_shape = levels[l + 1].shape
    for i in range(parent_shape[0]):
        for j in range(parent_shape[1]):
            for k_ in range(parent_shape[2]):
                diff = n_bar_levels[-1][i, j, k_] - sum_child[i, j, k_]
                diff_per_child = diff / c
                adjustment[i*b:(i+1)*b, j*b:(j+1)*b, k_*b:(k_+1)*b] += diff_per_child
    n_bar = z_levels[l] + adjustment
    n_bar_levels.append(n_bar)

# The published histogram is the leaf level after inference, cropped to original size
data_rec = n_bar_levels[-1][:H_, :W_, :T_]

time2 = time.time()
print('Running time: ', time2 - time1)


data_rec_cal = np.maximum(gamma * data_rec, 0.0)
noisy_cal = np.maximum(gamma * noisy_counts, 0.0)


# mae, re = get_eval_results(counts_true, data_rec_cal, test_samples, sm=config["test"]["sm"])
# logger.info(f"MAE: {mae}, RE: {re}")

# hot_res = gather_hotspot_results(
#     reco_grid=data_rec_cal, H=counts_true, hot_levels=hot_levels,
#     H_slow_qs=H_slow_qs, Hress_slow_payload=Hress_slow_payload, radius=50
# )
# for lv in hot_levels:
#     logger.info(f"[Hotspot] MAE={hot_res['mae'][lv]:.4f}")

# # Forecast evaluation (FMAE / sMAPE)
# fmae, fsmape, n_eff = gather_forecasting_results(
#     reco_grid=data_rec_cal, H=counts_true, fcast_qs=fcast_qs, fh=forecast_horizon
# )
# logger.info(f"[Forecast] fh={forecast_horizon} n={n_eff} sMAPE={fsmape:.4f}")

# save published npy
save_path = config["train"]["save_dir"] + "/{}/{}/eps_{}_userlevel_k{}".format(
    config["datasets"]["name"], config["datasets"]["cell_size"], eps, k
)
os.makedirs(save_path, exist_ok=True)
np.save(save_path + "/published_data_rec_HB-Striped_userlevel.npy", data_rec)
np.save(save_path + "/published_data_rec_HB-Striped_userlevel_gamma.npy", data_rec_cal)

# min_cell_count_re = 999
# min_hotspot_mae = 999
# min_forecast_smape = 999
# if min_cell_count_re > re[0]:
#     min_cell_count_re = re[0]
# if min_hotspot_mae > hot_res['mae'][config['test']['sm'][0]]:
#     min_hotspot_mae = hot_res['mae'][config['test']['sm'][0]]
# if min_forecast_smape > fsmape:
#     min_forecast_smape = fsmape


logger.info('data saved at'+ save_path + '/published_data_rec_HB-Striped_userlevel.npy')
logger.info('Refine data saved at'+ save_path + '/published_data_rec_Striped_userlevel_gamma.npy')
# logger.info('min_cell_count_re: {}'.format(min_cell_count_re))
# logger.info('min_hotspot_mae: {}'.format(min_hotspot_mae))
# logger.info('min_forecast_smape: {}'.format(min_forecast_smape))
logger.info("HB-Striped (UserLevel) finished.")
