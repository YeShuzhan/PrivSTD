# userlevel_PrivBayes_main.py
# -*- coding: utf-8 -*-

"""
PrivBayes user-level version (align to  user-level)

Only apply user-level necessary changes; keep all other logic unchanged:
1) Do not use read_dataset(config). Instead, read 5-column user-level txt:
   [user time lat lon loc]
2) Truncate each user to k records -> Ds (db_s); PrivBayes structure learning +
   conditional distribution noise run ONLY on Ds
3) user-level delta = 1/(n_users^2)
4) user-level global sensitivity: for all "count/frequency/joint count/cluster sum"
   queries, Δ = k
   -> all Gaussian noise sigmas are multiplied by k
   - baseline noisy_counts: sigma *= k
   - PrivBayes parameter distribution noisy_counts_m: sigma *= k
5) Evaluation aligns to : counts_true (from full D) vs data_rec_cal (=gamma*data_rec)
6) Hotspot & forecasting: prepare queries using full D (counts_true / db_full);
   evaluate using counts_true
7) gamma = |D|/|Ds| for global calibration (post-processing; does not affect DP)
"""

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')

import os
import torch
import numpy as np
from parse import config
import math
from logger.logger import ConfigParser
from utils.dataset import get_counts
from utils.eval import *
from tqdm import tqdm
from utils.results_stats import ResultStats
import time
from collections import Counter


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
    uid_s = uid_sorted[kept_idx_sorted]
    return db_s, uid_s


# =========================================================
# main
# =========================================================
config_parser = ConfigParser(name='MultiView', save_dir='./')
logger = config_parser.get_logger(config_parser.exper_name)
torch.manual_seed(2024)
np.random.seed(2024)
os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpu']
logger.info(f'config: {config}')

# ---- user-level read full D ----
data_dir = "./data/"
name = config["datasets"]["name"]
raw_filename = name + ".txt"

k = 5
seed = 2024

db_full, user_ids, min_vals, max_vals, n_records, n_users = read_userlevel_txt_dataset(
    name=name, data_dir=data_dir, filename=raw_filename, delimiter=None, skip_bad_lines=True
)

# user-level delta
delta = 1.0 / (n_users ** 2)

logger.info(f'[PrivBayes UserLevel] n_records={n_records}, n_users={n_users}, delta={delta:.3e}, k={k}')
logger.info(f'max_lon: {max_vals[0]}, min_lon: {min_vals[0]}, max_lat: {max_vals[1]}, min_lat: {min_vals[1]}')
logger.info(f'time_interval: {(max_vals[2] - min_vals[2]) / 3600} hours')

# counts_true from full D (for eval/query)
counts_true, test_samples = get_counts(
    config, db_full, min_vals, max_vals, config['datasets']['sample_size'], config['datasets']['time_grid']
)
counts_true = counts_true[:, :, :config['datasets']['time_grid']]
logger.info(f'counts_true shape: {counts_true.shape}')
logger.info(f'counts_true_max: {np.max(counts_true)}, counts_true_min: {np.min(counts_true)}, '
            f'counts_true_mean: {np.mean(counts_true)}, median: {np.median(counts_true)}, '
            f'counts_true_sum: {np.sum(counts_true)}')

# truncate -> Ds
db, _ = truncate_per_user(db_full, user_ids, k=k, seed=seed)
gamma = float(db_full.shape[0]) / float(db.shape[0])
logger.info(f'[Refinement] gamma = |D|/|Ds| = {db_full.shape[0]}/{db.shape[0]} = {gamma:.6f}')

# counts_s from Ds (mechanism input)
counts, _ = get_counts(
    config, db, min_vals, max_vals, config['datasets']['sample_size'], config['datasets']['time_grid']
)
counts = counts[:, :, :config['datasets']['time_grid']]
logger.info(f'counts(Ds) shape: {counts.shape}')
logger.info(f'counts_max: {np.max(counts)}, counts_min: {np.min(counts)}, counts_mean: {np.mean(counts)}, '
            f'median: {np.median(counts)}, counts_sum: {np.sum(counts)}')

# ---------------------------
# baseline add noise on Ds counts (user-level Δ=k)
# ---------------------------
eps = float(config['privacy']['eps'])
sigma = math.sqrt(2 * math.log(1.25 / delta)) * (float(k) / eps)  # Δ=k
noise = np.random.normal(0, sigma, counts.shape)
noisy_counts = counts + noise
logger.info(f'noisy_max: {np.max(noisy_counts)}, noisy_min: {np.min(noisy_counts)}, '
            f'noisy_mean: {np.mean(noisy_counts)}, noisy_median: {np.median(noisy_counts)}, '
            f'noisy_sum: {np.sum(noisy_counts)}')

# ====== one-time preparation: use full D's counts_true / db_full for forecasting/hotspot (align ) ======
H_, W_, T_ = counts_true.shape
max_val_vdr = float(config['datasets'].get('max_val', 10.0))
rho_xy = max_val_vdr / float(H_)
rho_t = max_val_vdr / float(T_)
data_filled_slices = compute_data_filled_slices_from_counts(counts_true)
forecast_horizon = int(config['test'].get('forecast_horizon', 3))

datafile_full = os.path.join(data_dir, f"{name}_userlevel_full.npy")
np.save(datafile_full, db_full)

fcast_qs, _h_mae_ref, _h_mape_ref = get_queries_for_forecasting_vdr_exact(
    datafile=datafile_full,
    max_val=max_val_vdr,
    min_vals=min_vals,
    max_vals=max_vals,
    rho_xy=rho_xy,
    rho_t=rho_t,
    test_size=config['datasets']['sample_size'],
    H=counts_true,
    data_filled_slices=data_filled_slices,
    fh=3,
)
logger.info(f'Forecast queries completed')

_raw = np.load(datafile_full)
_raw = ((_raw - min_vals)/(max_vals - min_vals) - 0.5) * max_val_vdr
_loc_ijk = convert_db_to_ijk_aniso(_raw, rho_xy=rho_xy, rho_t=rho_t, max_val=max_val_vdr)
loc_ijk_arr = np.asarray(list(zip(*_loc_ijk)), dtype=int)

hot_levels = config['test'].get('hotspot_levels', [20])
Hc_slow_payload = {}
Hress_slow_payload = {}
H_slow_qs = {}
for _hot in hot_levels:
    _qs, _Hc, _Hress = get_hotspot_queries(_hot, counts_true, loc_ijk_arr, limit=500, radius=50)
    Hc_slow_payload[_hot] = np.asarray(_Hc, dtype=int)
    Hress_slow_payload[_hot] = np.asarray(_Hress, dtype=float).reshape(-1,1)
    H_slow_qs[_hot] = np.asarray(_qs, dtype=int)
logger.info(f'Hotspot queries completed')


# =========================================================
# PrivBayes (same logic; just user-level n := |Ds|, Δ := k)
# =========================================================
time1 = time.time()
beta = 0.5

eps_1 = beta * eps
eps_2 = (1 - beta) * eps

H, W, T = counts.shape
dom = np.array([H, W, T])
height = [int(math.ceil(math.log2(domi))) for domi in dom]

# ---- discretize Ds into D (NOTE: n := |Ds|) ----
n = int(db.shape[0])  # overwrite n to be |Ds| (user-level)
D = np.zeros((n, 3), dtype=int)

range_lon = max_vals[0] - min_vals[0]
range_lat = max_vals[1] - min_vals[1]
range_time = max_vals[2] - min_vals[2]

D[:, 0] = np.floor((db[:, 0] - min_vals[0]) / range_lon * H).astype(int)
D[:, 0] = np.minimum(D[:, 0], H - 1)
D[:, 1] = np.floor((db[:, 1] - min_vals[1]) / range_lat * W).astype(int)
D[:, 1] = np.minimum(D[:, 1], W - 1)
D[:, 2] = np.floor((db[:, 2] - min_vals[2]) / range_time * T).astype(int)
D[:, 2] = np.minimum(D[:, 2], T - 1)

def maximal_parent_sets_star(V, tau):
    if tau < 1:
        return set()
    if len(V) == 0:
        return {frozenset()}
    x = V[0]
    V_minus = V[1:]
    S = set()
    U = set()
    for i in range(height[x]):
        dom_i = (dom[x] - 1) // (2 ** i) + 1
        tau_i = tau / dom_i
        for Z in maximal_parent_sets_star(V_minus, tau_i):
            Z_fs = frozenset(Z)
            if Z_fs in U:
                continue
            U.add(Z_fs)
            S.add(frozenset(Z) | frozenset([(x, i)]))
    for Z in maximal_parent_sets_star(V_minus, tau):
        Z_fs = frozenset(Z)
        if Z_fs in U:
            continue
        S.add(frozenset(Z))
    return S

def compute_R(x_attr, pi, D_local):
    pi_list = [p if isinstance(p, tuple) else (p, 0) for p in pi]
    attrs = [(x_attr, 0)] + pi_list
    D_gen = np.zeros((n, len(attrs)), dtype=int)
    for col, (a, lev) in enumerate(attrs):
        D_gen[:, col] = D_local[:, a] // (2 ** lev)
    joint_count = Counter()
    for row in D_gen:
        joint_count[tuple(row)] += 1

    count_X = Counter()
    count_Pi = Counter()
    for key, cnt in joint_count.items():
        x0 = key[0]
        pi_tup = key[1:]
        count_X[x0] += cnt
        count_Pi[pi_tup] += cnt

    abs_diff_pos = 0.0
    for key, cnt in joint_count.items():
        p = cnt / n
        x0 = key[0]
        pi_tup = key[1:]
        q = (count_X[x0] / n) * (count_Pi[pi_tup] / n)
        if p > q:
            abs_diff_pos += p - q
    return abs_diff_pos

# ---- user-level: n changed; S_R uses this n (same formula) ----
S_R = 3 / n + 2 / (n ** 2)

d = 3
effective_eps_2 = eps_2

N = []
V = []
import random
x1 = random.choice(range(d))
N.append((x1, frozenset()))
V.append(x1)

for ii in range(1, d):
    Omega = []
    A_minus_V = [a for a in range(d) if a not in V]
    for x in A_minus_V:
        tau = n * effective_eps_2 / (4 * d * 32.0 * dom[x])
        max_parent_sets = maximal_parent_sets_star(V, tau)
        if len(max_parent_sets) == 0:
            Omega.append((x, frozenset()))
        else:
            for pi in max_parent_sets:
                Omega.append((x, pi))

    eps_prime = eps_1 / (d * (d - 1) / 2)
    Delta = S_R
    scores = [compute_R(om[0], om[1], D) for om in Omega]

    if scores:
        max_s = max(scores)
        factor = eps_prime / (2 * Delta)
        exp_weights = [math.exp(factor * (s - max_s)) for s in scores]
        sum_w = sum(exp_weights)
        if sum_w == 0:
            probs = [1.0 / len(scores)] * len(scores)
        else:
            probs = [w / sum_w for w in exp_weights]
    else:
        raise ValueError("No candidate parent sets found")

    chosen_idx = np.random.choice(len(Omega), p=probs)
    chosen = Omega[chosen_idx]
    N.append(chosen)
    x, pi = chosen
    V.append(x)

P_star = {}

# ---- conditional distribution noise: user-level Δ=k ----
eps_per_attr = eps_2 / (d * (d - 1) / 2)
sigma_cond = 2 * math.sqrt(2 * math.log(1.25 / delta)) * (float(k) / eps_per_attr)

for idx, (x, pi) in enumerate(N):
    pi_list = [p if isinstance(p, tuple) else (p, 0) for p in pi]
    attrs = [(x, 0)] + pi_list

    dom_gen = [(dom[a] - 1) // (2 ** lev) + 1 for a, lev in attrs]
    shape = tuple(dom_gen)

    joint_counts = np.zeros(shape)
    D_gen = np.zeros((n, len(attrs)), dtype=int)

    for col, (a, lev) in enumerate(attrs):
        D_gen[:, col] = D[:, a] // (2 ** lev)

    for row in D_gen:
        joint_counts[tuple(row)] += 1

    noisy_counts_m = joint_counts + np.random.normal(0, sigma_cond, shape)
    noisy_counts_m = np.maximum(0, noisy_counts_m)

    alpha = 1.0
    noisy_counts_m = noisy_counts_m + alpha
    sum_noisy = np.sum(noisy_counts_m)
    noisy_pr = noisy_counts_m / sum_noisy

    P_star[idx] = (attrs, noisy_pr)

# ---- synthesize D_star with size = n (same as original); then build histogram ----
D_star = np.zeros((n, 3), dtype=int)
for j in range(n):
    sampled = {}
    for i in range(d):
        x_attr, pi = N[i]
        pi_list = [p if isinstance(p, tuple) else (p, 0) for p in pi]
        pi_vals = tuple(sampled.get(a, 0) // (2 ** lev) for a, lev in pi_list)

        attrs, noisy_pr = P_star[i]
        slice_joint = noisy_pr[(slice(None),) + pi_vals]
        marg_pi = np.sum(slice_joint)
        if marg_pi == 0:
            cond = np.ones(dom[x_attr]) / dom[x_attr]
        else:
            cond = slice_joint / marg_pi
        x_val = np.random.choice(dom[x_attr], p=cond)
        sampled[x_attr] = x_val

    D_star[j, 0] = sampled[0]
    D_star[j, 1] = sampled[1]
    D_star[j, 2] = sampled[2]

data_rec = np.zeros((H, W, T), dtype=np.float64)
for row in D_star:
    h0, w0, t0 = row
    data_rec[h0, w0, t0] += 1.0

time2 = time.time()
traintime = time2 - time1
results_stats = ResultStats(config)

# ===== Data cleaning and validation (keep unchanged) =====
if np.any(np.isnan(data_rec)) or np.any(np.isinf(data_rec)):
    logger.warning(f'data_rec contains {np.sum(np.isnan(data_rec))} NaN and {np.sum(np.isinf(data_rec))} inf values')
    data_rec = np.nan_to_num(data_rec, nan=0.0, posinf=0.0, neginf=0.0)

data_rec = np.maximum(0, data_rec)

nonzero_cells = np.sum(data_rec > 0)
total_cells = data_rec.size
logger.info(f'data_rec: {nonzero_cells}/{total_cells} ({100*nonzero_cells/total_cells:.2f}%) non-zero cells')
logger.info(f'data_rec stats: min={np.min(data_rec):.2f}, max={np.max(data_rec):.2f}, '
            f'mean={np.mean(data_rec):.2f}, median={np.median(data_rec):.2f}')

# ======  global calibration gamma (post-processing; does not affect DP) ======
data_rec_cal = np.maximum(gamma * data_rec, 0.0)
noisy_cal = np.maximum(gamma * noisy_counts, 0.0)

# ====== eval: counts_true vs gamma*reco (align with DP) ======
# mae, re = get_eval_results(counts_true, data_rec_cal, test_samples, sm=config['test']['sm'])
# logger.info(f'PrivBayes(UserLevel), MAE: {mae}, RE: {re}')

# hot_res = gather_hotspot_results(
#     reco_grid=data_rec_cal,
#     H=counts_true,
#     hot_levels=hot_levels,
#     H_slow_qs=H_slow_qs,
#     Hress_slow_payload=Hress_slow_payload,
#     radius=50
# )
# for lv in hot_levels:
#     logger.info(f"[Hotspot] MAE={hot_res['mae'][lv]:.4f}")

# fmae, fsmape, n_eff = gather_forecasting_results(
#     reco_grid=data_rec_cal,
#     H=counts_true,
#     fcast_qs=fcast_qs,
#     fh=forecast_horizon
# )
# logger.info(f"[Forecast] fh={forecast_horizon} n={n_eff} sMAPE={fsmape:.4f}")
# ===== save =====
save_path = config['train']['save_dir'] + '/{}/{}/eps_{}_userlevel_k{}'.format(
    config['datasets']['name'],
    config['datasets']['sample_size'],
    eps,
    k
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
np.save(save_path + '/published_data_rec_PrivBayes_userlevel.npy', data_rec)
np.save(save_path + '/published_data_rec_PrivBayes_userlevel_gamma.npy', data_rec_cal)

logger.info("Saved:")
logger.info(save_path + "/published_data_rec_PrivBayes_userlevel.npy")
logger.info(save_path + "/published_data_rec_PrivBayes_userlevel_gamma.npy")
# logger.info('min_cell_count_re: {}'.format(min_cell_count_re))
# logger.info('min_hotspot_mae: {}'.format(min_hotspot_mae))
# logger.info('min_forecast_smape: {}'.format(min_forecast_smape))
logger.info("PrivBayes (user-level, overlap-conserving, gamma-calibrated) finished.")
