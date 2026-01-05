import warnings
warnings.filterwarnings(
    'ignore',
    category=RuntimeWarning,
    message='invalid value encountered in divide'
)

import os
import torch
import torch.nn as nn
import numpy as np
from parse import config
import math
from logger.logger import ConfigParser
from utils.dataset import read_dataset, get_counts, MVDataset, pad_data
from torch.utils.data import DataLoader
from model.MultiView import MultiView
from model.DnCNN import DnCNN
from utils.eval import *
from tqdm import tqdm
from utils.results_stats import ResultStats
import time
from collections import Counter


config_parser = ConfigParser(name='MultiView', save_dir='./')
logger = config_parser.get_logger(config_parser.exper_name)
torch.manual_seed(2024)
np.random.seed(2024)
os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpu']
logger.info(f'config: {config}')

db, min_vals, max_vals, n = read_dataset(config)

# delta for Gaussian mechanism
delta = 1 / (n ** 2)

logger.info(f'max_lon: {max_vals[0]}, min_lon: {min_vals[0]}, max_lat: {max_vals[1]}, min_lat: {min_vals[1]}')
logger.info(f'time_interval: {(max_vals[2] - min_vals[2]) / 3600} hours')
logger.info(f'number of samples: {n}')

counts, test_samples = get_counts(
    config, db, min_vals, max_vals,
    config['datasets']['sample_size'],
    config['datasets']['time_grid']
)
counts = counts[:, :, :config['datasets']['time_grid']]
logger.info(f'counts shape: {counts.shape}')
logger.info(
    f'counts_max: {np.max(counts)}, counts_min: {np.min(counts)}, '
    f'counts_mean: {np.mean(counts)}, median: {np.median(counts)}, '
    f'counts_sum: {np.sum(counts)}'
)

# Add noise
eps = config['privacy']['eps']

# Use approximate DP; no need to compute rho_min
sigma = math.sqrt(2 * math.log(1.25 / delta)) / eps
noise = np.random.normal(0, sigma, counts.shape)
noisy_counts = counts + noise

logger.info(
    f'noisy_max: {np.max(noisy_counts)}, noisy_min: {np.min(noisy_counts)}, '
    f'noisy_mean: {np.mean(noisy_counts)}, noisy_median: {np.median(noisy_counts)}, '
    f'noisy_sum: {np.sum(noisy_counts)}'
)

# ====== one-time preparation (two lines modified) ======
H_, W_, T_ = counts.shape
max_val_vdr = float(config['datasets'].get('max_val', 10.0))
rho_xy = max_val_vdr / float(H_)   # spatial resolution uses H
rho_t = max_val_vdr / float(T_)    # temporal resolution uses T (fixed)
data_filled_slices = compute_data_filled_slices_from_counts(counts)
forecast_horizon = int(config['test'].get('forecast_horizon', 3))
datafile = './data/' + config['datasets']['name'] + '.npy'

fcast_qs, _h_mae_ref, _h_mape_ref = get_queries_for_forecasting_vdr_exact(
    datafile=datafile,
    max_val=max_val_vdr,
    min_vals=min_vals,
    max_vals=max_vals,
    rho_xy=rho_xy,
    rho_t=rho_t,
    test_size=config['datasets']['sample_size'],
    H=counts,
    data_filled_slices=data_filled_slices,
    fh=3,
)
logger.info(f'Forecast queries completed')

# Hotspot preparation remains unchanged, but coordinate mapping is updated to anisotropic
_raw = np.load(datafile)
_raw = ((_raw - min_vals) / (max_vals - min_vals) - 0.5) * max_val_vdr
_loc_ijk = convert_db_to_ijk_aniso(_raw, rho_xy=rho_xy, rho_t=rho_t, max_val=max_val_vdr)
loc_ijk_arr = np.asarray(list(zip(*_loc_ijk)), dtype=int)

hot_levels = config['test'].get('hotspot_levels', [20])
Hc_slow_payload = {}
Hress_slow_payload = {}
H_slow_qs = {}
for _hot in hot_levels:
    _qs, _Hc, _Hress = get_hotspot_queries(_hot, counts, loc_ijk_arr, limit=500, radius=50)
    Hc_slow_payload[_hot] = np.asarray(_Hc, dtype=int)
    Hress_slow_payload[_hot] = np.asarray(_Hress, dtype=float).reshape(-1, 1)
    H_slow_qs[_hot] = np.asarray(_qs, dtype=int)
logger.info(f'Hotspot queries completed')


# PrivBayes replacement
time1 = time.time()
beta = 0.5

# Use approximate DP: split privacy budget eps into two parts
eps_1 = beta * eps       # for exponential mechanism to select Bayesian network structure
eps_2 = (1 - beta) * eps # for adding noise to conditional distributions

H, W, T = counts.shape
dom = np.array([H, W, T])
height = [int(math.ceil(math.log2(domi))) for domi in dom]

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
        dom_i = (dom[x] - 1) // (2 ** i) + 1  # Use accurate bin count to avoid underestimation
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


def compute_R(x_attr, pi, D):
    pi_list = [p if isinstance(p, tuple) else (p, 0) for p in pi]
    attrs = [(x_attr, 0)] + pi_list
    D_gen = np.zeros((n, len(attrs)), dtype=int)
    for col, (a, lev) in enumerate(attrs):
        D_gen[:, col] = D[:, a] // (2 ** lev)

    joint_count = Counter()
    for row in D_gen:
        joint_count[tuple(row)] += 1

    count_X = Counter()
    count_Pi = Counter()
    for key, cnt in joint_count.items():
        count_X[key[0]] += cnt
        count_Pi[key[1:]] += cnt

    abs_diff_pos = 0
    for key, cnt in joint_count.items():
        p = cnt / n
        q = (count_X[key[0]] / n) * (count_Pi[key[1:]] / n)
        if p > q:
            abs_diff_pos += p - q
    return abs_diff_pos


S_R = 3 / n + 2 / n ** 2
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
        tau = n * effective_eps_2 / (4 * d * 32 * dom[x])
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
    V.append(chosen[0])


P_star = {}
eps_per_attr = eps_2 / (d * (d - 1) / 2)
sigma = 2 * math.sqrt(2 * math.log(1.25 / delta)) / eps_per_attr

# Build P_star
for idx, (x, pi) in enumerate(N):
    pi_list = [p if isinstance(p, tuple) else (p, 0) for p in pi]
    attrs = [(x, 0)] + pi_list

    # Correct domain size computation
    dom_gen = [(dom[a] - 1) // (2 ** lev) + 1 for a, lev in attrs]
    shape = tuple(dom_gen)

    joint_counts = np.zeros(shape)
    D_gen = np.zeros((n, len(attrs)), dtype=int)

    for col, (a, lev) in enumerate(attrs):
        D_gen[:, col] = D[:, a] // (2 ** lev)

    for row in D_gen:
        joint_counts[tuple(row)] += 1

    noisy_counts_m = joint_counts + np.random.normal(0, sigma, shape)
    noisy_counts_m = np.maximum(0, noisy_counts_m)

    alpha = 1.0
    noisy_counts_m += alpha
    noisy_pr = noisy_counts_m / np.sum(noisy_counts_m)

    P_star[idx] = (attrs, noisy_pr)


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
        sampled[x_attr] = np.random.choice(dom[x_attr], p=cond)

    D_star[j] = [sampled[0], sampled[1], sampled[2]]


data_rec = np.zeros((H, W, T))
for h, w, t in D_star:
    data_rec[h, w, t] += 1

time2 = time.time()
traintime = time2 - time1
results_stats = ResultStats(config)

# ===== Data cleaning and validation =====
# 1. Check and fix invalid values
if np.any(np.isnan(data_rec)) or np.any(np.isinf(data_rec)):
    logger.warning(
        f'data_rec contains {np.sum(np.isnan(data_rec))} NaN and '
        f'{np.sum(np.isinf(data_rec))} inf values'
    )
    data_rec = np.nan_to_num(data_rec, nan=0.0, posinf=0.0, neginf=0.0)

# 2. Ensure non-negativity (count data)
data_rec = np.maximum(0, data_rec)

# 3. Log data quality information
nonzero_cells = np.sum(data_rec > 0)
total_cells = data_rec.size
logger.info(
    f'data_rec stats: min={np.min(data_rec):.2f}, max={np.max(data_rec):.2f}, '
    f'mean={np.mean(data_rec):.2f}, median={np.median(data_rec):.2f}'
)

# min_cell_count_re = 999
# min_hotspot_mae = 999
# min_forecast_smape = 999

# mae, re = get_eval_results(counts, data_rec, test_samples, sm=config['test']['sm'])
# logger.info(f'PrivBayes, MAE: {mae}, RE: {re}')

# hot_res = gather_hotspot_results(
#     reco_grid=data_rec,
#     H=counts,
#     hot_levels=hot_levels,
#     H_slow_qs=H_slow_qs,
#     Hress_slow_payload=Hress_slow_payload,
#     radius=50
# )
# for lv in hot_levels:
#     logger.info(f"[Hotspot] MAE={hot_res['mae'][lv]:.4f}")

# fmae, fsmape, n_eff = gather_forecasting_results(
#     reco_grid=data_rec,
#     H=counts,
#     fcast_qs=fcast_qs,
#     fh=forecast_horizon
# )
# logger.info(f"[Forecast] fh={forecast_horizon} n={n_eff} sMAPE={fsmape:.4f}")


save_path = config['train']['save_dir'] + '/{}/{}/eps_{}'.format(
    config['datasets']['name'],
    config['datasets']['sample_size'],
    eps
)
# if min_cell_count_re > re[0]:
#     min_cell_count_re = re[0]
# if min_hotspot_mae > hot_res['mae'][config['test']['sm'][0]]:
#     min_hotspot_mae = hot_res['mae'][config['test']['sm'][0]]
# if min_forecast_smape > fsmape:
#     min_forecast_smape = fsmape

np.save(save_path + '/published_data_rec_PrivBayes.npy', data_rec)
logger.info('data saved at'+ save_path + '/published_data_rec_PrivBayes.npy')
# logger.info('min_cell_count_re: {}'.format(min_cell_count_re))
# logger.info('min_hotspot_mae: {}'.format(min_hotspot_mae))
# logger.info('min_forecast_smape: {}'.format(min_forecast_smape))
logger.info("PrivBayes finished.")
