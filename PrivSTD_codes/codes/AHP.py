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

# === New: Noise calibration for approximate DP (ε, δ) Gaussian mechanism ===
def gaussian_sigma(epsilon: float, delta: float, sensitivity: float = 1.0) -> float:
    """
    Standard Gaussian mechanism calibration for approximate DP (ε, δ)
    (classical upper bound, commonly used when ε ∈ (0, 1]):
        sigma = S * sqrt(2 ln(1.25/delta)) / epsilon
    where S is the L2 sensitivity; here we take S = 1 for counts / histogram sums.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon

config_parser = ConfigParser(name='MultiView', save_dir='./')
logger = config_parser.get_logger(config_parser.exper_name)
torch.manual_seed(2024)
np.random.seed(2024)
os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpu']
logger.info(f'config: {config}')

db, min_vals, max_vals, n = read_dataset(config)
# Commonly used delta setting under approximate DP (keep your original setting)
delta_total = 1 / (n ** 2)

logger.info(f'max_lon: {max_vals[0]}, min_lon: {min_vals[0]}, max_lat: {max_vals[1]}, min_lat: {min_vals[1]}')

logger.info(f'number of samples: {n}')

counts, test_samples = get_counts(
    config, db, min_vals, max_vals,
    config['datasets']['cell_size'], config['datasets']['time_grid']
)
counts = counts[:, :, :config['datasets']['time_grid']]

logger.info(f'counts shape: {counts.shape}')
logger.info(f'counts_max: {np.max(counts)}, counts_min: {np.min(counts)}, counts_mean: {np.mean(counts)}, '
            f'median: {np.median(counts)}, counts_sum: {np.sum(counts)}')

# =========================
#     AHP - Gaussian version
# =========================
time1 = time.time()
# Budget split: (ε, δ) = (ε1, δ1) + (ε2, δ2)
eps_total = float(config['privacy']['eps'])
# Can also read ratios from config; here we split evenly
eps1 = eps_total / 2.0
eps2 = eps_total / 2.0
delta1 = delta_total / 2.0
delta2 = delta_total / 2.0

# === Modification: use approximate DP Gaussian mechanism for sigma calibration instead of zCDP ===
sigma1 = gaussian_sigma(eps1, delta1, sensitivity=1.0)  # Stage 1: private view used for sorting and clustering
sigma2 = gaussian_sigma(eps2, delta2, sensitivity=1.0)  # Stage 2: add noise to cluster sums and then average

logger.info(f'[Gaussian DP] eps_total={eps_total}, delta_total={delta_total} | '
            f'eps1={eps1}, delta1={delta1}, sigma1={sigma1:.6f} | '
            f'eps2={eps2}, delta2={delta2}, sigma2={sigma2:.6f}')

# ====== Keep your existing evaluation preparation unchanged ======
H_, W_, T_ = counts.shape
max_val_vdr = float(config['datasets'].get('max_val', 10.0))
rho_xy = max_val_vdr / float(H_)   # Spatial dimension uses H
rho_t  = max_val_vdr / float(T_)   # Temporal dimension uses T
data_filled_slices = compute_data_filled_slices_from_counts(counts)
forecast_horizon = int(config['test'].get('forecast_horizon', 3))
datafile = '/home/hyj/MyCodes/MultiView/data/' + config['datasets']['name'] + '.npy'

fcast_qs, _h_mae_ref, _h_mape_ref = get_queries_for_forecasting_vdr_exact(
    datafile=datafile, max_val=max_val_vdr, min_vals=min_vals, max_vals=max_vals,
    rho_xy=rho_xy, rho_t=rho_t, test_size=config['datasets']['cell_size'],
    H=counts, data_filled_slices=data_filled_slices, fh=3,
)
logger.info(f'Forecast queries completed')

_raw = np.load(datafile)
_raw = ((_raw - min_vals)/(max_vals - min_vals) - 0.5) * max_val_vdr
_loc_ijk = convert_db_to_ijk_aniso(_raw, rho_xy=rho_xy, rho_t=rho_t, max_val=max_val_vdr)
loc_ijk_arr = np.asarray(list(zip(*_loc_ijk)), dtype=int)

hot_levels = config['test'].get('hotspot_levels', [20])
Hc_slow_payload = {}
Hress_slow_payload = {}
H_slow_qs = {}
for _hot in hot_levels:
    _qs, _Hc, _Hress = get_hotspot_queries(_hot, counts, loc_ijk_arr, limit=500, radius=50)
    Hc_slow_payload[_hot] = np.asarray(_Hc, dtype=int)
    Hress_slow_payload[_hot] = np.asarray(_Hress, dtype=float).reshape(-1,1)
    H_slow_qs[_hot] = np.asarray(_qs, dtype=int)
logger.info(f'Hotspot queries completed')


# ====== AHP Stage 1: generate "private view", threshold, and sort ======
# === Modification: add Gaussian noise in Stage 1 using (ε1, δ1) to obtain private view H_hat ===
time1 = time.time()
H_hat = counts + np.random.normal(0.0, sigma1, size=counts.shape)

# Flatten for thresholding and sorting
H_hat_flat = H_hat.flatten().copy()
m = H_hat_flat.size

# === Modification: Gaussian-compatible threshold (extreme order statistic) ===
eta = 1.0  # Keep your tuning parameter
theta = eta * math.sqrt(2.0 * math.log(m)) * sigma1
H_hat_flat[H_hat_flat < theta] = 0.0

# === Modification: all subsequent sorting/clustering is based on H_hat_flat (private data),
#                   no longer accessing true values ===
sorted_indices = np.argsort(H_hat_flat)  # Same as original logic (ascending); can use [::-1] for descending

# ====== AHP greedy clustering (performed on private data) ======
clusters = []
i = 0
while i < m:
    cluster_start = i
    # Cluster statistics based on H_hat
    v = H_hat_flat[sorted_indices[i]]
    curr_size = 1
    curr_sum = float(v)
    curr_sq = float(v) ** 2
    # AE = sum(x^2) - sum(x)^2 / k, computed only from H_hat
    curr_ae = 0.0  # AE of a singleton cluster is 0
    # In Stage 2, noise is added to the cluster sum and then averaged,
    # so LE_total = sigma2^2 / k
    curr_err = curr_ae + (sigma2 ** 2) / curr_size
    i += 1

    while i < m:
        v_next = float(H_hat_flat[sorted_indices[i]])
        new_size = curr_size + 1
        new_sum = curr_sum + v_next
        new_sq  = curr_sq  + (v_next ** 2)
        new_ae  = new_sq - (new_sum ** 2) / new_size
        new_err = new_ae + (sigma2 ** 2) / new_size

        # === Modification: conservative singleton lower-bound comparison
        #                   (privacy-correct, simple and stable)
        # If not merging, the upper-bound error of making v_next a singleton cluster:
        # AE = 0, LE = sigma2^2
        no_merge_err_upper = curr_err + (sigma2 ** 2)

        if new_err < no_merge_err_upper:
            # Merging is better
            curr_size = new_size
            curr_sum  = new_sum
            curr_sq   = new_sq
            curr_ae   = new_ae
            curr_err  = new_err
            i += 1
        else:
            break

    clusters.append((cluster_start, i))

# ====== AHP Stage 2: add Gaussian noise to "true cluster sums" and distribute to cluster elements ======
# Note: cluster boundaries are fully determined by H_hat (private data),
#       so accessing true values here is only for producing noisy outputs
counts_flat = counts.flatten().copy()
data_rec_flat = np.zeros_like(counts_flat)

for start, end in clusters:
    if start == end:
        continue
    cl_indices = sorted_indices[start:end]
    cl_size = end - start

    # True cluster sum (sensitivity = 1), add a single N(0, sigma2^2) noise
    true_sum = float(np.sum(counts_flat[cl_indices]))
    noisy_sum = true_sum + np.random.normal(0.0, sigma2)
    # Evenly distribute to each bin in the cluster (released)
    data_rec_flat[cl_indices] = noisy_sum / cl_size

data_rec = data_rec_flat.reshape(counts.shape)
time2 = time.time()
traintime = time2-time1
print("Running time",traintime)
# ====== Evaluation (kept consistent with yours) ======
# mae, re = get_eval_results(counts, data_rec, test_samples, sm=config['test']['sm'])
# logger.info(f'MAE: {mae}, RE: {re}')

# hot_res = gather_hotspot_results(
#     reco_grid=data_rec, H=counts, hot_levels=hot_levels,
#     H_slow_qs=H_slow_qs, Hress_slow_payload=Hress_slow_payload, radius=50
# )
# for lv in hot_levels:
#     logger.info(f"[Hotspot] MAE={hot_res['mae'][lv]:.4f}")

# # Forecasting evaluation (FMAE / sMAPE)
# fmae, fsmape, n_eff = gather_forecasting_results(
#     reco_grid=data_rec, H=counts, fcast_qs=fcast_qs, fh=forecast_horizon
# )
# logger.info(f"[Forecast] fh={forecast_horizon} n={n_eff} sMAPE={fsmape:.4f}")

logger.info('Saving released npy...')

# save npy
save_path = config['train']['save_dir'] + '/{}/{}/eps_{}'.format(
    config['datasets']['name'], config['datasets']['cell_size'], eps_total
)
os.makedirs(save_path, exist_ok=True)
# res_str = 'Gaussian-AHP\tMAE:\t{}\tRE:\t{}\n'.format(mae, re)
np.save(save_path + '/published_data_rec_AHP.npy', data_rec)
# min_cell_count_re = 999
# min_hotspot_mae = 999
# min_forecast_smape = 999
# if min_cell_count_re > re[0]:
#     min_cell_count_re = re[0]
# if min_hotspot_mae > hot_res['mae'][config['test']['sm'][0]]:
#     min_hotspot_mae = hot_res['mae'][config['test']['sm'][0]]
# if min_forecast_smape > fsmape:
#     min_forecast_smape = fsmape

logger.info('data saved at'+ save_path + '/published_data_rec_AHP.npy')
# logger.info('min_cell_count_re: {}'.format(min_cell_count_re))
# logger.info('min_hotspot_mae: {}'.format(min_hotspot_mae))
# logger.info('min_forecast_smape: {}'.format(min_forecast_smape))
logger.info("AHP finished.")
