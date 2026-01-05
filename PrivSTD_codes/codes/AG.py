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
    For a coarse cell (p,q) in longitude/latitude directions, construct 1D overlap weight matrices
    A_w (I_p x m2) and B_w (J_p x m2).
    Weights equal (overlap length / sub-cell length), used for an area-normalized separable approximation:
      A_w @ U @ B_w^T  == sum_{sub_i,sub_j} (overlap_lon/Δx_sub)*(overlap_lat/Δy_sub)*U
    """
    p, p1 = p_range
    q, q1 = q_range
    # Cell index ranges in the original grid that fall within this coarse cell's lon/lat span
    I_idx = np.arange(p, p1)   # longitude-direction cell indices (left-closed, right-open)
    J_idx = np.arange(q, q1)   # latitude-direction cell indices

    # Original-grid cell boundaries; note lon_bins/lat_bins lengths are H_+1 and W_+1
    L_left  = lon_bins[I_idx][:, None]      # (I_p,1)
    L_right = lon_bins[I_idx + 1][:, None]
    T_left  = lat_bins[J_idx][:, None]      # (J_p,1)
    T_right = lat_bins[J_idx + 1][:, None]

    # Sub-cell boundaries
    S_left  = sub_lon_bins[None, :-1]       # (1,m2)
    S_right = sub_lon_bins[None, 1:]
    U_left  = sub_lat_bins[None, :-1]       # (1,m2)
    U_right = sub_lat_bins[None, 1:]

    # 1D overlap lengths
    A = overlapping_lengths_1d(L_left, L_right, S_left, S_right)  # (I_p, m2)
    B = overlapping_lengths_1d(T_left, T_right, U_left, U_right)  # (J_p, m2)

    # Sub-cell lengths (to convert overlap lengths into separable "area share" weights)
    sub_lon_len = (S_right - S_left)  # (1, m2)
    sub_lat_len = (U_right - U_left)  # (1, m2)

    # Avoid division by zero (theoretically shouldn't happen since m2>=1 and boundaries are strictly increasing)
    sub_lon_len = np.clip(sub_lon_len, 1e-12, None)
    sub_lat_len = np.clip(sub_lat_len, 1e-12, None)

    A_w = A / sub_lon_len  # (I_p, m2)
    B_w = B / sub_lat_len  # (J_p, m2)
    return I_idx, J_idx, A_w.astype(np.float32), B_w.astype(np.float32)

def find_covering_cell_indices(bin_edges, left, right):
    """
    Given a continuous interval [left, right], find the "original-grid cell index range".
    Return (i0, i1) meaning covered cell indices satisfy i0 <= i < i1.
    """
    # First cell: its right boundary > left -> i0 = searchsorted(edges, left, 'right') - 1
    i0 = np.searchsorted(bin_edges, left, side='right') - 1
    i0 = max(i0, 0)
    # One past the last cell: its left boundary < right -> i1 = searchsorted(edges, right, 'left')
    i1 = np.searchsorted(bin_edges, right, side='left')
    i1 = min(i1, len(bin_edges) - 1)
    if i1 <= i0:
        i1 = i0  # empty-interval protection
    return i0, i1

# ---------------------------
# main
# ---------------------------
config_parser = ConfigParser(name='MultiView', save_dir='./')
logger = config_parser.get_logger(config_parser.exper_name)
torch.manual_seed(2024)
np.random.seed(2024)
os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpu']
logger.info(f'config: {config}')

db, min_vals, max_vals, n = read_dataset(config)
# total (ε, δ)
eps = float(config['privacy']['eps'])
delta_total = 1.0 / (n ** 2)
logger.info(f'max_lon: {max_vals[0]}, min_lon: {min_vals[0]}, max_lat: {max_vals[1]}, min_lat: {min_vals[1]}')
logger.info(f'time_interval: {(max_vals[2] - min_vals[2]) / 3600} hours')
logger.info(f'number of samples: {n}')

counts, test_samples = get_counts(
    config, db, min_vals, max_vals, config['datasets']['sample_size'], config['datasets']['time_grid']
)
counts = counts[:, :, :config['datasets']['time_grid']]
H_, W_, T_ = counts.shape
logger.info(f'counts shape: {counts.shape}')
logger.info(f'counts_max: {np.max(counts)}, counts_min: {np.min(counts)}, counts_mean: {np.mean(counts)}, '
            f'median: {np.median(counts)}, counts_sum: {np.sum(counts)}')

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
delta_coarse = delta_total/2
delta_fine = delta_total/2

sigma_coarse = gaussian_sigma(eps_coarse, delta_coarse)   # Coarse-layer Gaussian noise
sigma_fine   = gaussian_sigma(eps_fine,   delta_fine)     # Fine-layer Gaussian noise

# Uniform bins of the original grid
lon_bins = np.linspace(min_vals[0], max_vals[0], H_ + 1)
lat_bins = np.linspace(min_vals[1], max_vals[1], W_ + 1)
time_bins = np.linspace(min_vals[2], max_vals[2], T_ + 1)

# Empirical m1 formula (keep your style; ceil matches the paper better; switch back to round if needed)
m_ug = int(np.ceil(np.sqrt(n * eps / c)))
m1 = max(10, int(np.ceil(m_ug / 4.0)))
logger.info(f'AdaptiveGrid m1 (coarse grid): {m1}')

# Coarse-grid boundaries
lon_bins_coarse = np.linspace(min_vals[0], max_vals[0], m1 + 1)
lat_bins_coarse = np.linspace(min_vals[1], max_vals[1], m1 + 1)

# 1) Get coarse counts once (no noise)
counts_coarse = np.histogramdd(
    db, bins=(lon_bins_coarse, lat_bins_coarse, time_bins)
)[0].astype(np.float32)  # shape (m1, m1, T_)

# 2) Add Gaussian noise to coarse counts
noisy_coarse = counts_coarse + np.random.normal(
    loc=0.0, scale=sigma_coarse, size=counts_coarse.shape
).astype(np.float32)

# 3) Pre-assign each point to a coarse cell (p,q) for fine refinement (avoid repeated masking)
p_idx = np.clip(np.searchsorted(lon_bins_coarse, db[:, 0], side='right') - 1, 0, m1 - 1)
q_idx = np.clip(np.searchsorted(lat_bins_coarse, db[:, 1], side='right') - 1, 0, m1 - 1)
pq_flat = p_idx * m1 + q_idx
order = np.argsort(pq_flat)
db_sorted = db[order]
pq_sorted = pq_flat[order]
starts = np.searchsorted(pq_sorted, np.arange(m1 * m1), side='left')
ends   = np.searchsorted(pq_sorted, np.arange(m1 * m1), side='right')

adaptive_grids = []
# 4) For each coarse cell: adaptive refinement + fine counts + Gaussian noise + constrained inference (vectorized over T)
for p in tqdm(range(m1), desc='Fine partition per coarse cell (Gaussian DP)'):
    for q in range(m1):
        # Aggregate N' from noisy coarse counts (across T); used for choosing m2 (keep your original strategy)
        N_prime = float(np.sum(noisy_coarse[p, q, :]))
        # Guideline2: m2 = ceil( sqrt( N' * (1-α) * eps / c2 ) )
        m2 = max(1, int(np.ceil(np.sqrt(max(0.0, N_prime * (1.0 - alpha) * eps / c2)))))
        # Sub-cell boundaries
        sub_lon_bins = np.linspace(lon_bins_coarse[p],   lon_bins_coarse[p + 1],   m2 + 1)
        sub_lat_bins = np.linspace(lat_bins_coarse[q],   lat_bins_coarse[q + 1],   m2 + 1)

        # Extract data within this coarse cell (O(1) slicing)
        idx_flat = p * m1 + q
        s, e = starts[idx_flat], ends[idx_flat]
        db_sub = db_sorted[s:e]

        # Fine-grid true counts (histogram over time as well)
        if e > s:
            counts_fine = np.histogramdd(
                db_sub, bins=(sub_lon_bins, sub_lat_bins, time_bins)
            )[0].astype(np.float32)  # (m2, m2, T_)
        else:
            counts_fine = np.zeros((m2, m2, T_), dtype=np.float32)

        # Add Gaussian noise (fine layer: ((1-α)ε, δ/2))
        noisy_fine = counts_fine + np.random.normal(
            loc=0.0, scale=sigma_fine, size=counts_fine.shape
        ).astype(np.float32)

        # Constrained inference (per time slice): pull leaf sums back toward a convex combination v' (vectorized over T)
        denom = (1.0 - alpha)**2 + (alpha**2) * (m2**2)
        coeff_v   = (alpha**2) * (m2**2) / denom
        coeff_sum = (1.0 - alpha)**2 / denom
        sum_u = noisy_fine.sum(axis=(0, 1), keepdims=True)          # (1,1,T_)
        v    = noisy_coarse[p, q, :][None, None, :]                 # (1,1,T_)
        v_prime = coeff_v * v + coeff_sum * sum_u                   # (1,1,T_)
        noisy_fine += (v_prime - sum_u) / (m2**2)                   # broadcast update

        adaptive_grids.append({
            'p': p, 'q': q, 'm2': m2,
            'sub_lon_bins': sub_lon_bins.astype(np.float64),
            'sub_lat_bins': sub_lat_bins.astype(np.float64),
            'noisy_fine': noisy_fine.astype(np.float32)
        })

# ---------------------------
# Reconstruct data_rec (vectorized blocks: A_w @ U @ B_w^T)
# ---------------------------
data_rec = np.zeros_like(counts, dtype=np.float32)

for grid in tqdm(adaptive_grids, desc='Reconstructing full grid from fine leaves'):
    p = grid['p']; q = grid['q']; m2 = grid['m2']
    sub_lon_bins = grid['sub_lon_bins']; sub_lat_bins = grid['sub_lat_bins']
    noisy_fine   = grid['noisy_fine']   # (m2, m2, T_)

    coarse_lon_left,  coarse_lon_right  = lon_bins_coarse[p],   lon_bins_coarse[p + 1]
    coarse_lat_left,  coarse_lat_right  = lat_bins_coarse[q],   lat_bins_coarse[q + 1]

    # Find the original-grid cell index ranges covering this coarse cell
    i0, i1 = find_covering_cell_indices(lon_bins, coarse_lon_left, coarse_lon_right)
    j0, j1 = find_covering_cell_indices(lat_bins, coarse_lat_left, coarse_lat_right)
    if i1 <= i0 or j1 <= j0:
        continue

    # Construct separable 1D weight matrices
    I_idx, J_idx, A_w, B_w = build_block_weights(
        lon_bins, lat_bins, sub_lon_bins, sub_lat_bins, (i0, i1), (j0, j1)
    )  # A_w: (I_p,m2), B_w: (J_p,m2)

    # Batch bilinear multiplication via einsum: (I_p,m2) @ (m2,m2,T) @ (J_p,m2)^T
    # Result: (I_p, J_p, T_)
    block = np.einsum('im,mnt,jn->ijt', A_w, noisy_fine, B_w, optimize=True)

    # Write back to the corresponding sub-block of data_rec
    data_rec[I_idx[:, None], J_idx[None, :], :] += block.astype(np.float32)

time2 = time.time()
traintime = time2 - time1

# ---------------------------
# Statistics and evaluation
# ---------------------------
noisy_values = np.concatenate([g['noisy_fine'].ravel() for g in adaptive_grids])
logger.info(f'noisy_max: {np.max(noisy_values)}, noisy_min: {np.min(noisy_values)}, '
            f'noisy_mean: {np.mean(noisy_values)}, noisy_median: {np.median(noisy_values)}, '
            f'noisy_sum: {np.sum(noisy_values)}')

# As the ID baseline: directly apply (ε,δ)-DP Gaussian noise to the original counts (same total budget as above)
sigma_total = gaussian_sigma(eps, delta_total)
noisy_counts_fine = counts + np.random.normal(0.0, sigma_total, counts.shape).astype(np.float32)
id_mae, id_re = get_eval_results(counts, noisy_counts_fine, test_samples, sm=config['test']['sm'])

# ====== one-time preparation (keep your original logic, only fix anisotropic parameters) ======
max_val_vdr = float(config['datasets'].get('max_val', 10.0))
rho_xy = max_val_vdr / float(H_)   # Use H for spatial
rho_t  = max_val_vdr / float(T_)   # Use T for temporal
data_filled_slices = compute_data_filled_slices_from_counts(counts)
forecast_horizon = int(config['test'].get('forecast_horizon', 3))
datafile = './data/' + config['datasets']['name'] + '.npy'

fcast_qs, _h_mae_ref, _h_mape_ref = get_queries_for_forecasting_vdr_exact(
    datafile=datafile, max_val=max_val_vdr, min_vals=min_vals, max_vals=max_vals,
    rho_xy=rho_xy, rho_t=rho_t, test_size=config['datasets']['sample_size'],
    H=counts, data_filled_slices=data_filled_slices, fh=3,
)
logger.info(f'Forecast queries completed')

_raw = np.load(datafile)
_raw = ((_raw - min_vals) / (max_vals - min_vals) - 0.5) * max_val_vdr
_loc_ijk = convert_db_to_ijk_aniso(_raw, rho_xy=rho_xy, rho_t=rho_t, max_val=max_val_vdr)
loc_ijk_arr = np.asarray(list(zip(*_loc_ijk)), dtype=int)

hot_levels = config['test'].get('hotspot_levels', [20])
Hc_slow_payload, Hress_slow_payload, H_slow_qs = {}, {}, {}
for _hot in hot_levels:
    _qs, _Hc, _Hress = get_hotspot_queries(_hot, counts, loc_ijk_arr, limit=500, radius=50)
    Hc_slow_payload[_hot] = np.asarray(_Hc, dtype=int)
    Hress_slow_payload[_hot] = np.asarray(_Hress, dtype=float).reshape(-1, 1)
    H_slow_qs[_hot] = np.asarray(_qs, dtype=int)
logger.info(f'Hotspot queries completed')

# Evaluation
# mae, re = get_eval_results(counts, data_rec, test_samples, sm=config['test']['sm'])
# logger.info(f'MAE: {mae}, RE: {re}')

# hot_res = gather_hotspot_results(
#     reco_grid=data_rec, H=counts, hot_levels=hot_levels, H_slow_qs=H_slow_qs,
#     Hress_slow_payload=Hress_slow_payload, radius=50
# )
# for lv in hot_levels:
#     logger.info(f"[Hotspot] MAE={hot_res['mae'][lv]:.4f}")

# fmae, fsmape, n_eff = gather_forecasting_results(
#     reco_grid=data_rec, H=counts, fcast_qs=fcast_qs, fh=forecast_horizon
# )
# logger.info(f"[Forecast] fh={forecast_horizon} n={n_eff} sMAPE={fsmape:.4f}")
# Save
save_path = config['train']['save_dir'] + '/{}/{}/eps_{}'.format(
    config['datasets']['name'], config['datasets']['sample_size'], eps
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

np.save(save_path + '/published_data_rec_AG.npy', data_rec)
logger.info('data saved at'+ save_path + '/published_data_rec_AG.npy')
# logger.info('min_cell_count_re: {}'.format(min_cell_count_re))
# logger.info('min_hotspot_mae: {}'.format(min_hotspot_mae))
# logger.info('min_forecast_smape: {}'.format(min_forecast_smape))
logger.info("AG finished.")
