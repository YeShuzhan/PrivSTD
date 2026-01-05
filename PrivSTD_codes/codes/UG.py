import os
import math
import time
import numpy as np
import torch
from tqdm import tqdm

from parse import config
from logger.logger import ConfigParser
from utils.dataset import read_dataset, get_counts, MVDataset, pad_data
from utils.eval import *
from utils.results_stats import ResultStats


# ---------------------------
# DP helpers
# ---------------------------
def gaussian_sigma(eps: float, delta: float, sensitivity: float = 1.0) -> float:
    """Std of the (ε, δ)-DP Gaussian mechanism; default L2 sensitivity = 1."""
    if eps <= 0:
        raise ValueError("epsilon must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return math.sqrt(2.0 * math.log(1.25 / delta)) * (sensitivity / eps)


# ---------------------------
# main
# ---------------------------
config_parser = ConfigParser(name='UG', save_dir='./')
logger = config_parser.get_logger(config_parser.exper_name)
torch.manual_seed(2024)
np.random.seed(2024)
os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpu']
logger.info(f'config: {config}')

# Read dataset
db, min_vals, max_vals, n = read_dataset(config)
# Recommended delta for event-level (add/remove) DP
delta = 1.0 / (n ** 2)

logger.info(f'max_lon: {max_vals[0]}, min_lon: {min_vals[0]}, max_lat: {max_vals[1]}, min_lat: {min_vals[1]}')

logger.info(f'number of samples: {n}')

counts, test_samples = get_counts(
    config, db, min_vals, max_vals, config['datasets']['cell_size'], config['datasets']['time_grid']
)
counts = counts[:, :, :config['datasets']['time_grid']]
H_, W_, T_ = counts.shape
logger.info(f'counts shape: {counts.shape}')
logger.info(f'counts_max: {np.max(counts)}, counts_min: {np.min(counts)}, counts_mean: {np.mean(counts)}, '
            f'median: {np.median(counts)}, counts_sum: {np.sum(counts)}')

# ---------------------------
# UniformGrid with Gaussian DP
# ---------------------------
eps = float(config['privacy']['eps'])
# c can be tuned in a small range (e.g., [5, 15]); here we use 10 as a default starting point
c = float(config['privacy'].get('c_for_ug', 10.0))
m = int(np.round(np.sqrt(n * eps / c)))
m = max(1, m)
logger.info(f'UniformGrid m: {m} (from c={c})')

# Fine-grid and time boundaries (aligned with get_counts)
lon_bins = np.linspace(min_vals[0], max_vals[0], H_ + 1)
lat_bins = np.linspace(min_vals[1], max_vals[1], W_ + 1)
time_bins = np.linspace(min_vals[2], max_vals[2], T_ + 1)

# UG coarse grid boundaries
lon_bins_ug = np.linspace(min_vals[0], max_vals[0], m + 1)
lat_bins_ug = np.linspace(min_vals[1], max_vals[1], m + 1)

# 1) Count on the coarse grid (no noise)
counts_ug = np.histogramdd(db, bins=(lon_bins_ug, lat_bins_ug, time_bins))[0].astype(np.float32)

# 2) Add Gaussian noise once to the coarse-grid counts (sensitivity=1)
sigma = gaussian_sigma(eps, delta, sensitivity=1.0)
noisy_counts_ug = counts_ug + np.random.normal(0.0, sigma, counts_ug.shape).astype(np.float32)

logger.info(f'noisy_max: {np.max(noisy_counts_ug)}, noisy_min: {np.min(noisy_counts_ug)}, '
            f'noisy_mean: {np.mean(noisy_counts_ug)}, noisy_median: {np.median(noisy_counts_ug)}, '
            f'noisy_sum: {np.sum(noisy_counts_ug)}')

# 3) Mass-conserving backfill: overlap-area ratios + einsum (fully vectorized)
time1 = time.time()

# Longitude overlap ratios (H_, m)
lon_start = lon_bins[:-1, None]     # (H_, 1)
lon_end   = lon_bins[1:,  None]     # (H_, 1)
ug_lon_s  = lon_bins_ug[None, :-1]  # (1, m)
ug_lon_e  = lon_bins_ug[None,  1:]  # (1, m)

overlap_lon = np.clip(np.minimum(lon_end, ug_lon_e) - np.maximum(lon_start, ug_lon_s), 0.0, None)
ug_lon_w    = np.maximum(ug_lon_e - ug_lon_s, 1e-12)
lon_overlap = (overlap_lon / ug_lon_w).astype(np.float32)  # (H_, m)

# Latitude overlap ratios (W_, m)
lat_start = lat_bins[:-1, None]     # (W_, 1)
lat_end   = lat_bins[1:,  None]     # (W_, 1)
ug_lat_s  = lat_bins_ug[None, :-1]  # (1, m)
ug_lat_e  = lat_bins_ug[None,  1:]  # (1, m)

overlap_lat = np.clip(np.minimum(lat_end, ug_lat_e) - np.maximum(lat_start, ug_lat_s), 0.0, None)
ug_lat_w    = np.maximum(ug_lat_e - ug_lat_s, 1e-12)
lat_overlap = (overlap_lat / ug_lat_w).astype(np.float32)  # (W_, m)

# Mass-conserving reconstruction: for each t, compute A·U·B^T; einsum processes all t at once
# lon_overlap: (H_, m), lat_overlap: (W_, m), noisy_counts_ug: (m, m, T_)
data_rec = np.einsum('ip,jq,pqt->ijt', lon_overlap, lat_overlap, noisy_counts_ug, optimize=True).astype(np.float32)


time2 = time.time()
logger.info(f"data_rec reconstruction completed in {time2 - time1:.2f} seconds")
logger.info(f"data_rec shape: {data_rec.shape}")
logger.info(f"data_rec stats - max: {np.max(data_rec)}, min: {np.min(data_rec)}, "
            f"mean: {np.mean(data_rec)}, sum: {np.sum(data_rec)}")

# ---------------------------
# Evaluation (kept consistent with the original script)
# ---------------------------
# Only as an ID baseline: add one-shot Gaussian noise on the fine grid with the same parameters (do NOT publish)
noisy_counts_fine = counts + np.random.normal(0.0, sigma, counts.shape).astype(np.float32)
id_mae, id_re = get_eval_results(counts, noisy_counts_fine, test_samples, sm=config['test']['sm'])

# ====== One-time preparation (same as original logic; keep anisotropic parameters) ======
max_val_vdr = float(config['datasets'].get('max_val', 10.0))
rho_xy = max_val_vdr / float(H_)   # spatial uses H
rho_t  = max_val_vdr / float(T_)   # time uses T
data_filled_slices = compute_data_filled_slices_from_counts(counts)
forecast_horizon = int(config['test'].get('forecast_horizon', 3))
datafile = './data/' + config['datasets']['name'] + '.npy'

fcast_qs, _h_mae_ref, _h_mape_ref = get_queries_for_forecasting_vdr_exact(
    datafile=datafile, max_val=max_val_vdr, min_vals=min_vals, max_vals=max_vals,
    rho_xy=rho_xy, rho_t=rho_t, test_size=config['datasets']['cell_size'],
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

# min_cell_count_re = 999
# min_hotspot_mae = 999
# min_forecast_smape = 999
# # Aggregate evaluation
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
# if min_cell_count_re > re[0]:
#     min_cell_count_re = re[0]
# if min_hotspot_mae > hot_res['mae'][config['test']['sm'][0]]:
#     min_hotspot_mae = hot_res['mae'][config['test']['sm'][0]]
# if min_forecast_smape > fsmape:
#     min_forecast_smape = fsmape

# Save
save_path = config['train']['save_dir'] + '/{}/{}/eps_{}'.format(
    config['datasets']['name'], config['datasets']['cell_size'], eps
)
os.makedirs(save_path, exist_ok=True)
np.save(save_path + '/published_data_rec_UG.npy', data_rec)
logger.info('data saved at'+ save_path + '/published_data_rec_UG.npy')
# logger.info('min_cell_count_re: {}'.format(min_cell_count_re))
# logger.info('min_hotspot_mae: {}'.format(min_hotspot_mae))
# logger.info('min_forecast_smape: {}'.format(min_forecast_smape))
logger.info("UG finished.")
