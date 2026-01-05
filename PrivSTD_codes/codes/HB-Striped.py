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


config_parser = ConfigParser(name='HB-Striped', save_dir='./')
logger = config_parser.get_logger(config_parser.exper_name)

torch.manual_seed(2024)
np.random.seed(2024)
os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpu']
logger.info(f'config: {config}')

db, min_vals, max_vals, n = read_dataset(config)
# delta for gaussian mechanism
delta = 1 / (n ** 2)
logger.info(f'max_lon: {max_vals[0]}, min_lon: {min_vals[0]}, max_lat: {max_vals[1]}, min_lat: {min_vals[1]}')
logger.info(f'time_interval: {(max_vals[2] - min_vals[2]) / 3600} hours')
logger.info(f'number of samples: {n}')

counts, test_samples = get_counts(config, db, min_vals, max_vals, config['datasets']['sample_size'], config['datasets']['time_grid'])
counts = counts[:, :, :config['datasets']['time_grid']]
logger.info(f'counts shape: {counts.shape}')
logger.info(f'counts_max: {np.max(counts)}, counts_min: {np.min(counts)}, counts_mean: {np.mean(counts)}, '
           f'median: {np.median(counts)}, counts_sum: {np.sum(counts)}')

# add noise
eps = config['privacy']['eps']
# sigma = math.sqrt(2)/(math.sqrt(2*rho_min))
# print('Gaussian Sigma:', sigma)
sigma = math.sqrt(2 * math.log(1.25 / delta)) / eps
noise = np.random.normal(0, sigma, counts.shape)
noisy_counts = counts + noise
# normalize to [0, 1]
# noisy_counts = (noisy_counts - np.min(noisy_counts)) / (np.max(noisy_counts) - np.min(noisy_counts))
logger.info(f'noisy_max: {np.max(noisy_counts)}, noisy_min: {np.min(noisy_counts)}, '
           f'noisy_mean: {np.mean(noisy_counts)}, noisy_median: {np.median(noisy_counts)}, '
           f'noisy_sum: {np.sum(noisy_counts)}')

# noisy_counts: [H, W, N]
train_data = np.transpose(noisy_counts.copy(), (2, 0, 1))  # [N, H, W]
train_data = np.expand_dims(train_data, axis=1)  # [N, C, H, W]
train_dataset = MVDataset(train_data, img_size=config['net']['img_size'], is_train=True)
train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=False, num_workers=8)


# ====== one-time preparation (modify two lines) ======
H_, W_, T_ = counts.shape
max_val_vdr = float(config['datasets'].get('max_val', 10.0))
rho_xy = max_val_vdr / float(H_)  # use H for space
rho_t = max_val_vdr / float(T_)  # use T for time (fix point)
data_filled_slices = compute_data_filled_slices_from_counts(counts)
forecast_horizon = int(config['test'].get('forecast_horizon', 3))
datafile = './data/' + config['datasets']['name'] + '.npy'
fcast_qs, _h_mae_ref, _h_mape_ref = get_queries_for_forecasting_vdr_exact(
    datafile=datafile, max_val=max_val_vdr, min_vals=min_vals, max_vals=max_vals,
    rho_xy=rho_xy,  # <== new
    rho_t=rho_t,  # <== new
    test_size=config['datasets']['sample_size'],
    H=counts, data_filled_slices=data_filled_slices, fh=3,
)
logger.info(f'Forecast queries completed')

# Hotspot preparation remains the same, but also change the coordinate mapping to anisotropic:
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


# HB-Striped method replacement

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

padded_counts = np.pad(counts, ((0, H_p - H_), (0, W_p - W_), (0, T_p - T_)), mode='constant').astype(float)
padded_noisy_counts = np.pad(noisy_counts, ((0, H_p - H_), (0, W_p - W_), (0, T_p - T_)), mode='constant').astype(float)

# Build levels bottom-up for true sums (for structure, but not used for DP)
levels = []
current = padded_counts.copy()
levels.append(current)

while max(current.shape) > 1:
    new_shape = tuple(math.ceil(s / b) for s in current.shape)
    new = np.zeros(new_shape)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            for k in range(new_shape[2]):
                sub = current[i*b:(i+1)*b, j*b:(j+1)*b, k*b:(k+1)*b]
                new[i, j, k] = sub.sum()
    levels.append(new)
    current = new

num_levels = len(levels)  # h+1 levels, index 0: leaf (level 1), index h: root (level h+1)
h = num_levels - 1  # number of noised levels

# Compute sigma_level = sigma * sqrt(h) to match variance scaling
sigma_level = sigma * math.sqrt(h)

# Create noisy levels
noisy_levels = [levels[l].copy() for l in range(num_levels)]
for l in range(h):  # noise levels 0 to h-1
    noise = np.random.normal(0, sigma_level, noisy_levels[l].shape)
    noisy_levels[l] += noise

# Weighted averaging (bottom-up)
z_levels = [noisy_levels[0].copy()]
sum_child_list = []
for l in range(1, num_levels):
    new_shape = levels[l].shape
    sum_child = np.zeros(new_shape)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            for k in range(new_shape[2]):
                sub = z_levels[l-1][i*b:(i+1)*b, j*b:(j+1)*b, k*b:(k+1)*b]
                sum_child[i, j, k] = sub.sum()
    sum_child_list.append(sum_child)

    i = l  # level index starting from 1 for leaf
    alpha = (c ** (i + 1) - c ** i) / (c ** (i + 1) - 1)
    beta = (c ** i - 1) / (c ** (i + 1) - 1)
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
            for k in range(parent_shape[2]):
                diff = n_bar_levels[-1][i, j, k] - sum_child[i, j, k]
                diff_per_child = diff / c
                adjustment[i*b:(i+1)*b, j*b:(j+1)*b, k*b:(k+1)*b] += diff_per_child
    n_bar = z_levels[l] + adjustment
    n_bar_levels.append(n_bar)

# The published histogram is the leaf level after inference, cropped to original size
data_rec = n_bar_levels[-1][:H_, :W_, :T_]

time2 = time.time()
print(f'Running time: {time2 - time1}')

# Evaluation (same as in the original code's eval block)
# mae, re = get_eval_results(counts, data_rec, test_samples, sm=config['test']['sm'])
# logger.info(f'MAE: {mae}, RE: {re}')

# hot_res = gather_hotspot_results(
#     reco_grid=data_rec, H=counts, hot_levels=hot_levels,
#     H_slow_qs=H_slow_qs, Hress_slow_payload=Hress_slow_payload, radius=50
# )
# for lv in hot_levels:
#     logger.info(f"[Hotspot] MAE={hot_res['mae'][lv]:.4f}")

# # Forecast evaluation (FMAE / sMAPE)
# fmae, fsmape, n_eff = gather_forecasting_results(
#     reco_grid=data_rec, H=counts, fcast_qs=fcast_qs, fh=forecast_horizon
# )
# logger.info(f"[Forecast] fh={forecast_horizon} n={n_eff} sMAPE={fsmape:.4f}")
# save published npy
save_path = config['train']['save_dir'] + '/{}/{}/eps_{}'.format(
    config['datasets']['name'], config['datasets']['sample_size'], eps
)
os.makedirs(save_path, exist_ok=True)
np.save(save_path + '/published_data_rec_HB-Striped.npy', data_rec)

# min_cell_count_re = 999
# min_hotspot_mae = 999
# min_forecast_smape = 999
# if min_cell_count_re > re[0]:
#     min_cell_count_re = re[0]
# if min_hotspot_mae > hot_res['mae'][config['test']['sm'][0]]:
#     min_hotspot_mae = hot_res['mae'][config['test']['sm'][0]]
# if min_forecast_smape > fsmape:
#     min_forecast_smape = fsmape

logger.info('data saved at'+ save_path + '/published_data_rec_HB-Striped.npy')
# logger.info('min_cell_count_re: {}'.format(min_cell_count_re))
# logger.info('min_hotspot_mae: {}'.format(min_hotspot_mae))
# logger.info('min_forecast_smape: {}'.format(min_forecast_smape))
logger.info("HB-Striped finished.")

