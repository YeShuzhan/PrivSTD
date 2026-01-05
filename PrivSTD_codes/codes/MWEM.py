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

# =========================
# Numerical-stability utility functions (added)
# =========================
def softmax_stable(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax; includes a fallback for all-NaN/Inf cases."""
    x = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(x)):
        # Fallback: if non-finite, choose argmax
        p = np.zeros_like(x, dtype=np.float64)
        p[int(np.nanargmax(x))] = 1.0
        return p
    x_max = np.max(x)
    z = x - x_max
    # Clip to avoid exp overflow
    z = np.clip(z, -50.0, 50.0)
    e = np.exp(z)
    s = e.sum()
    if s <= 0 or not np.isfinite(s):
        p = np.ones_like(x, dtype=np.float64) / len(x)
    else:
        p = e / s
    # Extra safety: ensure sum is 1
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
        # Fallback: distribute uniformly
        A[...] = target_mass / A.size
    else:
        A *= (target_mass / s)
    nan_to_num_inplace(A, val=0.0)

def safe_exp_update_factor(delta: np.ndarray, denom: float, lr: float, clip_val: float = 30.0):
    """
    Compute the exp factor for multiplicative weight update: exp(lr * delta / denom)
    with clipping to avoid overflow.
    """
    denom = max(float(denom), 1.0)
    expo = lr * (delta / denom)
    expo = np.clip(expo, -clip_val, clip_val)
    return np.exp(expo)

# =========================
# zCDP conversion
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
# Logger & random seeds
# =========================
config_parser = ConfigParser(name='MWEM', save_dir='./')
logger = config_parser.get_logger(config_parser.exper_name)
torch.manual_seed(2024)
np.random.seed(2024)
os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpu']

logger.info(f'config: {config}')
db, min_vals, max_vals, n = read_dataset(config)

# delta for gaussian mechanism
delta = 1 / (n ** 4)
logger.info(f'max_lon: {max_vals[0]}, min_lon: {min_vals[0]}, max_lat: {max_vals[1]}, min_lat: {min_vals[1]}')
logger.info(f'time_interval: {(max_vals[2] - min_vals[2]) / 3600} hours')
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
# One-time global Gaussian noise (if needed)
# =========================
eps = config['privacy']['eps']
rho_min = dp_to_required_zcdp(eps, delta)
# zCDP Gaussian with Δ2=1: sigma = Δ2 / sqrt(2ρ); if rho=0 then disable this step
if rho_min > 0:
    sigma_global = 1.0 / math.sqrt(2.0 * rho_min)
else:
    sigma_global = 0.0  # limiting case when ε=0: do not add noise in this step
print('Global Gaussian Sigma (zCDP):', sigma_global)

if sigma_global > 0:
    noise = np.random.normal(0.0, sigma_global, counts.shape)
    noisy_counts = counts + noise
else:
    noisy_counts = counts.copy()

nan_to_num_inplace(noisy_counts, val=0.0)
logger.info(f'noisy_max: {np.max(noisy_counts)}, noisy_min: {np.min(noisy_counts)}, '
            f'noisy_mean: {np.mean(noisy_counts)}, noisy_median: {np.median(noisy_counts)}, '
            f'noisy_sum: {np.sum(noisy_counts)}')

# ====== one-time preparation (adjust anisotropic grid parameters) ======
H_, W_, T_ = counts.shape
max_val_vdr = float(config['datasets'].get('max_val', 10.0))
rho_xy = max_val_vdr / float(max(H_, 1))  # spatial
rho_t = max_val_vdr / float(max(T_, 1))   # temporal (fix)
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
    test_size=config['datasets']['cell_size'],
    H=counts,
    data_filled_slices=data_filled_slices,
    fh=forecast_horizon,
)
logger.info(f'Forecast queries completed')

# Hotspot preparation (anisotropic coordinate mapping)
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



# =========================
# MEWM (numerically-stable revised version)
# =========================
def get_marginal(arr: np.ndarray, keep):
    if not keep:
        return np.sum(arr)
    sum_axes = tuple(i for i in range(len(arr.shape)) if i not in keep)
    return np.sum(arr, axis=sum_axes)

def update_A(A: np.ndarray, keep, m: np.ndarray, n_slice: float, iter_idx: int):
    """
    Numerically stable multiplicative update:
    - use lr (can decay with iterations)
    - clip exponent
    - renormalize after each update
    """
    # mild annealing can improve stability
    lr = 1.0 / float(1.0 + 0.25 * iter_idx)  # from 1.0 -> decays
    marg_A = get_marginal(A, keep)
    d = m - marg_A
    # broadcast exp factors
    exp_factors = safe_exp_update_factor(d, denom=max(n_slice, 1.0), lr=lr, clip_val=30.0)
    if len(keep) == 0:
        # scalar update: same factor for all entries
        A *= float(exp_factors)
    else:
        factor_shape = [1] * len(A.shape)
        # reshape exp_factors to a broadcastable shape
        if isinstance(keep, tuple) and len(keep) > 0:
            for ii, ax in enumerate(keep):
                factor_shape[ax] = exp_factors.shape[ii] if exp_factors.ndim > ii else 1
        exp_factors = exp_factors.reshape(factor_shape)
        A *= exp_factors
    nan_to_num_inplace(A, val=0.0)
    # renormalize to slice mass
    renorm_to_mass(A, n_slice)

H, W, T_ = counts.shape
T = 20  # Number of MWEM iterations
alpha = config['train']['alpha']  # keep but not directly used
data_rec_list = []
time1 = time.time()

# Temperature/scale for the Exponential Mechanism (scaled by privacy budget / iterations)
exp_mech_scale = (eps / max(2.0 * T, 1.0)) * 0.5  # keep the original 0.5 constant

for slice_idx in tqdm(range(T_), total=T_):
    B = counts[:, :, slice_idx].astype(np.float64)
    noisy_B = noisy_counts[:, :, slice_idx].astype(np.float64)

    n_slice = float(np.sum(B))
    # initialize A: nonnegative + normalized
    A = np.maximum(0.0, noisy_B)
    renorm_to_mass(A, n_slice)

    # constraint set: total mass, by-row, by-column, by-cell
    Q_keep = [(), (0,), (1,), (0, 1)]
    history = []

    # —— Main loop —— #
    for iter_idx in range(1, T + 1):
        scores = []
        # 1) compute "error score" for each query
        for keep in Q_keep:
            marg_A = get_marginal(A, keep)
            marg_B = get_marginal(B, keep)
            # normalize by n_slice to prevent scale blow-up
            diff = np.sum(np.abs(marg_A - marg_B))
            size = 1 if not keep else np.prod([B.shape[k] for k in keep])
            s = (diff - size) / max(n_slice, 1.0)  # key: scaling
            scores.append(float(s))
        scores = np.asarray(scores, dtype=np.float64)

        # 2) stable softmax for sampling probabilities (instead of raw exp)
        #    exp_mech_scale corresponds to ε allocation per selection step; logic kept
        probs = softmax_stable(exp_mech_scale * scores)

        # guard: probabilities must sum to 1
        if not np.isfinite(probs.sum()) or abs(probs.sum() - 1.0) > 1e-8:
            probs = np.ones(len(Q_keep), dtype=np.float64) / len(Q_keep)

        # 3) sample query
        idx = int(np.random.choice(len(Q_keep), p=probs))
        keep = Q_keep[idx]

        # 4) observe marginal + Gaussian noise (DP)
        marg_B = get_marginal(B, keep)
        # Gaussian noise consistent with classic MWEM; already fairly stable numerically
        sigma_mwem = math.sqrt(2.0 * T * math.log(1.25 / delta)) / max(eps, 1e-12)
        noise_shape = () if np.isscalar(marg_B) else marg_B.shape
        m = marg_B + np.random.normal(0.0, sigma_mwem, size=noise_shape)

        history.append((keep, np.array(m, dtype=np.float64)))

        # 5) multiplicative update (numerically stable)
        update_A(A, keep, m, n_slice, iter_idx=iter_idx)

    # —— Projection Refinement (replay) —— #
    for refine_round in range(10):
        change = 0.0
        for it, (keep, m) in enumerate(history, start=1):
            marg_A = get_marginal(A, keep)
            update_A(A, keep, m, n_slice, iter_idx=it)
            change += float(np.sum(np.abs(m - marg_A)))
        if change < 1e-3:
            break

    data_rec_list.append(A)

data_rec = np.stack(data_rec_list, axis=2).astype(np.float64)
time2 = time.time()
print(f'Running time: {time2 - time1} ')

# =========================
# Evaluation
# =========================
# mae, re = get_eval_results(counts, data_rec, test_samples, sm=config['test']['sm'])
# logger.info(f'MAE: {mae}, RE: {re}')

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
# # Forecast evaluation (FMAE / sMAPE)
# fmae, fsmape, n_eff = gather_forecasting_results(
#     reco_grid=data_rec,
#     H=counts,
#     fcast_qs=fcast_qs,
#     fh=forecast_horizon
# )
# logger.info(f"[Forecast] fh={forecast_horizon} n={n_eff} sMAPE={fsmape:.4f}")

# =========================
# Save
# =========================
logger.info('Saving...')

save_path = os.path.join(
    config['train']['save_dir'],
    f"{config['datasets']['name']}/{config['datasets']['cell_size']}/eps_{eps}"
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

np.save(os.path.join(save_path, 'published_data_rec_MWEM.npy'), data_rec)
logger.info('data saved at'+ save_path + '/published_data_rec_MWEM.npy')
# logger.info('min_cell_count_re: {}'.format(min_cell_count_re))
# logger.info('min_hotspot_mae: {}'.format(min_hotspot_mae))
# logger.info('min_forecast_smape: {}'.format(min_forecast_smape))
logger.info("MWEM finished.")
