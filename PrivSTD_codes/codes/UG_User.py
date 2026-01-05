# userlevel_UG_main.py
# -*- coding: utf-8 -*-

"""
User-level UniformGrid (UG) main script (event-level -> user-level adaptation, aligned to  user-level)

Key alignment points (consistent with the earlier user-level PrivSDT version):
1) Read user-level raw data with 5 columns: [user] [check-in time] [lat] [lon] [location id]
   -> convert to db_full: [N,3] = [lon, lat, time_seconds] + user_ids
2) Truncate each user to k records to obtain Ds (db_s)
3) DP mechanism is applied only to Ds; evaluation aligns to the true dataset D (counts_true),
   and apply  global calibration gamma = |D| / |Ds|
4) User-level delta = 1 / (n_users^2)
5) User-level global sensitivity aligned to  as requested: Δ = k
   => Gaussian noise sigma = sqrt(2 ln(1.25/delta)) * k / eps
6) Include hotspot and forecasting evaluation (consistent with userlevel_PrivSDT_main)

Notes:
- UG coarse grid downsamples only spatial dimensions (m x m); the time dimension remains the same as the fine grid (T_).
- Reconstruction uses overlap-conserving vectorized einsum (your optimized version).
"""

import os
import math
import time
import numpy as np
import torch

from parse import config
from logger.logger import ConfigParser
from utils.dataset import get_counts
from utils.eval import *
from utils.results_stats import ResultStats


# =========================================================
# 1) User-level data reader (dedicated function; do not reuse read_dataset(config))
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
    No header in the first line.

    Returns:
      db_full: np.ndarray [N,3] -> [lon, lat, time_seconds]
      user_ids: np.ndarray [N]  -> int user id
      min_vals/max_vals: np.ndarray [3]
      n_records: int
      n_users: int
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
    """Truncate each user's contribution to k records (uniform sampling without replacement)."""
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
# 2) DP helpers (user-level: Δ = k)
# =========================================================
def gaussian_sigma(eps: float, delta: float, sensitivity: float) -> float:
    """Std of the (ε, δ)-DP Gaussian mechanism."""
    if eps <= 0:
        raise ValueError("epsilon must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0,1)")
    return math.sqrt(2.0 * math.log(1.25 / delta)) * (float(sensitivity) / float(eps))


# =========================================================
# 3) UG reconstruction (overlap-conserving + einsum)
# =========================================================
def ug_reconstruct_overlap_conserving(
    noisy_counts_ug: np.ndarray,
    lon_bins_fine: np.ndarray,
    lat_bins_fine: np.ndarray,
    lon_bins_ug: np.ndarray,
    lat_bins_ug: np.ndarray,
) -> np.ndarray:
    """
    Inputs:
      noisy_counts_ug: (m, m, T)
      lon_bins_fine: (H+1,)
      lat_bins_fine: (W+1,)
      lon_bins_ug: (m+1,)
      lat_bins_ug: (m+1,)
    Output:
      data_rec: (H, W, T)
    """
    H_ = len(lon_bins_fine) - 1
    W_ = len(lat_bins_fine) - 1

    # Longitude overlap ratios (H_, m)
    lon_start = lon_bins_fine[:-1, None]
    lon_end = lon_bins_fine[1:, None]
    ug_lon_s = lon_bins_ug[None, :-1]
    ug_lon_e = lon_bins_ug[None, 1:]

    overlap_lon = np.clip(np.minimum(lon_end, ug_lon_e) - np.maximum(lon_start, ug_lon_s), 0.0, None)
    ug_lon_w = np.maximum(ug_lon_e - ug_lon_s, 1e-12)
    lon_overlap = (overlap_lon / ug_lon_w).astype(np.float32)  # (H_, m)

    # Latitude overlap ratios (W_, m)
    lat_start = lat_bins_fine[:-1, None]
    lat_end = lat_bins_fine[1:, None]
    ug_lat_s = lat_bins_ug[None, :-1]
    ug_lat_e = lat_bins_ug[None, 1:]

    overlap_lat = np.clip(np.minimum(lat_end, ug_lat_e) - np.maximum(lat_start, ug_lat_s), 0.0, None)
    ug_lat_w = np.maximum(ug_lat_e - ug_lat_s, 1e-12)
    lat_overlap = (overlap_lat / ug_lat_w).astype(np.float32)  # (W_, m)

    # Mass-conserving reconstruction: einsum
    data_rec = np.einsum("ip,jq,pqt->ijt", lon_overlap, lat_overlap, noisy_counts_ug, optimize=True).astype(np.float32)
    return data_rec


# =========================================================
# =========================================================
def main():
    # Follow your current style with fixed params: only add k here; everything else comes from config
    data_dir = "./data/"
    name = config["datasets"]["name"]
    raw_filename = name + ".txt"

    k = int(config["datasets"]["truncate_size"])
    seed = 2024

    config_parser = ConfigParser(name="UG", save_dir="./")
    logger = config_parser.get_logger(config_parser.exper_name)

    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = config["train"]["gpu"]
    logger.info(f"config: {config}")
    logger.info(f"[UG UserLevel] k={k}")

    # ---------- read full D ----------
    db_full, user_ids, min_vals, max_vals, n_records, n_users = read_userlevel_txt_dataset(
        name=name,
        data_dir=data_dir,
        filename=raw_filename,
        delimiter=None,
        skip_bad_lines=True,
    )

    # user-level delta
    delta = 1.0 / (n_users ** 2)
    logger.info(f"[UG UserLevel] n_records={n_records}, n_users={n_users}, delta={delta:.3e}")
    logger.info(f"max_lon: {max_vals[0]}, min_lon: {min_vals[0]}, max_lat: {max_vals[1]}, min_lat: {min_vals[1]}")
    logger.info(f"time_interval: {(max_vals[2] - min_vals[2]) / 3600} hours")

    # ---------- true counts for evaluation (w.r.t. D) ----------
    counts_true, test_samples = get_counts(
        config,
        db_full,
        min_vals,
        max_vals,
        config["datasets"]["sample_size"],
        config["datasets"]["time_grid"],
    )
    counts_true = counts_true[:, :, :config["datasets"]["time_grid"]]
    H_, W_, T_ = counts_true.shape
    logger.info(f"[TrueCounts] shape={counts_true.shape} max={np.max(counts_true)} min={np.min(counts_true)} mean={np.mean(counts_true)}")

    # ---------- truncate to Ds ----------
    db_s, _user_ids_s = truncate_per_user(db_full, user_ids, k=k, seed=seed)
    n_s = int(db_s.shape[0])
    gamma = float(db_full.shape[0]) / float(db_s.shape[0])
    logger.info(f"[ Refinement] gamma = |D|/|Ds| = {db_full.shape[0]}/{db_s.shape[0]} = {gamma:.6f}")
    logger.info(f"[Sampled Ds] n_s={n_s} (<= n_users*k = {n_users*k})")

    # ---------- build fine-grid bin edges (must align with get_counts's uniform binning) ----------
    lon_bins = np.linspace(min_vals[0], max_vals[0], H_ + 1)
    lat_bins = np.linspace(min_vals[1], max_vals[1], W_ + 1)
    time_bins = np.linspace(min_vals[2], max_vals[2], T_ + 1)

    # ---------- UG coarse grid size m ----------
    eps = float(config["privacy"]["eps"])
    c = float(config["privacy"].get("c_for_ug", 10.0))
    # Under user-level DP, do not use n_records; use n_users instead (more reasonable and consistent with user-level delta)
    m = int(np.round(np.sqrt(n_users * eps / c)))
    m = max(1, m)
    logger.info(f"[UG UserLevel] m={m} (from c={c}, n_users={n_users})")

    lon_bins_ug = np.linspace(min_vals[0], max_vals[0], m + 1)
    lat_bins_ug = np.linspace(min_vals[1], max_vals[1], m + 1)

    # ---------- counts_ug computed from Ds ----------
    counts_ug = np.histogramdd(db_s, bins=(lon_bins_ug, lat_bins_ug, time_bins))[0].astype(np.float32)

    # ---------- user-level Gaussian noise with sensitivity Δ=k ----------
    sigma = gaussian_sigma(eps, delta, sensitivity=float(k))
    noisy_counts_ug = counts_ug + np.random.normal(0.0, sigma, counts_ug.shape).astype(np.float32)

    logger.info(f"[UG Release] sigma={sigma:.6f} (sens=k={k})")
    logger.info(f"noisy_counts_ug stats - max={np.max(noisy_counts_ug)}, min={np.min(noisy_counts_ug)}, mean={np.mean(noisy_counts_ug)}, sum={np.sum(noisy_counts_ug)}")

    # ---------- reconstruct fine grid from coarse noisy grid ----------
    logger.info("[UG] Starting overlap-conserving reconstruction (einsum)...")
    t1 = time.time()
    data_rec = ug_reconstruct_overlap_conserving(
        noisy_counts_ug=noisy_counts_ug,
        lon_bins_fine=lon_bins,
        lat_bins_fine=lat_bins,
        lon_bins_ug=lon_bins_ug,
        lat_bins_ug=lat_bins_ug,
    )
    if bool(config["test"].get("clip_nonneg", False)):
        np.maximum(data_rec, 0.0, out=data_rec)
    t2 = time.time()
    logger.info(f"[UG] Reconstruction time: {t2 - t1:.3f}s ; data_rec stats max={np.max(data_rec)} min={np.min(data_rec)} mean={np.mean(data_rec)} sum={np.sum(data_rec)}")

    # ----------  global calibration gamma (post-processing) ----------
    data_rec_cal = np.maximum(gamma * data_rec, 0.0)

    # ID baseline (for comparison only): add one-shot independent noise to fine-grid counts (here use Ds fine-grid counts_s) and then multiply by gamma
    counts_s_fine, _ = get_counts(
        config,
        db_s,
        min_vals,
        max_vals,
        config["datasets"]["sample_size"],
        config["datasets"]["time_grid"],
    )
    counts_s_fine = counts_s_fine[:, :, :config["datasets"]["time_grid"]]
    noisy_counts_fine = counts_s_fine + np.random.normal(0.0, sigma, counts_s_fine.shape).astype(np.float32)
    noisy_counts_fine_cal = np.maximum(gamma * noisy_counts_fine, 0.0)

    # =========================================================
    # Forecasting & Hotspot queries (style) — w.r.t counts_true
    # =========================================================
    max_val_vdr = float(config["datasets"].get("max_val", 10.0))
    rho_xy = max_val_vdr / float(H_)
    rho_t = max_val_vdr / float(T_)
    forecast_horizon = int(config["test"].get("forecast_horizon", 3))

    datafile_full = os.path.join(data_dir, f"{name}_userlevel_full.npy")
    np.save(datafile_full, db_full)

    data_filled_slices = compute_data_filled_slices_from_counts(counts_true)
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
        fh=forecast_horizon,
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
    min_cell_count_re = 999
    min_hotspot_mae = 999
    min_forecast_smape = 999
    # =========================================================
    # Evaluation (w.r.t counts_true) + save
    # =========================================================

    # mae, re = get_eval_results(counts_true, data_rec_cal, test_samples, sm=config["test"]["sm"])
    # logger.info(f"[UG UserLevel] MAE(gamma): {mae}, RE(gamma): {re}")

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
    #     reco_grid=data_rec_cal,
    #     H=counts_true,
    #     fcast_qs=fcast_qs,
    #     fh=forecast_horizon,
    # )
    # logger.info(f"[Forecast] fh={forecast_horizon} n={n_eff} sMAPE={fsmape:.4f}")
    # if min_cell_count_re > re[0]:
    #     min_cell_count_re = re[0]
    # if min_hotspot_mae > hot_res['mae'][config['test']['sm'][0]]:
    #     min_hotspot_mae = hot_res['mae'][config['test']['sm'][0]]
    # if min_forecast_smape > fsmape:
    #     min_forecast_smape = fsmape
    # Save (consistent with userlevel_PrivSDT_main style: save both uncalibrated and calibrated outputs)
    save_path = config["train"]["save_dir"] + "/{}/{}/eps_{}_userlevel_k{}".format(
        config["datasets"]["name"], config["datasets"]["sample_size"], eps, k
    )
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path + "/published_data_rec_UG_userlevel.npy", data_rec)
    np.save(save_path + "/published_data_rec_UG_userlevel_gamma.npy", data_rec_cal)
    logger.info("Saved:")
    logger.info(save_path + "/published_data_rec_UG_userlevel.npy")
    logger.info(save_path + "/published_data_rec_UG_userlevel_gamma.npy")

    # logger.info('min_cell_count_re: {}'.format(min_cell_count_re))
    # logger.info('min_hotspot_mae: {}'.format(min_hotspot_mae))
    # logger.info('min_forecast_smape: {}'.format(min_forecast_smape))
    logger.info("UG (user-level, overlap-conserving, gamma-calibrated) finished.")


if __name__ == "__main__":
    main()
