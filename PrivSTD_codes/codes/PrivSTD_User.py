# userlevel_PrivSDT_main.py
# -*- coding: utf-8 -*-

"""
User-level PrivSDT main script (PrivSDT event-level -> user-level adaptation)

Key changes (user-level):
1) Read user-level raw file with 5 cols: [user] [check-in time] [lat] [lon] [location id]
   -> convert to db array of shape [N,3] in the SAME format expected by your pipeline: [lon, lat, time_seconds]
2) Clip each user's contribution to k (truncate parameter).
3) Global sensitivity for histogram release becomes Δ = k (user-level), so Gaussian noise sigma *= k.
4) delta is set based on number of users: delta = 1 / (n_users^2)
5) statistical refinement : global gamma = |D|/|Ds| applied ONLY as post-processing
   for evaluation/queries (and optionally saved outputs).
"""

import os
import math
import time
import contextlib
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==== Project modules (unchanged) ====
from parse import config
from logger.logger import ConfigParser
from utils.dataset import get_counts, MVDataset, pad_data
from model.DnCNN import DnCNN
from utils.eval import *  # noqa: F401,F403
from utils.results_stats import ResultStats


# ---------------------------
# Training/inference speed knobs (no algorithm change; only numeric precision / layout / parallelism)
# ---------------------------
TRAIN_ACCEL = {
    "tf32": True,
    "channels_last": True,
    "amp_bf16": True,
    "torch_compile": False,
}


# =========================================================
# 1) User-level data reader (dedicated function)
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
# 2) DCT utilities & Sigma-aware noise/loss
# =========================================================
def _dct_1d_ortho_matrix(N: int) -> np.ndarray:
    n = np.arange(N, dtype=float)
    C = np.zeros((N, N), dtype=float)
    C[:, 0] = 1.0 / math.sqrt(N)
    for k in range(1, N):
        C[:, k] = math.sqrt(2.0 / N) * np.cos((math.pi * (2.0 * n + 1.0) * k) / (2.0 * N))
    return C


def _dct_low_torch(H: int, U: int, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    U = min(U, H)
    C = torch.from_numpy(_dct_1d_ortho_matrix(H)).to(device=device, dtype=dtype)
    return C[:, :U].contiguous()


def _canonize_dct_prefix(prefix: torch.Tensor, expected_rows: int, name: str) -> torch.Tensor:
    if prefix.dim() != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(prefix.shape)}")
    H0, U0 = int(prefix.shape[0]), int(prefix.shape[1])
    if H0 == expected_rows:
        return prefix
    if U0 == expected_rows:
        return prefix.t().contiguous()
    raise ValueError(
        f"{name} has incompatible shape {tuple(prefix.shape)}; "
        f"neither row nor col equals expected_rows={expected_rows}."
    )


def sample_corr_noise_from_coeff(
    B: int,
    Cx_low: torch.Tensor,
    Cy_low: torch.Tensor,
    sigma: float,
    expected_H: int | None = None,
    expected_W: int | None = None,
) -> torch.Tensor:
    if expected_H is None:
        expected_H = int(Cx_low.shape[0]) if Cx_low.shape[0] >= Cx_low.shape[1] else int(Cx_low.shape[1])
    if expected_W is None:
        expected_W = int(Cy_low.shape[0]) if Cy_low.shape[0] >= Cy_low.shape[1] else int(Cy_low.shape[1])

    Cx_low_c = _canonize_dct_prefix(Cx_low, expected_rows=expected_H, name="Cx_low")
    Cy_low_c = _canonize_dct_prefix(Cy_low, expected_rows=expected_W, name="Cy_low")

    device = Cx_low_c.device
    dtype = Cx_low_c.dtype
    U = int(Cx_low_c.shape[1])
    V = int(Cy_low_c.shape[1])

    Xi = torch.randn(B, U, V, device=device, dtype=dtype) * float(sigma)
    tmp = torch.matmul(Cx_low_c, Xi)          # (B,H,V)
    Z = torch.matmul(tmp, Cy_low_c.t())       # (B,H,W)
    Z = Z.unsqueeze(1).contiguous()           # (B,1,H,W)

    if (Z.shape[-2], Z.shape[-1]) != (expected_H, expected_W):
        raise RuntimeError(f"Noise dims {tuple(Z.shape[-2:])} != expected {(expected_H, expected_W)}")
    return Z


def gls_coef_loss(
    model: nn.Module,
    x: torch.Tensor,
    Z_pos: torch.Tensor,
    Z_neg: torch.Tensor,
    Cx_low: torch.Tensor,
    Cy_low: torch.Tensor,
    sigma: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    if x.shape != Z_pos.shape or x.shape != Z_neg.shape:
        raise RuntimeError(
            f"[gls_coef_loss] shape mismatch: x{tuple(x.shape)} "
            f"Z_pos{tuple(Z_pos.shape)} Z_neg{tuple(Z_neg.shape)}"
        )

    y_pos = x + alpha * Z_pos
    y_neg = x + Z_neg
    out = model(y_pos)                # [B,1,H,W]
    R = (out - y_neg).squeeze(1)      # [B,H,W]

    R1 = torch.matmul(R, Cy_low)      # [B,H,V]
    R_theta = torch.matmul(Cx_low.t(), R1)  # [B,U,V]

    loss_per = (R_theta ** 2).flatten(1).mean(dim=1) / (float(sigma) ** 2)
    return loss_per.mean()


@torch.no_grad()
def _flatten_norm2_and_diffnorm2(grads_a, grads_b, params):
    g_bar_list = []
    device = params[0].device
    g_norm2 = torch.zeros((), device=device)
    v = torch.zeros((), device=device)
    for ga, gb, p in zip(grads_a, grads_b, params):
        if ga is None:
            ga = torch.zeros_like(p)
        if gb is None:
            gb = torch.zeros_like(p)
        gbar = 0.5 * (ga + gb)
        g_bar_list.append(gbar)
        g_norm2 += (gbar * gbar).sum()
        diff = ga - gb
        v += 0.25 * (diff * diff).sum()
    return g_bar_list, g_norm2, v


# =========================================================
# 3) DP release (user-level: sigma *= k)
# =========================================================
def _gaussian_sigma(eps: float, delta: float, sens: float) -> float:
    return math.sqrt(2.0 * math.log(1.25 / delta)) * float(sens) / float(eps)


def dp_release_DCT_dct_lowfreq_userlevel(
    counts: np.ndarray, eps: float, delta: float, sens_k: int, U: int = 8, V: int = 8
):
    """User-level DCT: add noise to low-frequency DCT coefficients with sigma = sigma_base * k."""
    H, W, T = counts.shape
    U = min(U, H)
    V = min(V, W)

    Cx = _dct_1d_ortho_matrix(H)
    Cy = _dct_1d_ortho_matrix(W)

    sum_Cx2 = (Cx[:, :U] ** 2).sum(axis=1)
    sum_Cy2 = (Cy[:, :V] ** 2).sum(axis=1)

    sigma = _gaussian_sigma(eps, delta, sens=sens_k)

    Cx_low = Cx[:, :U]
    Cy_low = Cy[:, :V]

    noisy_counts = np.zeros_like(counts, dtype=float)
    Theta_list = []
    for t in range(T):
        X = counts[:, :, t].astype(float)
        Theta = (Cx_low.T @ X) @ Cy_low
        Theta_noisy = Theta + np.random.normal(0.0, sigma, size=Theta.shape)
        Theta_list.append(Theta_noisy)
        X_tilde = (Cx_low @ Theta_noisy) @ (Cy_low.T)
        X_tilde[X_tilde < 0] = 0.0
        noisy_counts[:, :, t] = X_tilde

    meta = {
        "sigma": float(sigma),
        "U": U,
        "V": V,
        "sum_Cx2": sum_Cx2,
        "sum_Cy2": sum_Cy2,
        "sens_k": int(sens_k),
    }
    return noisy_counts, meta, Theta_list


def dp_release_DCT_dct_lowfreq_ref_then_crop_userlevel(
    counts: np.ndarray, eps: float, delta: float, sens_k: int, U_sel: int, V_sel: int, U_ref: int, V_ref: int
):
    H, W, T = counts.shape
    U_ref = min(U_ref, H)
    V_ref = min(V_ref, W)
    U_sel = min(U_sel, H)
    V_sel = min(V_sel, W)

    _, meta_ref, Theta_noisy_list = dp_release_DCT_dct_lowfreq_userlevel(
        counts, eps, delta, sens_k=sens_k, U=U_ref, V=V_ref
    )
    sigma_ref = float(meta_ref["sigma"])

    Cx = _dct_1d_ortho_matrix(H)
    Cy = _dct_1d_ortho_matrix(W)
    Cx_low_sel = Cx[:, :U_sel]
    Cy_low_sel = Cy[:, :V_sel]
    sum_Cx2_sel = (Cx[:, :U_sel] ** 2).sum(axis=1)
    sum_Cy2_sel = (Cy[:, :V_sel] ** 2).sum(axis=1)

    noisy_counts_final = np.zeros_like(counts, dtype=float)
    for t in range(T):
        Theta_full = Theta_noisy_list[t]
        Theta_crop = Theta_full[:U_sel, :V_sel]
        X_tilde = (Cx_low_sel @ Theta_crop) @ (Cy_low_sel.T)
        X_tilde[X_tilde < 0] = 0.0
        noisy_counts_final[:, :, t] = X_tilde

    meta_final = {
        "sigma": sigma_ref,
        "U": U_sel,
        "V": V_sel,
        "sum_Cx2": sum_Cx2_sel,
        "sum_Cy2": sum_Cy2_sel,
        "sigma_ref": sigma_ref,
        "U_ref": U_ref,
        "V_ref": V_ref,
        "sens_k": int(sens_k),
    }
    return noisy_counts_final, meta_final, Theta_noisy_list


def build_varmap_DCT_from_meta(meta, H, W, T):
    sigma = float(meta["sigma"])
    sum_Cx2 = meta["sum_Cx2"]
    sum_Cy2 = meta["sum_Cy2"]
    var2 = (sigma ** 2) * np.outer(sum_Cx2, sum_Cy2)
    return np.repeat(var2[..., None], T, axis=2)


def dp_release_baseline_iid_userlevel(counts: np.ndarray, eps: float, delta: float, sens_k: int):
    sigma = _gaussian_sigma(eps, delta, sens=sens_k)
    noise = np.random.normal(0, sigma, counts.shape)
    noisy_counts = counts + noise
    meta = {"sigma": float(sigma), "sens_k": int(sens_k)}
    return noisy_counts, meta


def build_varmap_baseline(counts_shape, sigma):
    H, W, T = counts_shape
    return np.full((H, W, T), sigma ** 2, dtype=np.float64)


# =========================================================
# 4) UV selection (BH)
# =========================================================
def select_uv_bh_chi2(Theta_noisy_list, sigma_ref, prefer_equal=True, Umax=None, Vmax=None, q=0.10):
    import scipy.stats as st
    T = len(Theta_noisy_list)
    Umax = Umax or Theta_noisy_list[0].shape[0]
    Vmax = Vmax or Theta_noisy_list[0].shape[1]
    Y2 = np.zeros((Umax, Vmax), dtype=np.float64)
    for Th in Theta_noisy_list:
        Th = Th[:Umax, :Vmax].astype(np.float64)
        Y2 += Th * Th
    S = Y2 / (sigma_ref ** 2)
    p = 1.0 - st.chi2.cdf(S, df=T)
    flat = p.ravel()
    idx = np.argsort(flat)
    m = flat.size
    thresh = (np.arange(1, m + 1) / m) * q
    pass_mask = flat[idx] <= thresh
    if not pass_mask.any():
        return 8, 8, p
    kmax = np.where(pass_mask)[0].max()
    sig_ids = idx[:kmax + 1]
    us, vs = np.unravel_index(sig_ids, (Umax, Vmax))
    Urec = int(us.max() + 1)
    Vrec = int(vs.max() + 1)
    if prefer_equal:
        kk = min(Urec, Vrec)
        return kk, kk, p
    return Urec, Vrec, p


def DCT_select_UV_userlevel(
    counts: np.ndarray,
    eps: float,
    delta: float,
    sens_k: int,
    prefer_equal: bool = False,
    Umax_cap: int | None = None,
    seed: int = 2024,
    method: str = "bh",
    bh_q: float = 0.01,
):
    H, W, T = counts.shape
    Umax = min(H, Umax_cap or H)
    Vmax = min(W, Umax_cap or W)

    _, meta_ref, Theta_noisy_list = dp_release_DCT_dct_lowfreq_userlevel(
        counts, eps, delta, sens_k=sens_k, U=Umax, V=Vmax
    )
    sigma_ref = float(meta_ref["sigma"])

    if method == "bh":
        Urec, Vrec, _grid = select_uv_bh_chi2(
            Theta_noisy_list, sigma_ref, prefer_equal, Umax, Vmax, q=bh_q
        )
    else:
        raise ValueError(f"Unknown UV selector: {method}")

    noisy_counts_final, meta_final, _ = dp_release_DCT_dct_lowfreq_ref_then_crop_userlevel(
        counts, eps, delta, sens_k=sens_k, U_sel=Urec, V_sel=Vrec, U_ref=Umax, V_ref=Vmax
    )

    rep = {
        "Umax": int(Umax),
        "Vmax": int(Vmax),
        "Urec": int(Urec),
        "Vrec": int(Vrec),
        "sigma_ref": float(sigma_ref),
        "eps": float(eps),
        "method": method,
        "sens_k": int(sens_k),
    }
    return noisy_counts_final, meta_final, (Urec, Vrec), rep, Theta_noisy_list


# =========================================================
# 5) Main pipeline (user-level)
# =========================================================
def main():
    data_dir = "./data/"
    raw_filename = config["datasets"]["name"] + ".txt"
    k = int(config["datasets"]["truncate_size"])
    seed = 2024

    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.benchmark = True
    if TRAIN_ACCEL["tf32"]:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    config_parser = ConfigParser(name="PrivSTD", save_dir="./")
    logger = config_parser.get_logger(config_parser.exper_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = config["train"]["gpu"]
    logger.info(f"config: {config}")
    logger.info(f"[UserLevel] truncation k = {k}")

    name = config["datasets"]["name"]
    db_full, user_ids, min_vals, max_vals, n_records, n_users = read_userlevel_txt_dataset(
        name=name,
        data_dir=data_dir,
        filename=raw_filename,
        delimiter=None,
        skip_bad_lines=True,
    )

    delta = 1.0 / (n_users ** 2)
    logger.info(f"[UserLevel] n_records={n_records}, n_users={n_users}, delta={delta:.3e}")
    logger.info(f"max_lon: {max_vals[0]}, min_lon: {min_vals[0]}, max_lat: {max_vals[1]}, min_lat: {min_vals[1]}")
    logger.info(f"time_interval: {(max_vals[2] - min_vals[2]) / 3600} hours")

    counts_true, test_samples = get_counts(
        config,
        db_full,
        min_vals,
        max_vals,
        config["datasets"]["cell_size"],
        config["datasets"]["time_grid"],
    )
    counts_true = counts_true[:, :, :config["datasets"]["time_grid"]]
    logger.info(
        f"[TrueCounts] shape={counts_true.shape} max={np.max(counts_true)} "
        f"min={np.min(counts_true)} mean={np.mean(counts_true)}"
    )

    db_s, user_ids_s = truncate_per_user(db_full, user_ids, k=k, seed=seed)
    n_s = int(db_s.shape[0])
    gamma = float(db_full.shape[0]) / float(db_s.shape[0])
    logger.info(f"[Refinement] gamma = |D|/|Ds| = {db_full.shape[0]}/{db_s.shape[0]} = {gamma:.6f}")
    logger.info(f"[Sampled Ds] n_s={n_s} (<= n_users*k = {n_users*k})")

    counts_s, _ = get_counts(
        config,
        db_s,
        min_vals,
        max_vals,
        config["datasets"]["cell_size"],
        config["datasets"]["time_grid"],
    )
    counts_s = counts_s[:, :, :config["datasets"]["time_grid"]]
    logger.info(
        f"[SampledCounts] shape={counts_s.shape} max={np.max(counts_s)} "
        f"min={np.min(counts_s)} mean={np.mean(counts_s)}"
    )

    eps = float(config["privacy"]["eps"])
    logger.info(f"Privacy level: user-level, sens=Δ=k={k})")

    H, W, T = counts_s.shape
    results_stats = ResultStats(config)

    t0 = time.time()
    noisy_counts, dp_meta, (Urec, Vrec), rep, Theta_noisy_list = DCT_select_UV_userlevel(
        counts=counts_s,
        eps=eps,
        delta=delta,
        sens_k=k,
        prefer_equal=bool(config["privacy"].get("prefer_equal", False)),
        Umax_cap=min(H, W),
        seed=seed,
        method=str(config["privacy"].get("uv_selector", "bh")),
        bh_q=float(config["privacy"].get("bh_q", 0.1)),
    )
    logger.info(
        f'[DCT Select] ({rep["method"]}) U*={Urec}, V*={Vrec} ; '
        f'sigma_ref={dp_meta["sigma"]:.6f} ; sens_k={k}'
    )
    var_cube = build_varmap_DCT_from_meta(dp_meta, H, W, T)
    t1 = time.time()
    logger.info(f"[DCT Release] time: {t1 - t0:.3f} s")


    max_val_vdr = float(config["datasets"].get("max_val", 10.0))
    rho_xy = max_val_vdr / float(counts_true.shape[0])
    rho_t = max_val_vdr / float(counts_true.shape[2])
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
        test_size=config["datasets"]["cell_size"],
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
    Hc_slow_payload = {}
    Hress_slow_payload = {}
    H_slow_qs = {}
    for _hot in hot_levels:
        _qs, _Hc, _Hress = get_hotspot_queries(_hot, counts_true, loc_ijk_arr, limit=500, radius=50)
        Hc_slow_payload[_hot] = np.asarray(_Hc, dtype=int)
        Hress_slow_payload[_hot] = np.asarray(_Hress, dtype=float).reshape(-1, 1)
        H_slow_qs[_hot] = np.asarray(_qs, dtype=int)
    logger.info(f'Hotspot queries completed')

    os.makedirs("./data/" + name, exist_ok=True)
    np.save(f"./data/{name}/{eps}_train_h_userlevel.npy", noisy_counts)
    np.save(f"./data/{name}/counts_{config['datasets']['cell_size']}_userlevel.npy", counts_true)
    np.save(f"./data/{name}/test_samples_{config['datasets']['cell_size']}_userlevel.npy", test_samples)

    forecast_data = {
        "queries": fcast_qs,
        "h_mae_ref": _h_mae_ref,
        "h_mape_ref": _h_mape_ref,
        "forecast_horizon": forecast_horizon,
        "rho_xy": rho_xy,
        "rho_t": rho_t,
        "gamma": gamma,
        "k": k,
    }
    np.save(
        f"./data/{name}/forecast_queries_{config['datasets']['cell_size']}_userlevel.npy",
        forecast_data,
    )

    hotspot_data = {
        "hot_levels": hot_levels,
        "Hc_slow_payload": Hc_slow_payload,
        "Hress_slow_payload": Hress_slow_payload,
        "H_slow_qs": H_slow_qs,
        "loc_ijk_arr": loc_ijk_arr,
        "gamma": gamma,
        "k": k,
    }
    np.save(
        f"./data/{name}/hotspot_queries_{config['datasets']['cell_size']}_userlevel.npy",
        hotspot_data,
    )

    lr = float(config["train"]["lr"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DnCNN(
        config,
        channels=1,
        num_of_layers=17,
        dropout_p=0.1,
        dropout_start=4,
        dropout_end=14,
        spatial=True,
    ).to(device)

    if TRAIN_ACCEL["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    if TRAIN_ACCEL["torch_compile"] and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune", fullgraph=False, dynamic=True)
            logger.info("[torch.compile] enabled")
        except Exception as e:
            logger.info(f"[torch.compile] disabled due to: {e}")

    optim = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    epochs = int(config["train"]["epochs"])
    alpha = float(config["train"].get("alpha", 1.0))

    sigma_dp = float(dp_meta["sigma"])
    U_sel = int(dp_meta.get("U", min(H, W)))
    V_sel = int(dp_meta.get("V", min(H, W)))
    Cx_full_t = _dct_low_torch(H, U_sel, device=device, dtype=torch.float32)
    Cy_full_t = _dct_low_torch(W, V_sel, device=device, dtype=torch.float32)

    train_data = np.transpose(noisy_counts, (2, 0, 1))[:, None, ...]
    var_train = np.transpose(var_cube, (2, 0, 1))[:, None, ...]

    train_dataset = MVDataset(train_data, img_size=config["net"]["img_size"], is_train=True)
    var_dataset = MVDataset(var_train, img_size=config["net"]["img_size"], is_train=True)

    num_workers = int(config["train"].get("num_workers", 8))
    dlr_common = dict(
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        dlr_common.update(dict(persistent_workers=True, prefetch_factor=4))

    train_loader = DataLoader(train_dataset, **dlr_common)
    var_loader = DataLoader(var_dataset, **dlr_common)

    use_amp = TRAIN_ACCEL["amp_bf16"] and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()

    _dct_cache = {}

    def _get_dct_prefix(Hp: int, Wp: int):
        key = (Hp, Wp)
        if key not in _dct_cache:
            Cx_t = _dct_low_torch(Hp, min(U_sel, Hp), device=device, dtype=torch.float32)
            Cy_t = _dct_low_torch(Wp, min(V_sel, Wp), device=device, dtype=torch.float32)
            _dct_cache[key] = (Cx_t, Cy_t, Cx_t.t().contiguous(), Cy_t.t().contiguous())
        return _dct_cache[key]

    g2_ema = None
    v_ema = None
    ema_beta = float(config["train"].get("ema_beta", 0.9))
    min_cell_count_re = 999
    min_hotspot_mae = 999
    min_forecast_smape = 999
    for epoch in range(epochs):
        model.train()
        loss_epoch = []
        logger.info(f"<---- epoch {epoch} ---->")
        t3 = time.time()

        for (x), (v) in zip(train_loader, var_loader):
            if len(x.shape) > 4:
                x = x.squeeze(0)
            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            if TRAIN_ACCEL["channels_last"]:
                x = x.to(memory_format=torch.channels_last)

            B = x.shape[0]
            Hp, Wp = int(x.shape[-2]), int(x.shape[-1])

            Cx_patch_t, Cy_patch_t, _, _ = _get_dct_prefix(Hp, Wp)

            with amp_ctx:
                Z_all = sample_corr_noise_from_coeff(
                    4 * B, Cx_patch_t, Cy_patch_t, sigma_dp, expected_H=Hp, expected_W=Wp
                )
            Zpos_a, Zneg_a, Zpos_b, Zneg_b = Z_all.chunk(4, dim=0)

            optim.zero_grad(set_to_none=True)

            with amp_ctx:
                L_a = gls_coef_loss(model, x, Zpos_a, Zneg_a, Cx_patch_t, Cy_patch_t, sigma_dp, alpha=alpha)
            grads_a = torch.autograd.grad(
                L_a, [p for p in model.parameters() if p.requires_grad],
                retain_graph=True, create_graph=False, allow_unused=True
            )

            with amp_ctx:
                L_b = gls_coef_loss(model, x, Zpos_b, Zneg_b, Cx_patch_t, Cy_patch_t, sigma_dp, alpha=alpha)
            grads_b = torch.autograd.grad(
                L_b, [p for p in model.parameters() if p.requires_grad],
                retain_graph=False, create_graph=False, allow_unused=True
            )

            params_req = [p for p in model.parameters() if p.requires_grad]

            with torch.no_grad():
                L_hat = 0.5 * (L_a + L_b)
                g_bar_list, g_norm2, vstat = _flatten_norm2_and_diffnorm2(grads_a, grads_b, params_req)

                eps_small = 1e-12
                beta_sps = 1.0
                eta_min, eta_max = 1e-6, 1e-1
                L_star = 0.0

                g2 = g_norm2.detach()
                vv = vstat.detach()
                if g2_ema is None:
                    g2_ema = g2
                    v_ema = vv
                else:
                    g2_ema = ema_beta * g2_ema + (1.0 - ema_beta) * g2
                    v_ema = ema_beta * v_ema + (1.0 - ema_beta) * vv

                denom = g2_ema + beta_sps * v_ema + eps_small
                eta = torch.clamp((L_hat - L_star) / denom, min=eta_min, max=eta_max).item()

                for group in optim.param_groups:
                    group["lr"] = eta
                for p, g in zip(params_req, g_bar_list):
                    p.grad = g.detach().clone()

                optim.step()

            loss_epoch.append(L_hat.item())

        logger.info(f"epoch {epoch}, loss: {np.mean(loss_epoch)}")
        results_stats.add_loss(np.mean(loss_epoch))
        t4 = time.time()
        results_stats.add_train_time(t4 - t3)

        if epoch % int(config["train"]["eval_freq"]) == 0:
            model.eval()
            data_rec = []
            mc_aver = int(config["test"].get("mc_aver", 8))

            with torch.no_grad():
                _, _, Nn = noisy_counts.shape
                for i in tqdm(range(Nn)):
                    x_np = noisy_counts[:, :, i]
                    x = torch.from_numpy(x_np).float().to(device)
                    orig_shape = x.shape
                    x_pad = pad_data(x, window_size=config["net"]["window_size"])

                    m = mc_aver
                    with amp_ctx:
                        z_full = sample_corr_noise_from_coeff(
                            m, Cx_full_t, Cy_full_t, sigma_dp, expected_H=H, expected_W=W
                        )

                    z_pad_list = []
                    for kk in range(m):
                        z_pad_k = pad_data(z_full[kk, 0], window_size=config["net"]["window_size"])
                        z_pad_list.append(z_pad_k)
                    z_pad = torch.stack(z_pad_list, dim=0)[:, None, ...]

                    p_x = (x_pad.unsqueeze(0).unsqueeze(0) + alpha * z_pad)
                    if TRAIN_ACCEL["channels_last"]:
                        p_x = p_x.to(memory_format=torch.channels_last)

                    with amp_ctx:
                        x_rec = model(p_x).mean(dim=0, keepdim=True)

                    x_rec = x_rec[:, :, :orig_shape[0], :orig_shape[1]]
                    x_rec_np = x_rec.squeeze(0).squeeze(0).detach().cpu().numpy()
                    x_rec_np = np.maximum(x_rec_np, 0.0)
                    data_rec.append(x_rec_np[..., None])

            data_rec = np.concatenate(data_rec, axis=-1)

            data_rec_cal = np.maximum(gamma * data_rec, 0.0)
            noisy_cal = np.maximum(gamma * noisy_counts, 0.0)

            # mae, re = get_eval_results(counts_true, data_rec_cal, test_samples, sm=config["test"]["sm"])
            # logger.info(f"epoch {epoch}, MAE(gamma): {mae}, RE(gamma): {re}")

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
            # logger.info(f"[Forecast] fh={forecast_horizon} n={n_eff} FMAE={fmae:.4f} sMAPE={fsmape:.4f}")


            if epoch %10 == 0 :
                logger.info("Saving model...")


                save_path = config["train"]["save_dir"] + "/{}/{}/eps_{}_userlevel_k{}".format(
                    config["datasets"]["name"], config["datasets"]["cell_size"], eps, k
                )
                # res_str = 'Epoch:\t{}\tMAE:\t{}\tRE:\t{}'.format(
                #     epoch, mae, re
                # )
                os.makedirs(save_path, exist_ok=True)

                np.save(save_path + "/published_data_rec_PrivSDT_userlevel.npy", data_rec)
                np.save(save_path + "/published_data_rec_PrivSDT_userlevel_gamma.npy", data_rec_cal)
                np.save(save_path + "/published_noisy_counts_userlevel_gamma.npy", noisy_cal)

                logger.info("data_rec saved at " + save_path + "/published_data_rec_PrivSDT_userlevel.npy")
                logger.info("data_rec_cal saved at " + save_path + "/published_data_rec_PrivSDT_userlevel_gamma.npy")

                # model.save_model(res_str, save_path)
                # if min_cell_count_re > re[0]:
                #     min_cell_count_re = re[0]
                # if min_hotspot_mae > hot_res['mae'][config['test']['sm'][0]]:
                #     min_hotspot_mae = hot_res['mae'][config['test']['sm'][0]]
                # if min_forecast_smape > fsmape:
                #     min_forecast_smape = fsmape
                

    # logger.info('min_cell_count_re: {}'.format(min_cell_count_re))
    # logger.info('min_hotspot_mae: {}'.format(min_hotspot_mae))
    # logger.info('min_forecast_smape: {}'.format(min_forecast_smape))
    logger.info('PrivSTD Completed')

if __name__ == "__main__":
    main()
