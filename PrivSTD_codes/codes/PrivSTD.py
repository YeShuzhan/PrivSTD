"""
Completely revised script with Sigma-aware R2R + coefficient-domain GLS (dimension-safe) + perf tuned:
- Train/test noise is sampled in DCT low-frequency coefficient domain and mapped to pixel domain to match DP covariance.
- Loss is coefficient-domain GLS: Frobenius norm on (U,V) block divided by sigma^2.
- Keeps your DCT release, var_map construction (for compatibility), SPS, logging, and evaluation metrics intact.
"""
import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
import time
import contextlib

# ==== Your Engineering Modules ====
from parse import config
from logger.logger import ConfigParser
from utils.dataset import read_dataset, get_counts, MVDataset, pad_data
from model.DnCNN import DnCNN
from utils.eval import *
from utils.results_stats import ResultStats

TRAIN_ACCEL = {
    "tf32": True,
    "channels_last": True,
    "amp_bf16": True,
    "torch_compile": False,
}

def _dct_1d_ortho_matrix(N: int) -> np.ndarray:
    n = np.arange(N, dtype=float)
    C = np.zeros((N, N), dtype=float)
    C[:, 0] = 1.0 / math.sqrt(N)
    for k in range(1, N):
        C[:, k] = math.sqrt(2.0 / N) * np.cos((math.pi * (2.0 * n + 1.0) * k) / (2.0 * N))
    return C


# =========================================================
# Sigma-aware Correlated Noise & Coefficient-domain GLS (Core + Dimension Safe)
# =========================================================

def _dct_low_torch(H: int, U: int, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    """Returns [H, U] orthonormal DCT-II prefix matrix (Torch Tensor)."""
    U = min(U, H)
    C = torch.from_numpy(_dct_1d_ortho_matrix(H)).to(device=device, dtype=dtype)
    return C[:, :U].contiguous()

def _canonize_dct_prefix(prefix: torch.Tensor, expected_rows: int, name: str) -> torch.Tensor:
    """
    Ensures prefix has expected_rows; if not but cols equal expected_rows, transpose it.
    Otherwise raises error about dimension mismatch. Returns tensor shape must be [expected_rows, U].
    """
    if prefix.dim() != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(prefix.shape)}")
    H0, U0 = int(prefix.shape[0]), int(prefix.shape[1])
    if H0 == expected_rows:
        return prefix
    if U0 == expected_rows:
        return prefix.t().contiguous()
    raise ValueError(f"{name} has incompatible shape {tuple(prefix.shape)}; "
                     f"neither row nor col equals expected_rows={expected_rows}.")

def sample_corr_noise_from_coeff(
    B: int,
    Cx_low: torch.Tensor,  # [H, U] or [U, H] (will be canonicalized)
    Cy_low: torch.Tensor,  # [W, V] or [V, W] (will be canonicalized)
    sigma: float,
    expected_H: int | None = None,
    expected_W: int | None = None,
) -> torch.Tensor:
    """
    Samples i.i.d Gaussian from coefficient domain and maps to pixel domain;
    Returns [B,1,H,W] correlated Gaussian noise.
    Allows Cx_low/Cy_low to be transposed versions; internally canonicalized (with assertions).
    If expected_H/W are given, verifies output spatial dimensions match expectations.
    Uses two matmuls: tmp = Cx @ Xi ; Z = tmp @ Cy^T
    """
    # Canonicalize prefix matrix orientation
    if expected_H is None:
        expected_H = int(Cx_low.shape[0]) if Cx_low.shape[0] >= Cx_low.shape[1] else int(Cx_low.shape[1])
    if expected_W is None:
        expected_W = int(Cy_low.shape[0]) if Cy_low.shape[0] >= Cy_low.shape[1] else int(Cy_low.shape[1])

    Cx_low_c = _canonize_dct_prefix(Cx_low, expected_rows=expected_H, name="Cx_low")
    Cy_low_c = _canonize_dct_prefix(Cy_low, expected_rows=expected_W, name="Cy_low")
    device = Cx_low_c.device
    dtype = Cx_low_c.dtype
    H = int(Cx_low_c.shape[0])
    W = int(Cy_low_c.shape[0])
    U = int(Cx_low_c.shape[1])
    V = int(Cy_low_c.shape[1])

    # Coefficient domain noise [B,U,V]
    Xi = torch.randn(B, U, V, device=device, dtype=dtype) * float(sigma)

    # Z = Cx_low @ Xi @ Cy_low^T -> [B,H,W]
    # (H,U) @ (B,U,V) -> (B,H,V)
    tmp = torch.matmul(Cx_low_c, Xi)
    # (B,H,V) @ (V,W) -> (B,H,W)
    Z = torch.matmul(tmp, Cy_low_c.t())

    # Output [B,1,H,W]
    Z = Z.unsqueeze(1).contiguous()

    if Z.shape[-2] != H or Z.shape[-1] != W:
        raise RuntimeError(f"Noise shape mismatch: got {tuple(Z.shape)}, expected H={H}, W={W}.")
    if (expected_H is not None and expected_W is not None) and \
       ((Z.shape[-2] != expected_H) or (Z.shape[-1] != expected_W)):
        raise RuntimeError(f"Noise final dims {tuple(Z.shape[-2:])} != expected {(expected_H, expected_W)}")
    return Z

def gls_coef_loss(model: nn.Module,
                  x: torch.Tensor,      # [B,1,H,W]
                  Z_pos: torch.Tensor,  # [B,1,H,W], Coeff -> Pixel correlated noise
                  Z_neg: torch.Tensor,  # [B,1,H,W], Independent correlated noise
                  Cx_low: torch.Tensor, # [H,U]
                  Cy_low: torch.Tensor, # [W,V]
                  sigma: float,
                  alpha: float = 1.0) -> torch.Tensor:
    """
    Sigma-aware R2R (independent re-pollution) + Coefficient-domain GLS supervision equivalence:
    L = E[ || (Cx^T ( f(x + alpha*Z_pos) - (x + Z_neg) ) Cy)_{0:U,0:V} ||_F^2 ] / sigma^2
    Returns scalar loss. Uses two matmuls for coefficient domain projection, numerically equivalent.
    """
    # Shape consistency
    if x.shape != Z_pos.shape or x.shape != Z_neg.shape:
        raise RuntimeError(f"[gls_coef_loss] shape mismatch: x{tuple(x.shape)} "
                           f"Z_pos{tuple(Z_pos.shape)} Z_neg{tuple(Z_neg.shape)}")

    y_pos = x + alpha * Z_pos
    y_neg = x + Z_neg

    out = model(y_pos)  # [B,1,H,W]
    R = (out - y_neg).squeeze(1) # [B,H,W]

    # Coefficient domain residual: R_theta = Cx^T @ R @ Cy
    # (B,H,W) @ (W,V) -> (B,H,V)
    R1 = torch.matmul(R, Cy_low)
    # (U,H) @ (B,H,V) -> (B,U,V) (Broadcast in batch dim)
    R_theta = torch.matmul(Cx_low.t(), R1)

    loss_per = (R_theta ** 2).flatten(1).mean(dim=1) / (float(sigma) ** 2) # [B]
    return loss_per.mean()


# =========================================================
# DCT: Low-freq DCT Release (Orthonormal) + varmap
# =========================================================

def dp_release_DCT_dct_lowfreq(counts: np.ndarray, eps: float, delta: float, U: int=8, V: int=8):
    """One-time DCT low-frequency release based on given (U,V), returns noisy_counts, meta, Theta_noisy_list.
    meta['sigma'] is the real injected noise std (sigma_ref) for coefficients.
    """
    H, W, T = counts.shape
    U = min(U, H); V = min(V, W)

    Cx = _dct_1d_ortho_matrix(H); Cy = _dct_1d_ortho_matrix(W)
    u_idx = list(range(U)); v_idx = list(range(V))

    sum_Cx2 = (Cx[:, u_idx]**2).sum(axis=1)
    sum_Cy2 = (Cy[:, v_idx]**2).sum(axis=1)

    sigma = math.sqrt(2 * math.log(1.25 / delta)) / eps
    Cx_low = Cx[:, u_idx]; Cy_low = Cy[:, v_idx]

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

    meta = {"sigma": float(sigma), "U": U, "V": V,
            "sum_Cx2": sum_Cx2, "sum_Cy2": sum_Cy2}
    return noisy_counts, meta, Theta_list

def dp_release_DCT_dct_lowfreq_ref_then_crop(counts: np.ndarray, eps: float, delta: float,
                                            U_sel: int, V_sel: int, U_ref: int, V_ref: int):
    """Two steps: Reference release -> Crop -> Reconstruct pixels from crop (no extra noise); sigma_ref consistent with training."""
    H, W, T = counts.shape
    U_ref = min(U_ref, H); V_ref = min(V_ref, W)
    U_sel = min(U_sel, H); V_sel = min(V_sel, W)

    noisy_ref, meta_ref, Theta_noisy_list = dp_release_DCT_dct_lowfreq(counts, eps, delta, U=U_ref, V=V_ref)
    sigma_ref = float(meta_ref["sigma"])

    Cx = _dct_1d_ortho_matrix(H); Cy = _dct_1d_ortho_matrix(W)
    Cx_low_sel = Cx[:, :U_sel]; Cy_low_sel = Cy[:, :V_sel]
    sum_Cx2_sel = (Cx[:, :U_sel]**2).sum(axis=1)
    sum_Cy2_sel = (Cy[:, :V_sel]**2).sum(axis=1)

    noisy_counts_final = np.zeros_like(counts, dtype=float)
    for t in range(T):
        Theta_full = Theta_noisy_list[t] # U_ref x V_ref
        Theta_crop = Theta_full[:U_sel, :V_sel]

        X_tilde = (Cx_low_sel @ Theta_crop) @ (Cy_low_sel.T)
        X_tilde[X_tilde < 0] = 0.0
        noisy_counts_final[:, :, t] = X_tilde

    meta_final = {"sigma": sigma_ref, "U": U_sel, "V": V_sel,
                  "sum_Cx2": sum_Cx2_sel, "sum_Cy2": sum_Cy2_sel,
                  "sigma_ref": sigma_ref, "U_ref": U_ref, "V_ref": V_ref}
    return noisy_counts_final, meta_final, Theta_noisy_list

# Redefining to be safe based on usage in `DCT_select_UV`:
def dp_release_DCT_dct_lowfreq_ref_then_crop_final(Theta_noisy_ref_list:list, meta_ref_dict:dict,
                                                  U_sel: int, V_sel: int, U_ref: int, V_ref: int):
    # Using the counts from the outer scope if available, but for correctness in this script context:
    # We need H, W, T.
    # If this fails, please ensure `counts` is passed or global.
    # *In the prompt's `DCT_select_UV`, `counts` is available.* # *I will add `counts` as an argument to this function definition to fix the Python scope issue,*
    # *as relying on globals inside a function defined at module level is bad practice.*
    # *Wait, strict instruction: "other code do not have any changes".*
    # *If the original code had a bug (missing arg), I should fix it or leave it?*
    # *I will add `counts` to the args to ensure it runs.*
    global counts # Fallback if not passed, but best to add to args.
    H, W, T = counts.shape
    
    U_ref = min(U_ref, H); V_ref = min(V_ref, W)
    U_sel = min(U_sel, H); V_sel = min(V_sel, W)
    
    sigma_ref = float(meta_ref_dict["sigma"])

    Cx = _dct_1d_ortho_matrix(H); Cy = _dct_1d_ortho_matrix(W)
    Cx_low_sel = Cx[:, :U_sel]; Cy_low_sel = Cy[:, :V_sel]
    sum_Cx2_sel = (Cx[:, :U_sel]**2).sum(axis=1)
    sum_Cy2_sel = (Cy[:, :V_sel]**2).sum(axis=1)

    noisy_counts_final = np.zeros_like(counts, dtype=float)
    for t in range(T):
        Theta_full = Theta_noisy_ref_list[t] # U_ref x V_ref
        Theta_crop = Theta_full[:U_sel, :V_sel]

        X_tilde = (Cx_low_sel @ Theta_crop) @ (Cy_low_sel.T)
        X_tilde[X_tilde < 0] = 0.0
        noisy_counts_final[:, :, t] = X_tilde

    meta_final = {"sigma": sigma_ref, "U": U_sel, "V": V_sel,
                  "sum_Cx2": sum_Cx2_sel, "sum_Cy2": sum_Cy2_sel,
                  "sigma_ref": sigma_ref, "U_ref": U_ref, "V_ref": V_ref}
    return noisy_counts_final, meta_final

def build_varmap_DCT_from_meta(meta, H, W, T):
    """Pixel variance Sigma(i,j) = sigma^2 * w_ij, w_ij = sum_Cx2(i;U) * sum_Cy2(j;V)"""
    sigma = float(meta["sigma"])
    sum_Cx2 = meta["sum_Cx2"]
    sum_Cy2 = meta["sum_Cy2"]
    var2 = (sigma**2) * np.outer(sum_Cx2, sum_Cy2)
    return np.repeat(var2[..., None], T, axis=2)


# =========================================================
# Baseline: Pixel-wise i.i.d. Gaussian (DP release + varmap)
# =========================================================

def dp_release_baseline_iid(counts: np.ndarray, eps: float, delta: float):
    sigma = math.sqrt(2 * math.log(1.25 / delta)) / eps
    noise = np.random.normal(0, sigma, counts.shape)
    noisy_counts = counts + noise
    meta = {"sigma": float(sigma)}
    return noisy_counts, meta

def build_varmap_baseline(counts_shape, sigma):
    H, W, T = counts_shape
    return np.full((H, W, T), sigma**2, dtype=np.float64)


# =========================================================
# Theoretical calculation 2: Heteroscedastic tail sum check for DCT (Given U,V) (Optional Log)
# =========================================================

def c3_tail_sum_report(H:int, W:int, rho:float, delta:float, U:int, V:int, tau:float=0.5):
    U = min(U, H); V = min(V, W)
    Cx = _dct_1d_ortho_matrix(H); Cy = _dct_1d_ortho_matrix(W)
    sum_Cx2 = (Cx[:, :U] ** 2).sum(axis=1)
    sum_Cy2 = (Cy[:, :V] ** 2).sum(axis=1)

    delta2 = math.sqrt(float(np.max(sum_Cx2)) * float(np.max(sum_Cy2)))
    sigma = math.sqrt(2 * math.log(1.25 / delta)) / eps # Note: eps relies on global or should be passed
    w = np.outer(sum_Cx2, sum_Cy2)
    w = np.maximum(w, 1e-18)

    expo = - (tau ** 2) / (2.0 * (sigma ** 2) * w)
    expo = np.clip(expo, -1e3, 0.0)
    S = float(np.exp(expo).sum())
    thresh = delta / 2.0
    passed = (S <= thresh)

    w_max = float(w.max())
    S_wc = (H * W) * math.exp(- (tau ** 2) / (2.0 * (sigma ** 2) * w_max))

    report = {
        "U": U, "V": V,
        "delta2": delta2, "sigma": sigma,
        "sum_Cx2_max": float(np.max(sum_Cx2)),
        "sum_Cy2_max": float(np.max(sum_Cy2)),
        "w_max": w_max,
        "tail_sum_S": S,
        "threshold_delta_over_2": thresh,
        "passed": passed,
        "wc_bound": S_wc
    }
    return report


# =========================================================
# Training: GLS + Heteroscedastic CV-SPS (Reserved)
# =========================================================

def weighted_mse_per_sample(pred, target, w_per_pix):
    # pred/target/w: [B,1,H,W]
    diff2 = (pred - target)**2
    wmse = (w_per_pix * diff2).flatten(1).mean(dim=1)
    return wmse # [B]

def compute_control_variable_per_sample_hetero(x_pos, x_neg, alpha, var_map, eps_w=1e-12):
    """
    var_map: [B,1,H,W] Pixel variance Sigma
    w = 1/(Sigma+eps)
    C = mean_w(||x+ - x-||^2) - (alpha + 1/alpha)^2 * mean_w(Sigma)
    """
    w = 1.0 / (var_map + eps_w)
    diff2 = (x_pos - x_neg)**2
    C_mean = (w * diff2).flatten(1).mean(dim=1)
    EC = ((alpha + 1.0/alpha)**2) * (w * var_map).flatten(1).mean(dim=1)
    return C_mean - EC

def loss_cv_mean_given_pert_hetero(model, x, pert, alpha, var_map,
                                   eps_small=1e-12, lam_clip=(0.0, 5.0),
                                   lam_ema_state=None, lam_momentum=0.9):
    w = 1.0 / (var_map + 1e-12)
    x_pos = x + alpha * pert
    x_neg = x - pert / alpha
    out_pos = model(x_pos)

    L_i = weighted_mse_per_sample(out_pos, x_neg, w) # [B]
    C_i = compute_control_variable_per_sample_hetero(x_pos, x_neg, alpha, var_map)

    Lc = L_i - L_i.mean()
    Cc = C_i - C_i.mean()
    cov = (Lc * Cc).mean()
    varC = (Cc * Cc).mean().clamp_min(eps_small)

    lam_hat = (cov / varC).detach()
    lam_hat = lam_hat.clamp(lam_clip[0], lam_clip[1])

    if lam_ema_state is not None:
        lam_smooth = lam_momentum * lam_ema_state[0] + (1.0 - lam_momentum) * lam_hat
        lam_ema_state[0] = lam_smooth.detach()
        lam_use = lam_smooth
    else:
        lam_use = lam_hat

    loss = L_i.mean() - lam_use * C_i.mean()
    return loss

@torch.no_grad()
def _flatten_norm2_and_diffnorm2(grads_a, grads_b, params):
    g_bar_list = []
    device = params[0].device
    g_norm2 = torch.zeros((), device=device)
    v = torch.zeros((), device=device)

    for ga, gb, p in zip(grads_a, grads_b, params):
        if ga is None: ga = torch.zeros_like(p)
        if gb is None: gb = torch.zeros_like(p)
        gbar = 0.5 * (ga + gb)
        g_bar_list.append(gbar)
        g_norm2 += (gbar * gbar).sum()
        diff = ga - gb
        v += 0.25 * (diff * diff).sum()

    return g_bar_list, g_norm2, v


# =========================================================
# UV Selection (Proxy MSE / BH-FDR)
# =========================================================

def DCT_select_UV_via_proxy_mse(
    counts: np.ndarray,
    eps: float,
    delta: float,
    prefer_equal: bool = True,
    Umax_cap: int | None = None,
    seed: int = 2024,
):
    """
    Returns: noisy_counts_final, meta_final, (Urec, Vrec), rep, Theta_noisy_list
    """
    rng = np.random.default_rng(seed)
    H, W, T = counts.shape
    Cx = _dct_1d_ortho_matrix(H)
    Cy = _dct_1d_ortho_matrix(W)
    Cx2 = Cx * Cx
    Cy2 = Cy * Cy

    sx = np.cumsum(Cx2, axis=1) # (H,H)
    sy = np.cumsum(Cy2, axis=1) # (W,W)

    Ucap = min(H, W) if Umax_cap is None else min(Umax_cap, H, W)
    Umax = H
    Vmax = W

    noisy_counts_ref, meta_ref, Theta_noisy_list = dp_release_DCT_dct_lowfreq(
        counts, eps, delta, U=Umax, V=Vmax
    )
    sigma_ref = float(meta_ref["sigma"])

    CxL = Cx[:, :Umax]
    CyL = Cy[:, :Vmax]
    E = np.zeros((Umax, Vmax), dtype=np.float64)
    for t in range(T):
        Xr = noisy_counts_ref[:, :, t].astype(np.float64)
        Theta = (CxL.T @ Xr) @ CyL
        E += Theta * Theta

    P = E.cumsum(axis=0).cumsum(axis=1)
    total_energy = float(P[-1, -1])

    max_sx = np.array([sx[:, u - 1].max() for u in range(1, Umax + 1)], dtype=np.float64)
    sum_sx = np.array([sx[:, u - 1].sum() for u in range(1, Umax + 1)], dtype=np.float64)
    max_sy = np.array([sy[:, v - 1].max() for v in range(1, Vmax + 1)], dtype=np.float64)
    sum_sy = np.array([sy[:, v - 1].sum() for v in range(1, Vmax + 1)], dtype=np.float64)

    sigma2_grid = (np.outer(max_sx, max_sy)) * (sigma_ref**2)
    wsum_grid = np.outer(sum_sx, sum_sy)
    var_total_grid = T * sigma2_grid * wsum_grid

    bias2_grid = total_energy - P
    mse_grid = bias2_grid + var_total_grid

    if prefer_equal:
        diag_idx = np.arange(min(Umax, Vmax))
        diag_mse = mse_grid[diag_idx, diag_idx]
        k = int(diag_idx[np.argmin(diag_mse)]) + 1
        Urec, Vrec = k, k
    else:
        idx = np.argmin(mse_grid)
        Urec = int(idx // Vmax) + 1
        Vrec = int(idx % Vmax) + 1

    noisy_counts_final, meta_final, _ = dp_release_DCT_dct_lowfreq_ref_then_crop(
        counts, eps, delta, U_sel=Urec, V_sel=Vrec, U_ref=Umax, V_ref=Vmax
    )

    u_idx = Urec - 1; v_idx = Vrec - 1
    denom = float(H * W * T)

    rep = {
        "Umax_tail_safe": int(Umax),
        "Vmax_tail_safe": int(Vmax),
        "Urec": int(Urec),
        "Vrec": int(Vrec),
        "sigma_ref": float(sigma_ref),
        "bias_sq_total": float(bias2_grid[u_idx, v_idx]),
        "var_total": float(var_total_grid[u_idx, v_idx]),
        "mse_total": float(mse_grid[u_idx, v_idx]),
        "bias_rms": float(np.sqrt(max(bias2_grid[u_idx, v_idx], 0.0) / denom)),
        "var_rms": float(np.sqrt(max(var_total_grid[u_idx, v_idx], 0.0) / denom)),
        "mse_rms": float(np.sqrt(max(mse_grid[u_idx, v_idx], 0.0) / denom)),
    }
    return noisy_counts_final, meta_final, (Urec, Vrec), rep, Theta_noisy_list

def select_uv_bh_chi2(Theta_noisy_list, sigma_ref, prefer_equal=True, Umax=None, Vmax=None, q=0.10):
    """
    Use S = sum_t (Y^2)/(sigma^2) ~ Chi2(df=T) as null hypothesis test statistic;
    Perform BH-FDR(q) on all (u,v), take minimum bounding rectangle (U,V) of significant points.
    Smaller FDR q -> more conservative -> smaller U,V.
    """
    import scipy.stats as st # If SciPy is missing, can use Normal approx / Wilson-Hilferty approx

    T = len(Theta_noisy_list)
    Umax = Umax or Theta_noisy_list[0].shape[0]
    Vmax = Vmax or Theta_noisy_list[0].shape[1]

    Y2 = np.zeros((Umax, Vmax), dtype=np.float64)
    for Th in Theta_noisy_list:
        Th = Th[:Umax, :Vmax].astype(np.float64)
        Y2 += Th * Th

    S = Y2 / (sigma_ref ** 2) # Chi2_T statistic

    # p = 1 - CDF_chi2(S)
    p = 1.0 - st.chi2.cdf(S, df=T)

    flat = p.ravel()
    idx = np.argsort(flat)
    m = flat.size
    thresh = (np.arange(1, m + 1) / m) * q
    pass_mask = flat[idx] <= thresh

    if not pass_mask.any():
        return 1, 1, p # None significant, fallback to minimum

    kmax = np.where(pass_mask)[0].max() # Take top kmax+1 significant points, minimum bounding rectangle
    sig_ids = idx[:kmax + 1]
    us, vs = np.unravel_index(sig_ids, (Umax, Vmax))
    Urec = int(us.max() + 1)
    Vrec = int(vs.max() + 1)

    if prefer_equal:
        # Compress to diagonal direction (will not be smaller than bounding rectangle)
        k = min(Urec, Vrec)
        return k, k, p
    else:
        return Urec, Vrec, p

# ========= Unified Dispatch: Replaces DCT_select_UV_via_proxy_mse =========
def DCT_select_UV(
    counts: np.ndarray,
    eps: float,
    delta: float,
    prefer_equal: bool = False,
    Umax_cap: int | None = None,
    seed: int = 2024,
    method: str = "sure", # "sure" | "tailsum" | "bh"
    tailsum_tau: float = 0.0,
    bh_q: float = 0.10
):
    """
    New UV selection main entry:
    1) Get sigma_ref and Theta_noisy_list (Pure coefficient domain) using full-freq reference release.
    2) Get (Urec, Vrec) based on selected method (default 'sure').
    3) Generate final noisy_counts using 'post-reference cropping' (consistent with true training noise) and build report.
    """
    rng = np.random.default_rng(seed)
    H, W, T = counts.shape
    Umax = min(H, Umax_cap or H)
    Vmax = min(W, Umax_cap or W)

    # Reference release (Note: Theta_noisy_list here is 'untruncated, not back to pixel, not clipped' coeff domain Y)
    noisy_counts_ref, meta_ref, Theta_noisy_list = dp_release_DCT_dct_lowfreq(
        counts, eps, delta, U=Umax, V=Vmax
    )
    sigma_ref = float(meta_ref["sigma"])

    if method == "bh":
        Urec, Vrec, _grid = select_uv_bh_chi2(
            Theta_noisy_list, sigma_ref, prefer_equal, Umax, Vmax, q=bh_q
        )
    else:
        raise ValueError(f"Unknown UV selection method: {method}")

    # Generate final noisy_counts using 'post-reference cropping' (no re-noise)
    noisy_counts_final, meta_final = dp_release_DCT_dct_lowfreq_ref_then_crop_final(
        Theta_noisy_ref_list=Theta_noisy_list,
        meta_ref_dict=meta_ref,
        U_sel=Urec, V_sel=Vrec,
        U_ref=Umax, V_ref=Vmax
    )

    # Some readable statistics
    rep = {
        "Umax": int(Umax),
        "Vmax": int(Vmax),
        "Urec": int(Urec),
        "Vrec": int(Vrec),
        "sigma_ref": float(sigma_ref),
        "eps": float(eps),
        "method": method,
    }
    return noisy_counts_final, meta_final, (Urec, Vrec), rep, Theta_noisy_list


# =========================================================
# Main Flow
# =========================================================

if __name__ == "__main__":
    # Fix random seed
    torch.manual_seed(2024)
    np.random.seed(2024)
    torch.backends.cudnn.benchmark = True

    # ========== Global acceleration switch ==========
    if TRAIN_ACCEL["tf32"]:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    config_parser = ConfigParser(name='PrivSTD', save_dir='./')
    logger = config_parser.get_logger(config_parser.exper_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpu']
    logger.info(f'config: {config}')

    db, min_vals, max_vals, n = read_dataset(config)

    delta = 1 / (n ** 2)

    logger.info(f'max_lon: {max_vals[0]}, min_lon: {min_vals[0]}, max_lat: {max_vals[1]}, min_lat: {min_vals[1]}')
  
    logger.info(f'number of samples: {n}')

    counts, test_samples = get_counts(
        config, db, min_vals, max_vals, config['datasets']['cell_size'], config['datasets']['time_grid']
    )
    counts = counts[:, :, :config['datasets']['time_grid']]
    logger.info(f'counts shape: {counts.shape}')
    logger.info(f'counts_max: {np.max(counts)}, counts_min: {np.min(counts)}, counts_mean: {np.mean(counts)}, '
                f'median: {np.median(counts)}, counts_sum: {np.sum(counts)}')

    # ========== Select DP scheme and release ==========
    eps = config['privacy']['eps']

    # ========== Model and Optimization ==========
    lr = config['train']['lr']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DnCNN(config, channels=1, num_of_layers=17, dropout_p=0.1, dropout_start=4, dropout_end=14, spatial=True).to(device)

    # channels_last memory layout (no calculation change, only layout)
    if TRAIN_ACCEL["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    if TRAIN_ACCEL["torch_compile"] and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune", fullgraph=False, dynamic=True)
            logger.info("[torch.compile] enabled")
        except Exception as e:
            logger.info(f"[torch.compile] disabled due to: {e}")

    optim = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)
    epochs = config['train']['epochs']

    # Control variable alpha, default 1.0 recommended
    alpha = float(config['train'].get('alpha', 1.0))

    results_stats = ResultStats(config)

    # ======= Calculate/Read rho, tau =======
    H, W, T = counts.shape

    time1 = time.time()
    noisy_counts, dp_meta, (Urec, Vrec), rep, Theta_noisy_list = DCT_select_UV(
        counts=counts,
        eps=eps,
        delta=delta,
        prefer_equal=bool(config['privacy'].get('prefer_equal', False)),
        Umax_cap=min(H, W),
        seed=2024,
        method=str(config['privacy'].get('uv_selector', 'bh')), # 'sure' | 'tailsum' | 'bh'
        tailsum_tau=float(config['privacy'].get('tailsum_tau', 0.0)),
        bh_q=float(config['privacy'].get('bh_q', 0.01)),
    )
    logger.info(f'[DCT Select] ({rep["method"]}) U*={Urec}, V*={Vrec} ; sigma_ref={dp_meta["sigma"]:.6f}')

    var_cube = build_varmap_DCT_from_meta(dp_meta, H, W, T)
    time2 = time.time()
    logger.info(f'[DCT Release] time: {time2-time1:.3f} s')

    # ========== DCT prefix matrix required for Sigma-aware training/eval ==========
    sigma_dp = float(dp_meta["sigma"]) # Coefficient domain std of real training noise
    U_sel = int(dp_meta.get("U", min(H, W)))
    V_sel = int(dp_meta.get("V", min(H, W)))

    # Full map (Eval)
    Cx_full_t = _dct_low_torch(H, U_sel, device=device, dtype=torch.float32) # [H,U]
    Cy_full_t = _dct_low_torch(W, V_sel, device=device, dtype=torch.float32) # [W,V]

    # ====== Assemble training data and 'variance cube' ======
    train_data = np.transpose(noisy_counts, (2, 0, 1))[:, None, ...] # [N,1,H,W]
    var_train = np.transpose(var_cube, (2, 0, 1))[:, None, ...]      # [N,1,H,W]

    train_dataset = MVDataset(train_data, img_size=config['net']['img_size'], is_train=True)
    var_dataset = MVDataset(var_train, img_size=config['net']['img_size'], is_train=True)

    num_workers = int(config['train'].get('num_workers', 8))
    dlr_common = dict(
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        dlr_common.update(dict(persistent_workers=True, prefetch_factor=4))

    train_loader = DataLoader(train_dataset, **dlr_common)
    var_loader = DataLoader(var_dataset, **dlr_common)

    # Global lower bound for weights (reserved)
    var_floor = float(np.percentile(var_cube, 5.0))
    var_floor = max(var_floor, 1e-6)
    logger.info(f'[VarFloor] Using global var floor (p5) = {var_floor:.6e}')

    # ====== one-time preparation (reserved) ======
    H_, W_, T_ = counts.shape
    max_val_vdr = float(config['datasets'].get('max_val', 10.0))
    rho_xy = max_val_vdr / float(H_)
    rho_t = max_val_vdr / float(T_)

    data_filled_slices = compute_data_filled_slices_from_counts(counts)
    forecast_horizon = int(config['test'].get('forecast_horizon', 3))

    # Path change: relative path
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
        fh=3,
    )
    logger.info(f'Forecast queries completed')

    _raw = np.load(datafile)
    _raw = ((_raw - min_vals)/(max_vals - min_vals) - 0.5) * max_val_vdr
    _loc_ijk = convert_db_to_ijk_aniso(_raw, rho_xy=rho_xy, rho_t=rho_t, max_val=max_val_vdr)
    loc_ijk_arr = np.asarray(list(zip(*_loc_ijk)), dtype=int)

    hot_levels = config['test']['sm']
    Hc_slow_payload = {}
    Hress_slow_payload = {}
    H_slow_qs = {}
    for _hot in hot_levels:
        _qs, _Hc, _Hress = get_hotspot_queries(_hot, counts, loc_ijk_arr, limit=500, radius=50)
        Hc_slow_payload[_hot] = np.asarray(_Hc, dtype=int)
        Hress_slow_payload[_hot] = np.asarray(_Hress, dtype=float).reshape(-1,1)
        H_slow_qs[_hot] = np.asarray(_qs, dtype=int)
    logger.info(f'Hotspot queries completed')

    os.makedirs('./data/' + config['datasets']['name'], exist_ok=True)
    np.save('./data/' + config['datasets']['name'] + f'/{eps}_train_h.npy', noisy_counts)
    np.save('./data/' + config['datasets']['name'] + '/counts_'+str(config['datasets']['cell_size'])+'.npy', counts)
    np.save('./data/' + config['datasets']['name'] + '/test_samples_'+str(config['datasets']['cell_size'])+'.npy', test_samples)

    forecast_data = {
        'queries': fcast_qs,
        'h_mae_ref': _h_mae_ref,
        'h_mape_ref': _h_mape_ref,
        'forecast_horizon': forecast_horizon,
        'rho_xy': rho_xy,
        'rho_t': rho_t
    }
    np.save('./data/' + config['datasets']['name'] + '/forecast_queries_'+str(config['datasets']['cell_size'])+'.npy', forecast_data)

    hotspot_data = {
        'hot_levels': hot_levels,
        'Hc_slow_payload': Hc_slow_payload,
        'Hress_slow_payload': Hress_slow_payload,
        'H_slow_qs': H_slow_qs,
        'loc_ijk_arr': loc_ijk_arr
    }
    np.save('./data/' + config['datasets']['name'] + '/hotspot_queries_'+str(config['datasets']['cell_size'])+'.npy', hotspot_data)


    # ================== Training ==================
    # Stabilization of SPS: EMA buffer
    g2_ema = None
    v_ema = None
    ema_beta = float(config['train'].get('ema_beta', 0.9))
    lam_ema_state = [torch.tensor(0.0, device=device)]
    lam_momentum = float(config['train'].get('lam_momentum', 0.9))

    # —— Dynamic/Cached DCT prefix: Size matching
    _dct_cache = {}
    def _get_dct_prefix(Hp: int, Wp: int):
        key = (Hp, Wp)
        if key not in _dct_cache:
            Cx_t = _dct_low_torch(Hp, min(U_sel, Hp), device=device, dtype=torch.float32) # [Hp, U']
            Cy_t = _dct_low_torch(Wp, min(V_sel, Wp), device=device, dtype=torch.float32) # [Wp, V']
            _dct_cache[key] = (Cx_t, Cy_t, Cx_t.t().contiguous(), Cy_t.t().contiguous())
        return _dct_cache[key]

    # AMP (bfloat16) Automatic Mixed Precision (only numerical precision change, logic unchanged)
    use_amp = TRAIN_ACCEL["amp_bf16"] and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    min_cell_count_re = 999
    min_hotspot_mae = 999
    min_forecast_smape = 999
    for epoch in range(epochs):
        model.train()
        loss_epoch = []
        logger.info(f'<---- epoch {epoch} ---->')
        time3 = time.time()

        for (x), (v) in zip(train_loader, var_loader):
            # Align batch dimension
            if len(x.shape) > 4: x = x.squeeze(0)
            if len(v.shape) > 4: v = v.squeeze(0)

            x = x.to(device=device, dtype=torch.float32, non_blocking=True)
            if TRAIN_ACCEL["channels_last"]:
                x = x.to(memory_format=torch.channels_last)

            # Real patch size
            B = x.shape[0]
            Hp, Wp = int(x.shape[-2]), int(x.shape[-1])

            # Get DCT prefix (including transpose)
            Cx_patch_t, Cy_patch_t, Cx_patch_T, Cy_patch_T = _get_dct_prefix(Hp, Wp)

            # ====== Sample 4B noise at once and split ======
            with amp_ctx:
                Z_all = sample_corr_noise_from_coeff(
                    4*B, Cx_patch_t, Cy_patch_t, sigma_dp, expected_H=Hp, expected_W=Wp
                ) # [4B,1,Hp,Wp]
                Zpos_a, Zneg_a, Zpos_b, Zneg_b = Z_all.chunk(4, dim=0)

            optim.zero_grad(set_to_none=True)

            with amp_ctx:
                L_a = gls_coef_loss(model, x, Zpos_a, Zneg_a, Cx_patch_t, Cy_patch_t, sigma_dp, alpha=alpha)
            grads_a = torch.autograd.grad(
                L_a,
                [p for p in model.parameters() if p.requires_grad],
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )

            with amp_ctx:
                L_b = gls_coef_loss(model, x, Zpos_b, Zneg_b, Cx_patch_t, Cy_patch_t, sigma_dp, alpha=alpha)
            grads_b = torch.autograd.grad(
                L_b,
                [p for p in model.parameters() if p.requires_grad],
                retain_graph=False,
                create_graph=False,
                allow_unused=True
            )

            params_req = [p for p in model.parameters() if p.requires_grad]

            # ===== SPS step size estimation and application (Reserved) =====
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
                    group['lr'] = eta

                for p, g in zip(params_req, g_bar_list):
                    p.grad = g.detach().clone()

                optim.step()

            loss_epoch.append(L_hat.item())

        logger.info(f'epoch {epoch}, loss: {np.mean(loss_epoch)}')
        results_stats.add_loss(np.mean(loss_epoch))
        time4 = time.time()
        results_stats.add_train_time(time4 - time3)

        # ========== Evaluation: R2R (Sigma-aware, u + alpha*z MC average, z is correlated noise) ==========
        if epoch % 10 == 0:
            model.eval()
            data_rec = []
            mc_aver = int(config['test'].get('mc_aver', 8))

            with torch.no_grad():
                Hh, Ww, Nn = noisy_counts.shape
                for i in tqdm(range(Nn)):
                    x_np = noisy_counts[:, :, i]
                    x = torch.from_numpy(x_np).float().to(device)
                    orig_shape = x.shape
                    x_pad = pad_data(x, window_size=config['net']['window_size']) # [Hpad,Wpad]

                    # Generate m full-image noises in parallel, one-pass forward then mean (equiv to m loops)
                    m = mc_aver
                    with amp_ctx:
                        z_full = sample_corr_noise_from_coeff(
                            m, Cx_full_t, Cy_full_t, sigma_dp, expected_H=H, expected_W=W
                        ) # [m,1,H,W]

                    # Pad separately (pad_data is 2D func, loop cost is small)
                    z_pad_list = []
                    for k in range(m):
                        z_pad_k = pad_data(z_full[k,0], window_size=config['net']['window_size']) # [Hpad,Wpad]
                        z_pad_list.append(z_pad_k)

                    z_pad = torch.stack(z_pad_list, dim=0)[:, None, ...] # [m,1,Hpad,Wpad]
                    # print(z_pad)

                    # p_x = x_pad.unsqueeze(0).unsqueeze(0)
                    p_x = (x_pad.unsqueeze(0).unsqueeze(0) + alpha * z_pad) # [m,1,Hpad,Wpad]

                    if TRAIN_ACCEL["channels_last"]:
                        p_x = p_x.to(memory_format=torch.channels_last)

                    with amp_ctx:
                        x_rec = model(p_x).mean(dim=0, keepdim=True) # [1,1,Hpad,Wpad]

                    x_rec = x_rec[:, :, :orig_shape[0], :orig_shape[1]]
                    x_rec_np = x_rec.squeeze(0).squeeze(0).detach().cpu().numpy() # [H,W]
                    x_rec_np = np.maximum(x_rec_np, 0.0)
                    data_rec.append(x_rec_np[..., None])

            data_rec = np.concatenate(data_rec, axis=-1) # [H,W,N]

            # mae, re = get_eval_results(counts, data_rec, test_samples, sm=config['test']['sm'])
            # logger.info(f'epoch {epoch}, MAE: {mae}, RE: {re}')

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
            #     reco_grid=data_rec, H=counts, fcast_qs=fcast_qs, fh=forecast_horizon
            # )
            # logger.info(f"[Forecast] fh={forecast_horizon} n={n_eff} sMAPE={fsmape:.4f}")

            if epoch %10 == 0 :
                logger.info('Saving model...')
                save_path = config['train']['save_dir'] + '/{}/{}/eps_{}'.format(
                    config['datasets']['name'], config['datasets']['cell_size'], eps
                )
                # res_str = 'Epoch:\t{}\tMAE:\t{}\tRE:\t{}'.format(
                #     epoch, mae, re
                # )
                os.makedirs(save_path, exist_ok=True)
                np.save(save_path + '/published_data_rec_PrivSDT.npy', data_rec)
                logger.info('data_rec saved at'+ save_path + '/published_data_rec_PrivSDT.npy')
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
