# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============== 基础：DCT 正交矩阵 ==============
def _dct_1d_ortho_matrix(N: int) -> np.ndarray:
    n = np.arange(N, dtype=float)
    C = np.zeros((N, N), dtype=float)
    C[:, 0] = 1.0 / math.sqrt(N)
    for k in range(1, N):
        C[:, k] = math.sqrt(2.0 / N) * np.cos((math.pi * (2.0 * n + 1.0) * k) / (2.0 * N))
    return C

# ============== 子带划分与 Δ2 ==============
def make_uv_bands(U: int, V: int, n_bands: int = 4) -> List[Dict[str, int]]:
    bands = []
    for b in range(n_bands):
        u1 = int(round((b + 1) / n_bands * U))
        v1 = int(round((b + 1) / n_bands * V))
        u0 = int(round(b / n_bands * U))
        v0 = int(round(b / n_bands * V))
        bands.append({"u0": u0, "u1": u1, "v0": v0, "v1": v1})
    return bands

def delta2_for_band(Cx2: np.ndarray, Cy2: np.ndarray, band: Dict[str, int]) -> float:
    u0,u1,v0,v1 = band["u0"],band["u1"],band["v0"],band["v1"]
    sx = Cx2[:, u0:u1].sum(axis=1)
    sy = Cy2[:, v0:v1].sum(axis=1)
    return float(np.sqrt(float(np.max(sx)) * float(np.max(sy))))

# ============== ρ 分配器（解析“水位”） ==============
def rho_allocator(delta2_list: List[float], kappa_list: List[float], rho_tot: float) -> List[float]:
    a = np.array([ (d2**2)/2.0 * max(k, 1e-12) for d2,k in zip(delta2_list, kappa_list) ], dtype=float)
    s = np.sqrt(a).sum()
    if s <= 0:
        return [rho_tot / len(delta2_list)] * len(delta2_list)
    return [ float(rho_tot * (np.sqrt(ai)/s)) for ai in a ]

# ============== 频域功率估计（近似 GSURE） ==============
def estimate_band_power(theta_noisy: np.ndarray, sigma_b: List[float], bands: List[Dict[str,int]]) -> List[float]:
    kappa = []
    for band, sig in zip(bands, sigma_b):
        u0,u1,v0,v1 = band["u0"],band["u1"],band["v0"],band["v1"]
        block = theta_noisy[u0:u1, v0:v1]
        power = float(np.mean(block**2) - sig**2)
        power = max(power, 1e-10)
        kappa.append(1.0/power)
    return kappa

# ============== (ε,δ)→ρ ==============
def _dp_to_required_zcdp(epsilon: float, delta: float) -> float:
    a = math.sqrt(math.log(1.0 / delta))
    x = max(0.0, math.sqrt(epsilon + a*a) - a)
    return x*x

# ============== 多子带 zCDP 发布 ==============
def dp_release_C3_multiband_lowfreq(counts: np.ndarray, eps: float, delta: float,
                                     U: int=8, V: int=8, n_bands: int=4,
                                     n_iter_alloc: int=2) -> Tuple[np.ndarray, Dict[str,Any]]:
    H,W,T = counts.shape
    U = min(U, H); V = min(V, W)
    Cx = _dct_1d_ortho_matrix(H); Cy = _dct_1d_ortho_matrix(W)
    Cx2 = Cx**2; Cy2 = Cy**2
    rho_tot = _dp_to_required_zcdp(eps, delta)
    bands = make_uv_bands(U, V, n_bands=n_bands)

    d2_list = [delta2_for_band(Cx2, Cy2, b) for b in bands]
    rho_b = [rho_tot / n_bands] * n_bands
    sigma_b = [ d2 / math.sqrt(2.0*rb) for d2,rb in zip(d2_list, rho_b) ]

    # 自洽 1~2 轮
    Cx_low = Cx[:, :U]; Cy_low = Cy[:, :V]
    X0 = counts[:, :, 0].astype(float)
    Theta0 = (Cx_low.T @ X0) @ Cy_low
    for _ in range(n_iter_alloc):
        kappa = estimate_band_power(Theta0, sigma_b, bands)
        rho_b = rho_allocator(d2_list, kappa, rho_tot)
        sigma_b = [ d2 / math.sqrt(2.0*rb) for d2,rb in zip(d2_list, rho_b) ]

    noisy_counts = np.zeros_like(counts, dtype=float)
    for t in range(T):
        X = counts[:, :, t].astype(float)
        Theta = (Cx_low.T @ X) @ Cy_low
        Theta_noisy = Theta.copy()
        for b, band in enumerate(bands):
            u0,u1,v0,v1 = band["u0"],band["u1"],band["v0"],band["v1"]
            Theta_noisy[u0:u1, v0:v1] += np.random.normal(0.0, sigma_b[b], size=(u1-u0, v1-v0))
        X_tilde = (Cx_low @ Theta_noisy) @ (Cy_low.T)
        X_tilde[X_tilde < 0] = 0.0
        noisy_counts[:, :, t] = X_tilde

    meta = {"U": U, "V": V, "bands": bands, "rho_b": rho_b, "sigma_b": sigma_b, "Cx": Cx, "Cy": Cy}
    return noisy_counts, meta

# ============== 频谱门（修正后的前/逆 DCT） ==============
class SpectralGate(nn.Module):
    """
    低频 DCT 软收缩：theta_hat = gate * theta + bias
    - 动态 H/W：按输入尺寸构造/缓存 DCT
    - 多通道兼容：逐通道相同门控
    """
    def __init__(self, H:int=None, W:int=None, U:int=8, V:int=8,
                 Cx:np.ndarray=None, Cy:np.ndarray=None):
        super().__init__()
        self.H = H; self.W = W
        self.U = U; self.V = V
        self.m = nn.Parameter(torch.zeros(U, V))
        self.bias = nn.Parameter(torch.zeros(U, V))
        self.register_buffer("_Cx", None, persistent=False)
        self.register_buffer("_Cy", None, persistent=False)
        if Cx is not None: self._Cx = torch.from_numpy(Cx).float()
        if Cy is not None: self._Cy = torch.from_numpy(Cy).float()
        self._hw_built = (None, None)

    def _ensure_dct(self, H:int, W:int, device, dtype):
        need = (self._Cx is None) or (self._Cy is None) or (self._hw_built != (H,W))
        if need:
            Cx_np = _dct_1d_ortho_matrix(H)
            Cy_np = _dct_1d_ortho_matrix(W)
            self._Cx = torch.from_numpy(Cx_np).to(device=device, dtype=dtype)
            self._Cy = torch.from_numpy(Cy_np).to(device=device, dtype=dtype)
            self._hw_built = (H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        self._ensure_dct(H, W, device, dtype)
        Cx = self._Cx; Cy = self._Cy

        Ueff = min(self.U, H)
        Veff = min(self.V, W)
        gate = torch.sigmoid(self.m[:Ueff, :Veff])
        bias = self.bias[:Ueff, :Veff]

        Cx_low = Cx[:, :Ueff]   # [H,U]
        Cy_low = Cy[:, :Veff]   # [W,V]

        outs = []
        for ch in range(C):
            xc = x[:, ch:ch+1, :, :]  # [B,1,H,W]

            # 正确二维 DCT：沿 H 再沿 W
            theta = torch.einsum('hu, bchw -> bcuw', Cx_low, xc)      # [B,1,U,W]
            theta = torch.einsum('bcuw, wv -> bcuv', theta, Cy_low)   # [B,1,U,V]
            theta = theta.squeeze(1)                                   # [B,U,V]

            theta_hat = gate * theta + bias                            # [B,U,V]

            # 正确逆 DCT：先 U→H，再 V→W
            X = torch.einsum('hu, buv -> bhv', Cx_low, theta_hat)      # [B,H,V]
            X = torch.einsum('bhv, wv -> bhw', X, Cy_low)              # [B,H,W]

            outs.append(X.unsqueeze(1))

        return torch.cat(outs, dim=1)  # [B,C,H,W]

# ============== GSURE（轻量近似） ==============
def gsure_loss(y: torch.Tensor, F_y: torch.Tensor, var_map: torch.Tensor,
               F: nn.Module, n_mc: int = 1) -> torch.Tensor:
    residual = (F_y - y)
    mse = (residual**2).mean()
    div_est = 0.0
    sigma2_bar = var_map.mean()
    for _ in range(n_mc):
        v = torch.randn_like(y)
        v = v / (v.norm() + 1e-12)
        Fy = F(y)
        hvp = torch.autograd.grad(Fy, y, grad_outputs=v, retain_graph=True, create_graph=False)[0]
        div_est = div_est + (v * hvp).sum() / float(y.numel())
    div_est = div_est / max(n_mc,1)
    return mse + 2.0 * sigma2_bar * div_est.detach()

# ============== 辅助：Sobel ==============
def sobel_grad(x: torch.Tensor) -> torch.Tensor:
    Kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    Ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    gx = F.conv2d(x, Kx, padding=1)
    gy = F.conv2d(x, Ky, padding=1)
    return torch.sqrt(gx*gx + gy*gy + 1e-12)

# ============== 向量控制变量 ==============
def compute_vector_control_variates(x_pos, x_neg, var_map, bands_meta: Dict[str,Any], alpha: float,
                                    use_grad: bool=True) -> Tuple[torch.Tensor, torch.Tensor]:
    w = 1.0/(var_map + 1e-12)
    diff = x_pos - x_neg
    C0  = (w * diff.pow(2)).flatten(1).mean(dim=1)
    EC0 = ((alpha + 1.0/alpha)**2) * (w * var_map).flatten(1).mean(dim=1)
    Cs=[C0]; ECs=[EC0]
    if use_grad:
        gdiff = sobel_grad(diff)
        k_grad = 8.0
        Cg  = (w * gdiff.pow(2)).flatten(1).mean(dim=1)
        ECg = k_grad * ((alpha + 1.0/alpha)**2) * (w * var_map).flatten(1).mean(dim=1)
        Cs.append(Cg); ECs.append(ECg)

    U = bands_meta["U"]; V = bands_meta["V"]
    Cx = torch.from_numpy(bands_meta["Cx"]).to(x_pos.device, dtype=x_pos.dtype)
    Cy = torch.from_numpy(bands_meta["Cy"]).to(x_pos.device, dtype=x_pos.dtype)
    Cx_low = Cx[:, :U]; Cy_low = Cy[:, :V]
    theta_diff = torch.einsum('hu, bchw -> bcuw', Cx_low, diff)
    theta_diff = torch.einsum('bcuw, wv -> bcuv', theta_diff, Cy_low)
    theta_diff = theta_diff.squeeze(1)  # [B,U,V]
    for idx, band in enumerate(bands_meta["bands"]):
        u0,u1,v0,v1 = band["u0"],band["u1"],band["v0"],band["v1"]
        blk = theta_diff[:, u0:u1, v0:v1]
        Cb = blk.pow(2).flatten(1).mean(dim=1)
        sigma_b = float(bands_meta["sigma_b"][idx])
        ECb = torch.full_like(Cb, ((alpha+1.0/alpha)**2) * (sigma_b**2))
        Cs.append(Cb); ECs.append(ECb)

    C_vec  = torch.stack(Cs, dim=1)   # [B,D]
    EC_vec = torch.stack(ECs, dim=1)  # [B,D]
    return C_vec, EC_vec

def loss_cv_mean_given_pert_hetero_vecCV(model: nn.Module, x: torch.Tensor, pert: torch.Tensor,
                                          alpha: float, var_map: torch.Tensor,
                                          bands_meta: Dict[str,Any],
                                          eps_small: float=1e-12, ridge: float=1e-3) -> torch.Tensor:
    w = 1.0/(var_map + 1e-12)
    x_pos = x + alpha * pert
    x_neg = x - pert / alpha
    out_pos = model(x_pos)
    diff2 = (out_pos - x_neg).pow(2)
    L_i = (w * diff2).flatten(1).mean(dim=1)  # [B]
    C_vec, EC_vec = compute_vector_control_variates(x_pos, x_neg, var_map, bands_meta, alpha)
    Cc = C_vec - EC_vec
    Lc = L_i - L_i.mean()
    B, D = Cc.shape
    CtC = torch.matmul(Cc.t(), Cc) / B + ridge*torch.eye(D, device=x.device, dtype=x.dtype)
    CtL = torch.matmul(Cc.t(), Lc) / B
    lam = torch.linalg.solve(CtC, CtL)
    loss = L_i.mean() - torch.dot(lam.detach(), (C_vec - EC_vec).mean(dim=0))
    return loss

# ============== 稳健 CV-SPS ==============
def stable_cv_sps_step(optim: torch.optim.Optimizer, L_a: torch.Tensor, L_b: torch.Tensor,
                       grads_a: List[torch.Tensor], grads_b: List[torch.Tensor],
                       params_req: List[torch.nn.Parameter],
                       beta: float, eta_min: float, eta_max: float,
                       v_floor: float = 1e-10, L_star_shift: float = 0.0) -> float:
    with torch.no_grad():
        L_hat = 0.5*(L_a + L_b)
        g_norm2 = torch.zeros((), device=params_req[0].device)
        vstat = torch.zeros((), device=params_req[0].device)
        for ga, gb, p in zip(grads_a, grads_b, params_req):
            if ga is None: ga = torch.zeros_like(p)
            if gb is None: gb = torch.zeros_like(p)
            gbar = 0.5*(ga + gb)
            g_norm2 += (gbar*gbar).sum()
            diff = ga - gb
            vstat += 0.25*(diff*diff).sum()
        vstat = torch.clamp(vstat, min=v_floor)
        denom = g_norm2 + beta * vstat + 1e-12
        eta = torch.clamp((L_hat - L_star_shift) / denom, min=eta_min, max=eta_max).item()
        for g in optim.param_groups:
            g['lr'] = eta
        # 写平均梯度
        for p, (ga,gb) in zip(params_req, zip(grads_a, grads_b)):
            if ga is None: ga = torch.zeros_like(p)
            if gb is None: gb = torch.zeros_like(p)
            p.grad = (0.5*(ga+gb)).detach().clone()
    optim.step()
    return eta

# ============== 额外指标 ==============
def extra_metrics_psnr_ssim(counts: np.ndarray, reco: np.ndarray, data_range: Optional[float]=None) -> Dict[str,float]:
    H,W,T = counts.shape
    if data_range is None:
        data_range = float(max(counts.max() - counts.min(), 1.0))
    psnrs=[]; ssim_list=[]
    C1 = (0.01*data_range)**2; C2 = (0.03*data_range)**2
    for t in range(T):
        gt = counts[:,:,t].astype(np.float64)
        pr = reco[:,:,t].astype(np.float64)
        mse = np.mean((gt-pr)**2)+1e-12
        psnrs.append(20*np.log10(data_range) - 10*np.log10(mse))
        mu_x, mu_y = gt.mean(), pr.mean()
        sig_x, sig_y = gt.var(), pr.var()
        sig_xy = np.mean((gt-mu_x)*(pr-mu_y))
        ssim = ((2*mu_x*mu_y + C1)*(2*sig_xy + C2))/((mu_x**2+mu_y**2 + C1)*(sig_x+sig_y + C2))
        ssim_list.append(float(ssim))
    return {"PSNR": float(np.mean(psnrs)), "SSIM": float(np.mean(ssim_list))}

def extra_metrics_edge_mae(counts: np.ndarray, reco: np.ndarray) -> float:
    import scipy.signal
    Kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=float)
    Ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=float)
    H,W,T = counts.shape
    maes=[]
    for t in range(T):
        gt = counts[:,:,t].astype(np.float64)
        pr = reco[:,:,t].astype(np.float64)
        gx_gt = scipy.signal.convolve2d(gt, Kx, mode='same', boundary='symm')
        gy_gt = scipy.signal.convolve2d(gt, Ky, mode='same', boundary='symm')
        gx_pr = scipy.signal.convolve2d(pr, Kx, mode='same', boundary='symm')
        gy_pr = scipy.signal.convolve2d(pr, Ky, mode='same', boundary='symm')
        mae = np.mean(np.abs(np.hypot(gx_gt,gy_gt) - np.hypot(gx_pr,gy_pr)))
        maes.append(float(mae))
    return float(np.mean(maes))

def extra_metrics_midband_energy_ratio(counts: np.ndarray, reco: np.ndarray, U:int, V:int,
                                       mid_ratio:Tuple[float,float]=(0.25,0.75)) -> float:
    Cx = _dct_1d_ortho_matrix(counts.shape[0]); Cy = _dct_1d_ortho_matrix(counts.shape[1])
    Cx_low = Cx[:, :U]; Cy_low = Cy[:, :V]
    def band_energy(X):
        Theta = (Cx_low.T @ X) @ Cy_low
        u0 = int(round(mid_ratio[0]*U)); u1=int(round(mid_ratio[1]*U))
        v0 = int(round(mid_ratio[0]*V)); v1=int(round(mid_ratio[1]*V))
        blk = Theta[u0:u1, v0:v1]
        return float(np.mean(blk**2))
    Es, Er = [], []
    for t in range(counts.shape[2]):
        Es.append(band_energy(counts[:,:,t])); Er.append(band_energy(reco[:,:,t]))
    Es = np.mean(Es); Er=np.mean(Er)
    return float(Er / max(Es,1e-12))

def extra_metrics_spectral_kl(counts: np.ndarray, reco: np.ndarray, U:int, V:int, eps:float=1e-9) -> float:
    Cx = _dct_1d_ortho_matrix(counts.shape[0]); Cy = _dct_1d_ortho_matrix(counts.shape[1])
    Cx_low = Cx[:, :U]; Cy_low = Cy[:, :V]
    def spec(X):
        Theta = (Cx_low.T @ X) @ Cy_low
        E = Theta**2; P = E/ max(E.sum(), 1e-12); return P
    P = 0; Q = 0
    for t in range(counts.shape[2]):
        Pg = spec(counts[:,:,t]); Pr = spec(reco[:,:,t])
        P += Pg; Q += Pr
    P /= counts.shape[2]; Q /= counts.shape[2]
    P = P + eps; Q = Q + eps
    return float(np.sum(P * np.log(P / Q)))
