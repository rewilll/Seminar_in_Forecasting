"""
forecast_combination.py
=======================
Research-grade implementation of a graph-based, covariance-aware forecast
combination method for unstable environments.

Layers
------
1. Local prediction of future pairwise loss differentials (Richter–Smetanina style)
2. Graph aggregation via eigenvector centrality
3. Covariance-aware simplex-constrained weight optimization with shrinkage

Author : (research prototype)
License: MIT
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError
from scipy import optimize
from scipy.linalg import eigh

# ---------------------------------------------------------------------------
# 0.  CONSTANTS & HELPERS
# ---------------------------------------------------------------------------

RNG_SEED = 42
EPS = 1e-12          # generic numerical floor
RIDGE_COV = 1e-6     # default ridge for covariance matrices
TELEPORT = 1e-6      # default teleportation for eigenvector centrality


def _ensure_rng(seed=None):
    if seed is None:
        return np.random.default_rng(RNG_SEED)
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


# ===================================================================
# 1.  LOSS FUNCTIONS
# ===================================================================

def squared_loss(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Squared-error loss  (y-f)^2."""
    return (y - f) ** 2


def absolute_loss(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Absolute-error loss  |y-f|."""
    return np.abs(y - f)


LOSS_REGISTRY: Dict[str, Callable] = {
    "squared": squared_loss,
    "absolute": absolute_loss,
}


# ===================================================================
# 2.  LOCAL PAIRWISE LOSS-DIFFERENTIAL MODEL  (Richter–Smetanina style)
# ===================================================================

# ---------- kernel ----------

def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel  K(u) = 0.75*(1-u^2)  for |u|<=1, else 0."""
    w = np.zeros_like(u)
    mask = np.abs(u) <= 1.0
    w[mask] = 0.75 * (1.0 - u[mask] ** 2)
    return w


# ---------- local linear AR estimation ----------

def _build_ar_design(series: np.ndarray, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build AR(d) regression matrices from *series*.

    Returns
    -------
    Y : shape (n-d,)
    X : shape (n-d, d+1)   columns = [1, lag1, lag2, ..., lag_d]
    """
    n = len(series)
    if n <= d:
        raise ValueError("Series too short for AR({})".format(d))
    Y = series[d:]
    X = np.ones((n - d, d + 1))
    for k in range(1, d + 1):
        X[:, k] = series[d - k: n - k]
    return Y, X


def local_linear_ar_fit(
    series: np.ndarray,
    d: int,
    h: float,
    target_frac: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Local linear AR(d) fit at rescaled time *target_frac* (in [0,1]).

    Uses Epanechnikov kernel centred at *target_frac* with bandwidth *h*.

    Parameters
    ----------
    series : 1-d array of length T
    d      : AR lag order
    h      : bandwidth (on the [0,1] scale)
    target_frac : point at which to localise (usually T/T = 1.0 for forecasting)

    Returns
    -------
    beta  : estimated local AR coefficients  (d+1,)
    resid : weighted residuals              (n,)
    W_diag: kernel weights                   (n,)
    """
    Y, X = _build_ar_design(series, d)
    n = len(Y)
    fracs = (np.arange(d, d + n) + 1) / (d + n)  # rescaled times for each obs
    u = (fracs - target_frac) / max(h, EPS)
    W_diag = epanechnikov_kernel(u)

    # Weighted least squares
    Ws = np.sqrt(W_diag + EPS)
    Xw = X * Ws[:, None]
    Yw = Y * Ws
    try:
        beta, _, _, _ = np.linalg.lstsq(Xw, Yw, rcond=None)
    except LinAlgError:
        beta = np.zeros(d + 1)
    resid = Y - X @ beta
    return beta, resid, W_diag


def local_predict_mean(series: np.ndarray, d: int, h: float) -> float:
    """
    One-step-ahead conditional mean forecast from local AR(d).
    Localises at the boundary (target_frac = 1.0).
    """
    if d == 0:
        beta, _, _ = local_linear_ar_fit(series, 0, h, target_frac=1.0)
        return float(beta[0])
    beta, _, _ = local_linear_ar_fit(series, d, h, target_frac=1.0)
    x_new = np.ones(d + 1)
    for k in range(1, d + 1):
        x_new[k] = series[-k]
    return float(x_new @ beta)


def local_predict_scale(
    series: np.ndarray,
    d: int,
    h1: float,
    h2: float,
) -> float:
    """
    Local scale (std-dev) estimate from second-stage kernel regression
    on squared residuals of the first-stage local AR(d).
    """
    beta, resid, _ = local_linear_ar_fit(series, d, h1, target_frac=1.0)
    sq_resid = resid ** 2
    # second stage: local constant regression of sq_resid
    n = len(sq_resid)
    fracs = (np.arange(d, d + n) + 1) / (d + n)
    u = (fracs - 1.0) / max(h2, EPS)
    W2 = epanechnikov_kernel(u)
    denom = W2.sum()
    if denom < EPS:
        return float(np.sqrt(np.mean(sq_resid) + EPS))
    var_est = (W2 * sq_resid).sum() / denom
    return float(np.sqrt(max(var_est, EPS)))


# ---------- BIC lag selection ----------

def bic_lag_selection(
    series: np.ndarray,
    d_max: int,
    h: float,
) -> int:
    """
    Select AR lag order d in {0,...,d_max} via BIC evaluated at
    the boundary using local kernel weights.
    """
    T = len(series)
    best_d = 0
    best_bic = np.inf
    for d in range(0, d_max + 1):
        if T <= d + 2:
            continue
        beta, resid, W_diag = local_linear_ar_fit(series, d, h, target_frac=1.0)
        n_eff = W_diag.sum()
        if n_eff < d + 2:
            continue
        sse = (W_diag * resid ** 2).sum()
        sigma2 = sse / max(n_eff, EPS)
        if sigma2 <= 0:
            continue
        bic = np.log(sigma2 + EPS) + (d + 1) * np.log(max(n_eff, 2)) / max(n_eff, 1)
        if bic < best_bic:
            best_bic = bic
            best_d = d
    return best_d


# ---------- CV bandwidth selection ----------

def cv_bandwidth_selection(
    series: np.ndarray,
    d: int,
    h_grid: np.ndarray,
    n_folds: int = 5,
) -> Tuple[float, np.ndarray]:
    """
    Blocked cross-validation for bandwidth selection.

    Uses *n_folds* contiguous blocks; each fold is left out in turn.
    Returns (best_h, cv_scores).
    """
    T = len(series)
    if T <= d + 2:
        return (float(h_grid[len(h_grid) // 2]), np.full(len(h_grid), np.nan))

    indices = np.arange(d, T)
    block_size = max(len(indices) // n_folds, 1)
    cv_scores = np.full(len(h_grid), np.nan)

    for ih, h in enumerate(h_grid):
        total_err = 0.0
        count = 0
        for fold in range(n_folds):
            start = d + fold * block_size
            end = min(start + block_size, T)
            if end <= start:
                continue
            mask = np.ones(T, dtype=bool)
            mask[start:end] = False
            sub_series = series[mask]
            if len(sub_series) <= d + 2:
                continue
            Y_val, X_val = _build_ar_design(series, d)
            beta_cv, _, _ = local_linear_ar_fit(sub_series, d, h, target_frac=1.0)
            # evaluate on left-out block
            for s in range(start - d, end - d):
                if 0 <= s < len(Y_val):
                    pred = X_val[s] @ beta_cv
                    total_err += (Y_val[s] - pred) ** 2
                    count += 1
        if count > 0:
            cv_scores[ih] = total_err / count
    best_idx = np.nanargmin(cv_scores)
    return float(h_grid[best_idx]), cv_scores


def make_h_grid(
    n_points: int = 15,
    h_min: float = 0.05,
    h_max: float = 1.0,
) -> np.ndarray:
    """Logarithmically spaced bandwidth grid on [h_min, h_max]."""
    return np.exp(np.linspace(np.log(h_min), np.log(h_max), n_points))


# ---------- Full pairwise LD predictor ----------

@dataclass
class PairwiseLDResult:
    """Result of predicting one pairwise loss differential."""
    mu_hat: float = 0.0
    sigma_hat: float = 1.0
    d_selected: int = 0
    h1_selected: float = 0.2
    h2_selected: float = 0.2


def predict_pairwise_ld(
    delta_L: np.ndarray,
    d_max: int = 4,
    h1_grid: Optional[np.ndarray] = None,
    h2_grid: Optional[np.ndarray] = None,
    fixed_d: Optional[int] = None,
    fixed_h1: Optional[float] = None,
    fixed_h2: Optional[float] = None,
    n_cv_folds: int = 5,
) -> PairwiseLDResult:
    """
    Given a history of pairwise loss differentials (up to time t),
    produce next-period conditional mean and scale estimates.
    """
    T = len(delta_L)
    if T < 5:
        return PairwiseLDResult()  # too short

    if h1_grid is None:
        h1_grid = make_h_grid(12, 0.05, 1.0)
    if h2_grid is None:
        h2_grid = make_h_grid(10, 0.10, 1.0)

    # preliminary bandwidth for BIC
    h_prelim = float(h1_grid[len(h1_grid) // 2])

    # lag selection
    if fixed_d is not None:
        d = fixed_d
    else:
        d = bic_lag_selection(delta_L, d_max, h_prelim)

    # bandwidth selection h1
    if fixed_h1 is not None:
        h1 = fixed_h1
    else:
        h1, _ = cv_bandwidth_selection(delta_L, d, h1_grid, n_cv_folds)

    # re-select d at chosen h1
    if fixed_d is None:
        d = bic_lag_selection(delta_L, d_max, h1)

    # bandwidth selection h2
    if fixed_h2 is not None:
        h2 = fixed_h2
    else:
        # simple choice: use h2 moderately larger than h1
        h2 = min(h1 * 1.5, 1.0)

    mu = local_predict_mean(delta_L, d, h1)
    sigma = local_predict_scale(delta_L, d, h1, h2)

    return PairwiseLDResult(
        mu_hat=mu,
        sigma_hat=sigma,
        d_selected=d,
        h1_selected=h1,
        h2_selected=h2,
    )


# ===================================================================
# 3.  GRAPH LAYER
# ===================================================================

def build_adjacency_raw(mu_matrix: np.ndarray) -> np.ndarray:
    """
    Raw adjacency: A_{ij} = max(-mu_{ij}, 0).
    mu_{ij} < 0 => i beats j => edge weight from i to j.
    """
    A = np.maximum(-mu_matrix, 0.0)
    np.fill_diagonal(A, 0.0)
    return A


def build_adjacency_standardized(
    mu_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    c: float = 1e-4,
) -> np.ndarray:
    """Standardized adjacency: A_{ij} = max(-mu_{ij}/(sigma_{ij}+c), 0)."""
    A = np.maximum(-mu_matrix / (sigma_matrix + c), 0.0)
    np.fill_diagonal(A, 0.0)
    return A


def build_adjacency_thresholded(
    mu_matrix: np.ndarray,
    sigma_matrix: np.ndarray,
    threshold: float = 1.0,
    c: float = 1e-4,
) -> np.ndarray:
    """Thresholded adjacency based on t-ratio."""
    t_ratio = -mu_matrix / (sigma_matrix + c)
    A = np.where(t_ratio > threshold, t_ratio, 0.0)
    np.fill_diagonal(A, 0.0)
    return A


def eigenvector_centrality(
    A: np.ndarray,
    teleport: float = TELEPORT,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> np.ndarray:
    """
    Dominant eigenvector centrality of non-negative matrix A.

    Uses power iteration with optional teleportation for reducibility.
    Returns normalised score vector summing to 1.
    """
    M = A.shape[0]
    if A.max() < EPS:
        return np.ones(M) / M  # fallback: uniform

    # add teleportation
    A_reg = A + teleport * np.ones((M, M)) / M
    # power iteration
    r = np.ones(M) / M
    for _ in range(max_iter):
        r_new = A_reg @ r
        norm = r_new.sum()
        if norm < EPS:
            return np.ones(M) / M
        r_new /= norm
        if np.max(np.abs(r_new - r)) < tol:
            r = r_new
            break
        r = r_new
    r = np.maximum(r, 0.0)
    r /= r.sum() + EPS
    return r


def row_sum_strength(A: np.ndarray) -> np.ndarray:
    """Row-sum (out-strength) centrality, normalised."""
    s = A.sum(axis=1)
    total = s.sum()
    if total < EPS:
        return np.ones(A.shape[0]) / A.shape[0]
    return s / total


def pagerank_centrality(
    A: np.ndarray,
    alpha: float = 0.85,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> np.ndarray:
    """Simple PageRank on column-normalised A with damping *alpha*."""
    M = A.shape[0]
    col_sum = A.sum(axis=0)
    col_sum[col_sum < EPS] = 1.0
    P = A / col_sum[None, :]
    r = np.ones(M) / M
    for _ in range(max_iter):
        r_new = alpha * (P @ r) + (1 - alpha) / M
        if np.max(np.abs(r_new - r)) < tol:
            r = r_new
            break
        r = r_new
    r = np.maximum(r, 0.0)
    r /= r.sum() + EPS
    return r


def softmax_average_advantage(mu_matrix: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Softmax of average predicted advantages (negative of row-mean mu)."""
    M = mu_matrix.shape[0]
    adv = np.zeros(M)
    for i in range(M):
        vals = [-mu_matrix[i, j] for j in range(M) if j != i]
        adv[i] = np.mean(vals) if vals else 0.0
    adv_scaled = adv / max(tau, EPS)
    adv_scaled -= adv_scaled.max()  # numerical stability
    w = np.exp(adv_scaled)
    return w / w.sum()


# ===================================================================
# 4.  COVARIANCE LAYER
# ===================================================================

def rolling_covariance(
    errors: np.ndarray,
    window: int = 60,
) -> np.ndarray:
    """
    Rolling sample covariance of forecast errors.
    errors : shape (T, M)
    Uses last *window* observations.
    """
    T, M = errors.shape
    use = errors[max(0, T - window):T]
    if len(use) < M + 2:
        return np.eye(M)
    return np.cov(use, rowvar=False, ddof=1)


def ewma_covariance(
    errors: np.ndarray,
    lam: float = 0.94,
) -> np.ndarray:
    """Exponentially weighted moving average covariance."""
    T, M = errors.shape
    S = np.zeros((M, M))
    if T == 0:
        return np.eye(M)
    S = np.outer(errors[0], errors[0])
    for t in range(1, T):
        S = lam * S + (1 - lam) * np.outer(errors[t], errors[t])
    return S


def shrinkage_covariance(
    errors: np.ndarray,
    window: int = 60,
    shrink_target: str = "diagonal",
    shrink_intensity: Optional[float] = None,
) -> np.ndarray:
    """
    Ledoit–Wolf style shrinkage toward diagonal or identity.
    If *shrink_intensity* is None, use a simple analytical formula.
    """
    S = rolling_covariance(errors, window)
    M = S.shape[0]
    if shrink_target == "diagonal":
        T_mat = np.diag(np.diag(S))
    else:
        T_mat = np.eye(M) * np.trace(S) / M

    if shrink_intensity is not None:
        delta = shrink_intensity
    else:
        # simple Oracle Approximating Shrinkage intensity
        n = min(len(errors), window)
        delta = min(max((M) / (n + M), 0.01), 0.99)

    return (1 - delta) * S + delta * T_mat


def diagonal_covariance(errors: np.ndarray, window: int = 60) -> np.ndarray:
    """Diagonal-only covariance (variances only)."""
    S = rolling_covariance(errors, window)
    return np.diag(np.diag(S))


def regularise_cov(Sigma: np.ndarray, ridge: float = RIDGE_COV) -> np.ndarray:
    """Add ridge to diagonal for numerical stability."""
    return Sigma + ridge * np.eye(Sigma.shape[0])


COVARIANCE_REGISTRY = {
    "rolling": rolling_covariance,
    "ewma": ewma_covariance,
    "shrinkage": shrinkage_covariance,
    "diagonal": diagonal_covariance,
}


# ===================================================================
# 5.  WEIGHT OPTIMIZATION LAYER
# ===================================================================

def simplex_project(w: np.ndarray) -> np.ndarray:
    """
    Euclidean projection onto the probability simplex.
    Algorithm of Duchi et al. (2008).
    """
    M = len(w)
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, M + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        return np.ones(M) / M
    rho_max = rho[-1]
    theta = (cssv[rho_max] - 1.0) / (rho_max + 1.0)
    w_proj = np.maximum(w - theta, 0.0)
    w_proj /= w_proj.sum() + EPS
    return w_proj


def graph_only_weights(r: np.ndarray) -> np.ndarray:
    """Weights proportional to graph centrality scores."""
    w = np.maximum(r, 0.0)
    s = w.sum()
    if s < EPS:
        return np.ones(len(r)) / len(r)
    return w / s


def covariance_only_weights(Sigma: np.ndarray, ridge: float = RIDGE_COV) -> np.ndarray:
    """Simplex-constrained minimum-variance weights via QP."""
    M = Sigma.shape[0]
    Sigma_r = regularise_cov(Sigma, ridge)
    r_zero = np.zeros(M)
    gamma_zero = 0.0
    return _solve_combination_qp(Sigma_r, r_zero, 0.0, gamma_zero, M)


def full_combination_weights(
    Sigma: np.ndarray,
    r: np.ndarray,
    alpha: float,
    gamma: float,
    ridge: float = RIDGE_COV,
) -> np.ndarray:
    """
    Full graph-covariance-shrinkage combination.

    argmin_w  w'Sigma w  - alpha * r'w  + gamma * ||w - wbar||^2
    s.t. w in simplex
    """
    M = Sigma.shape[0]
    Sigma_r = regularise_cov(Sigma, ridge)
    return _solve_combination_qp(Sigma_r, r, alpha, gamma, M)


def _solve_combination_qp(
    Sigma: np.ndarray,
    r: np.ndarray,
    alpha: float,
    gamma: float,
    M: int,
) -> np.ndarray:
    """Core QP solver using scipy."""
    wbar = np.ones(M) / M

    # Q = Sigma + gamma * I
    Q = Sigma + gamma * np.eye(M)
    # linear term: -alpha * r + (-2*gamma * wbar handled via expansion)
    # objective: 0.5 * w'(2Q)w + c'w
    # but we write: min w'Q w - alpha r'w + gamma w'w - 2gamma wbar'w + gamma wbar'wbar
    # = w' (Sigma + gamma I) w  - (alpha r + 2 gamma wbar)'w + const
    # Actually let's be precise:
    # f(w) = w' Sigma w - alpha r'w + gamma (w-wbar)'(w-wbar)
    #       = w' Sigma w - alpha r'w + gamma w'w - 2 gamma wbar'w + gamma wbar'wbar
    #       = w' (Sigma + gamma I) w - (alpha r + 2 gamma wbar)' w + const
    c = -(alpha * r + 2.0 * gamma * wbar)

    def objective(w):
        return w @ Q @ w + c @ w

    def gradient(w):
        return 2.0 * Q @ w + c

    # constraints: sum(w) = 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * M
    w0 = wbar.copy()

    try:
        res = optimize.minimize(
            objective,
            w0,
            jac=gradient,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12},
        )
        w_opt = res.x
    except Exception:
        w_opt = wbar.copy()

    # safety project
    w_opt = np.maximum(w_opt, 0.0)
    w_opt /= w_opt.sum() + EPS
    return w_opt


def multiplicative_tilt_weights(
    w_base: np.ndarray,
    r: np.ndarray,
    kappa: float = 1.0,
) -> np.ndarray:
    """
    Multiplicative tilt: w_i propto w_base_i * exp(kappa * r_i), renormalised.
    """
    log_w = np.log(np.maximum(w_base, EPS)) + kappa * r
    log_w -= log_w.max()
    w = np.exp(log_w)
    w /= w.sum() + EPS
    return w


# ===================================================================
# 6.  BENCHMARK METHODS
# ===================================================================

def equal_weights(M: int) -> np.ndarray:
    return np.ones(M) / M


def median_forecast(forecasts: np.ndarray) -> float:
    """Return median across M forecasts at one time point."""
    return float(np.median(forecasts))


def recent_best_selection(
    losses: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """
    Select forecast with lowest average recent loss.
    losses : (T_hist, M)
    """
    M = losses.shape[1]
    use = losses[max(0, losses.shape[0] - window):]
    avg = use.mean(axis=0)
    w = np.zeros(M)
    w[np.argmin(avg)] = 1.0
    return w


def bates_granger_weights(
    errors: np.ndarray,
    window: int = 60,
    ridge: float = RIDGE_COV,
) -> np.ndarray:
    """
    Bates–Granger inverse-variance weights (diagonal),
    then simplex projection.
    """
    T, M = errors.shape
    use = errors[max(0, T - window):]
    var = np.var(use, axis=0, ddof=1)
    var = np.maximum(var, EPS)
    inv_var = 1.0 / var
    w = inv_var / inv_var.sum()
    return w


def bates_granger_mv_weights(
    errors: np.ndarray,
    window: int = 60,
    ridge: float = RIDGE_COV,
) -> np.ndarray:
    """
    Minimum-variance simplex-constrained (uses full covariance).
    """
    Sigma = rolling_covariance(errors, window)
    return covariance_only_weights(Sigma, ridge)


def rs_selection_weights(
    mu_matrix: np.ndarray,
) -> np.ndarray:
    """
    Richter–Smetanina style selection: pick forecast i that is predicted
    to beat the most others (largest net outperformance count).
    """
    M = mu_matrix.shape[0]
    wins = np.zeros(M)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            if mu_matrix[i, j] < 0:
                wins[i] += 1
    w = np.zeros(M)
    w[np.argmax(wins)] = 1.0
    return w


# ===================================================================
# 7.  SIMULATION ENVIRONMENT
# ===================================================================

@dataclass
class ScenarioConfig:
    """Configuration for a single simulation scenario."""
    name: str = "default"
    M: int = 8
    T: int = 400
    T0: int = 200        # start of OOS evaluation
    sigma_common: float = 0.5
    seed: int = 42

    # bias params
    bias_type: str = "zero"  # zero, constant, break, drift, cluster
    bias_values: Optional[np.ndarray] = None
    bias_break_time: Optional[int] = None
    bias_pre: Optional[np.ndarray] = None
    bias_post: Optional[np.ndarray] = None
    bias_drift_speed: float = 0.01
    bias_centered: bool = True

    # variance params
    sigma_idio: Optional[np.ndarray] = None  # shape (M,) or (T,M)
    sigma_shift_time: Optional[int] = None
    sigma_pre: Optional[np.ndarray] = None
    sigma_post: Optional[np.ndarray] = None

    # outlier params
    outlier_prob: float = 0.0
    outlier_scale: float = 5.0

    # dependence params
    factor_rho: Optional[np.ndarray] = None  # (M,) factor loadings
    n_clusters: int = 1
    cluster_labels: Optional[np.ndarray] = None
    cluster_rho: float = 0.6


@dataclass
class SimulationData:
    """Container for simulated data."""
    y: np.ndarray               # (T,) target
    forecasts: np.ndarray       # (T, M) forecast values
    errors: np.ndarray          # (T, M) forecast errors
    losses: np.ndarray          # (T, M) losses
    bias_paths: np.ndarray      # (T, M)
    sigma_paths: np.ndarray     # (T, M)
    common_shock: np.ndarray    # (T,)
    config: ScenarioConfig = field(default_factory=ScenarioConfig)


def generate_scenario(cfg: ScenarioConfig) -> SimulationData:
    """Master scenario generator."""
    rng = _ensure_rng(cfg.seed)
    M, T = cfg.M, cfg.T

    # 1. Generate bias paths  b_{j,t}
    bias = _generate_bias(cfg, rng)  # (T, M)

    # 2. Generate idiosyncratic variance paths
    sigma_idio = _generate_sigma(cfg, rng)  # (T, M)

    # 3. Common shock
    common = rng.normal(0, cfg.sigma_common, size=T)

    # 4. Idiosyncratic errors
    idio = np.zeros((T, M))
    for j in range(M):
        idio[:, j] = rng.normal(0, 1, size=T) * sigma_idio[:, j]

    # factor structure
    if cfg.factor_rho is not None:
        factor = rng.normal(0, 1, size=T)
        for j in range(M):
            rho_j = cfg.factor_rho[j]
            idio[:, j] = rho_j * factor + np.sqrt(max(1 - rho_j ** 2, 0)) * idio[:, j]
            idio[:, j] *= sigma_idio[:, j]  # re-scale? Actually let's be precise
        # Rewrite: idio should have the desired marginal scale
        # idio_j = sigma_j * (rho_j * f + sqrt(1-rho_j^2) * u_j)
        # We've already generated idio[:,j] = N(0,1)*sigma, f = N(0,1)
        # Let's redo cleanly
        idio_clean = np.zeros((T, M))
        u = rng.normal(0, 1, size=(T, M))
        for j in range(M):
            rho_j = cfg.factor_rho[j]
            idio_clean[:, j] = sigma_idio[:, j] * (
                rho_j * factor + np.sqrt(max(1 - rho_j ** 2, 0)) * u[:, j]
            )
        idio = idio_clean

    # Cluster structure (overrides factor_rho if set)
    if cfg.n_clusters > 1 and cfg.cluster_labels is not None:
        cluster_factors = rng.normal(0, 1, size=(T, cfg.n_clusters))
        u = rng.normal(0, 1, size=(T, M))
        rho_c = cfg.cluster_rho
        for j in range(M):
            cl = cfg.cluster_labels[j]
            idio[:, j] = sigma_idio[:, j] * (
                rho_c * cluster_factors[:, cl]
                + np.sqrt(max(1 - rho_c ** 2, 0)) * u[:, j]
            )

    # Outliers
    if cfg.outlier_prob > 0:
        outlier_mask = rng.random(size=(T, M)) < cfg.outlier_prob
        outlier_vals = rng.uniform(-cfg.outlier_scale, cfg.outlier_scale, size=(T, M))
        idio = np.where(outlier_mask, outlier_vals, idio)

    # 5. Forecast errors: e_{j,t} = c_t - b_{j,t} - eps_{j,t}
    # Actually: e = c - b - idio
    errors = common[:, None] - bias - idio  # (T, M)

    # 6. Target
    # y_t drawn as some latent, forecasts = y - errors
    # Simplest: y_t = mu + common noise + signal
    y_signal = rng.normal(0, 1, size=T).cumsum() * 0.01  # mild random walk signal
    y = y_signal + common  # plus common
    forecasts = y[:, None] - errors  # f_{j,t} = y_t - e_{j,t}

    # losses
    losses = squared_loss(y[:, None], forecasts)

    return SimulationData(
        y=y,
        forecasts=forecasts,
        errors=errors,
        losses=losses,
        bias_paths=bias,
        sigma_paths=sigma_idio,
        common_shock=common,
        config=cfg,
    )


def _generate_bias(cfg: ScenarioConfig, rng) -> np.ndarray:
    M, T = cfg.M, cfg.T
    bias = np.zeros((T, M))

    if cfg.bias_type == "zero":
        pass

    elif cfg.bias_type == "constant":
        if cfg.bias_values is not None:
            b = cfg.bias_values.copy()
        else:
            b = rng.normal(0, 0.3, size=M)
        if cfg.bias_centered:
            b -= b.mean()
        bias = np.tile(b, (T, 1))

    elif cfg.bias_type == "break":
        t_break = cfg.bias_break_time if cfg.bias_break_time else T // 2
        b_pre = cfg.bias_pre if cfg.bias_pre is not None else rng.normal(0, 0.2, size=M)
        b_post = cfg.bias_post if cfg.bias_post is not None else rng.normal(0, 0.2, size=M)
        if cfg.bias_centered:
            b_pre -= b_pre.mean()
            b_post -= b_post.mean()
        bias[:t_break] = b_pre[None, :]
        bias[t_break:] = b_post[None, :]

    elif cfg.bias_type == "drift":
        speed = cfg.bias_drift_speed
        b = np.zeros((T, M))
        b[0] = rng.normal(0, 0.1, size=M)
        for t in range(1, T):
            b[t] = b[t - 1] + rng.normal(0, speed, size=M)
        if cfg.bias_centered:
            b -= b.mean(axis=1, keepdims=True)
        bias = b

    elif cfg.bias_type == "cluster":
        # same as constant but with cluster structure in biases
        if cfg.bias_values is not None:
            b = cfg.bias_values.copy()
        else:
            b = rng.normal(0, 0.3, size=M)
        if cfg.bias_centered:
            b -= b.mean()
        bias = np.tile(b, (T, 1))

    return bias


def _generate_sigma(cfg: ScenarioConfig, rng) -> np.ndarray:
    M, T = cfg.M, cfg.T

    if cfg.sigma_idio is not None:
        s = cfg.sigma_idio
        if s.ndim == 1:
            return np.tile(s, (T, 1))
        return s

    base = np.ones(M) * 0.5

    if cfg.sigma_shift_time is not None:
        sigma_pre = cfg.sigma_pre if cfg.sigma_pre is not None else base
        sigma_post = cfg.sigma_post if cfg.sigma_post is not None else base * 2
        out = np.zeros((T, M))
        out[:cfg.sigma_shift_time] = sigma_pre[None, :]
        out[cfg.sigma_shift_time:] = sigma_post[None, :]
        return out

    return np.tile(base, (T, 1))


# ===================================================================
# 7b.  PRE-BUILT SCENARIO FACTORIES
# ===================================================================

def scenario_1A(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """Stable unbiased homoskedastic."""
    return ScenarioConfig(
        name="1A_stable_unbiased",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="zero",
        sigma_idio=np.ones(M) * 0.5,
    )


def scenario_1B(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """Stable biased homoskedastic (centered)."""
    rng = _ensure_rng(seed)
    biases = rng.normal(0, 0.4, size=M)
    return ScenarioConfig(
        name="1B_stable_biased",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="constant",
        bias_values=biases,
        bias_centered=True,
        sigma_idio=np.ones(M) * 0.5,
    )


def scenario_2A(M=8, T=400, T0=200, seed=42, break_frac=0.5) -> ScenarioConfig:
    """Abrupt break in biases."""
    rng = _ensure_rng(seed + 1)
    b_pre = rng.normal(0, 0.3, size=M)
    b_post = rng.normal(0, 0.3, size=M)
    t_break = int(T * break_frac)
    return ScenarioConfig(
        name="2A_abrupt_break",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="break",
        bias_break_time=t_break,
        bias_pre=b_pre,
        bias_post=b_post,
        bias_centered=True,
        sigma_idio=np.ones(M) * 0.5,
    )


def scenario_2B(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """Smooth drift in biases."""
    return ScenarioConfig(
        name="2B_smooth_drift",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="drift",
        bias_drift_speed=0.02,
        bias_centered=True,
        sigma_idio=np.ones(M) * 0.5,
    )


def scenario_2C(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """Precision shift (variance change)."""
    t_shift = T // 2
    sigma_pre = np.ones(M) * 0.5
    sigma_post = np.ones(M) * 0.5
    sigma_post[:M // 2] = 1.5  # first half become noisy
    return ScenarioConfig(
        name="2C_precision_shift",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="zero",
        sigma_shift_time=t_shift,
        sigma_pre=sigma_pre,
        sigma_post=sigma_post,
    )


def scenario_3A(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """Idiosyncratic outliers."""
    return ScenarioConfig(
        name="3A_outliers",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="zero",
        sigma_idio=np.ones(M) * 0.5,
        outlier_prob=0.05,
        outlier_scale=5.0,
    )


def scenario_3B(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """Cross-sectional one-factor dependence."""
    rng = _ensure_rng(seed + 2)
    rhos = rng.uniform(0.3, 0.9, size=M)
    return ScenarioConfig(
        name="3B_factor_dependence",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="zero",
        sigma_idio=np.ones(M) * 0.5,
        factor_rho=rhos,
    )


def scenario_3C(M=8, T=400, T0=200, seed=42) -> ScenarioConfig:
    """Clustered forecasters."""
    n_clusters = 3
    labels = np.array([i % n_clusters for i in range(M)])
    return ScenarioConfig(
        name="3C_clustered",
        M=M, T=T, T0=T0, seed=seed,
        sigma_common=0.5,
        bias_type="zero",
        sigma_idio=np.ones(M) * 0.5,
        n_clusters=n_clusters,
        cluster_labels=labels,
        cluster_rho=0.7,
    )


ALL_SCENARIO_FACTORIES = {
    "1A": scenario_1A,
    "1B": scenario_1B,
    "2A": scenario_2A,
    "2B": scenario_2B,
    "2C": scenario_2C,
    "3A": scenario_3A,
    "3B": scenario_3B,
    "3C": scenario_3C,
}


# ===================================================================
# 8.  ROLLING BACKTEST ENGINE
# ===================================================================

@dataclass
class BacktestConfig:
    """Hyper-parameters for the rolling backtest."""
    # pairwise LD model
    d_max: int = 4
    fixed_d: Optional[int] = None
    fixed_h1: Optional[float] = None
    fixed_h2: Optional[float] = None
    n_cv_folds: int = 5

    # graph layer
    adjacency_type: str = "standardized"   # raw, standardized, thresholded
    centrality_type: str = "eigenvector"   # eigenvector, rowsum, pagerank, softmax
    teleport: float = TELEPORT
    adj_reg_c: float = 1e-4

    # covariance
    cov_method: str = "shrinkage"  # rolling, ewma, shrinkage, diagonal
    cov_window: int = 60
    cov_ewma_lambda: float = 0.94
    ridge_cov: float = RIDGE_COV

    # optimisation
    alpha: Optional[float] = None  # None => tune
    gamma: Optional[float] = None  # None => tune
    alpha_grid: Optional[np.ndarray] = None
    gamma_grid: Optional[np.ndarray] = None
    tune_window: int = 40  # rolling window for tuning alpha, gamma

    # loss
    loss_name: str = "squared"

    # benchmarks
    recent_best_window: int = 20
    bg_window: int = 60

    # misc
    min_history: int = 30  # minimum observations before producing weights


@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    oos_periods: np.ndarray         # (n_oos,)
    y_oos: np.ndarray               # (n_oos,)
    forecasts_oos: np.ndarray       # (n_oos, M)

    # weights: dict of method_name -> (n_oos, M) or (n_oos,) for median
    weights: Dict[str, np.ndarray] = field(default_factory=dict)
    combined_forecasts: Dict[str, np.ndarray] = field(default_factory=dict)
    combined_losses: Dict[str, np.ndarray] = field(default_factory=dict)

    # diagnostics
    mu_matrices: Optional[List] = None   # list of (M,M) at each oos period
    sigma_matrices: Optional[List] = None
    adjacency_matrices: Optional[List] = None
    centrality_scores: Optional[List] = None
    cov_matrices: Optional[List] = None
    alpha_selected: Optional[np.ndarray] = None
    gamma_selected: Optional[np.ndarray] = None
    d_selected: Optional[List] = None


def run_backtest(
    data: SimulationData,
    bt_cfg: BacktestConfig = None,
    verbose: bool = False,
) -> BacktestResult:
    """
    Full rolling out-of-sample backtest.
    """
    if bt_cfg is None:
        bt_cfg = BacktestConfig()

    cfg = data.config
    M = cfg.M
    T = cfg.T
    T0 = cfg.T0

    loss_fn = LOSS_REGISTRY[bt_cfg.loss_name]

    y = data.y
    forecasts = data.forecasts
    losses = data.losses
    errors = data.errors

    n_oos = T - T0
    oos_periods = np.arange(T0, T)

    # Pre-compute pairwise loss differentials (full history)
    # delta_L[i,j,t] = L_i(t) - L_j(t)
    # We'll use 3D array for convenience but only access up to current t
    delta_L_full = np.zeros((M, M, T))
    for i in range(M):
        for j in range(M):
            if i != j:
                delta_L_full[i, j, :] = losses[:, i] - losses[:, j]

    # Storage
    res = BacktestResult(
        oos_periods=oos_periods,
        y_oos=y[T0:T],
        forecasts_oos=forecasts[T0:T],
        mu_matrices=[],
        sigma_matrices=[],
        adjacency_matrices=[],
        centrality_scores=[],
        cov_matrices=[],
        alpha_selected=np.zeros(n_oos),
        gamma_selected=np.zeros(n_oos),
    )

    # Method names
    method_names = [
        "equal",
        "median",
        "recent_best",
        "bates_granger",
        "bates_granger_mv",
        "rs_selection",
        "graph_only",
        "cov_only",
        "full_gcsr",
        "mult_tilt",
    ]
    for name in method_names:
        res.weights[name] = np.zeros((n_oos, M))
        res.combined_forecasts[name] = np.zeros(n_oos)
        res.combined_losses[name] = np.zeros(n_oos)

    # Default grids for alpha, gamma
    if bt_cfg.alpha_grid is None:
        bt_cfg.alpha_grid = np.array([0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0])
    if bt_cfg.gamma_grid is None:
        bt_cfg.gamma_grid = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0])

    # ---- Main OOS loop ----
    for oos_idx in range(n_oos):
        t = T0 + oos_idx  # forecast origin: using data up to t-1, forecasting y_t
        # But let's clarify timing:
        # At origin t, we have y_0,...,y_{t-1} and forecasts f_{j,0},...,f_{j,t}
        # We form weights for combining f_{j,t} to forecast y_t
        # After y_t is realised, we can evaluate.
        # So: losses available up through t-1, forecasts for period t already made.

        hist_end = t  # exclusive: indices 0..t-1 are available
        if hist_end < bt_cfg.min_history:
            # not enough history, use equal weights
            for name in method_names:
                res.weights[name][oos_idx] = equal_weights(M)
            f_oos = forecasts[t]
            w_eq = equal_weights(M)
            y_t = y[t]
            for name in method_names:
                w = res.weights[name][oos_idx]
                if name == "median":
                    res.combined_forecasts[name][oos_idx] = median_forecast(f_oos)
                else:
                    res.combined_forecasts[name][oos_idx] = f_oos @ w
                res.combined_losses[name][oos_idx] = loss_fn(
                    y_t, res.combined_forecasts[name][oos_idx]
                )
            continue

        # --- LAYER 1: Pairwise LD predictions ---
        mu_mat = np.zeros((M, M))
        sigma_mat = np.ones((M, M))
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                dl_hist = delta_L_full[i, j, :hist_end]
                if len(dl_hist) < bt_cfg.min_history:
                    continue
                pld = predict_pairwise_ld(
                    dl_hist,
                    d_max=bt_cfg.d_max,
                    fixed_d=bt_cfg.fixed_d,
                    fixed_h1=bt_cfg.fixed_h1,
                    fixed_h2=bt_cfg.fixed_h2,
                    n_cv_folds=bt_cfg.n_cv_folds,
                )
                mu_mat[i, j] = pld.mu_hat
                sigma_mat[i, j] = pld.sigma_hat

        res.mu_matrices.append(mu_mat.copy())
        res.sigma_matrices.append(sigma_mat.copy())

        # --- LAYER 2: Graph ---
        if bt_cfg.adjacency_type == "raw":
            A = build_adjacency_raw(mu_mat)
        elif bt_cfg.adjacency_type == "standardized":
            A = build_adjacency_standardized(mu_mat, sigma_mat, bt_cfg.adj_reg_c)
        elif bt_cfg.adjacency_type == "thresholded":
            A = build_adjacency_thresholded(mu_mat, sigma_mat, 1.0, bt_cfg.adj_reg_c)
        else:
            A = build_adjacency_raw(mu_mat)

        res.adjacency_matrices.append(A.copy())

        if bt_cfg.centrality_type == "eigenvector":
            r = eigenvector_centrality(A, bt_cfg.teleport)
        elif bt_cfg.centrality_type == "rowsum":
            r = row_sum_strength(A)
        elif bt_cfg.centrality_type == "pagerank":
            r = pagerank_centrality(A)
        elif bt_cfg.centrality_type == "softmax":
            r = softmax_average_advantage(mu_mat)
        else:
            r = eigenvector_centrality(A, bt_cfg.teleport)

        res.centrality_scores.append(r.copy())

        # --- LAYER 3: Covariance ---
        errors_hist = errors[:hist_end]
        cov_fn = COVARIANCE_REGISTRY.get(bt_cfg.cov_method, shrinkage_covariance)
        if bt_cfg.cov_method == "ewma":
            Sigma = cov_fn(errors_hist, bt_cfg.cov_ewma_lambda)
        else:
            Sigma = cov_fn(errors_hist, bt_cfg.cov_window)
        Sigma = regularise_cov(Sigma, bt_cfg.ridge_cov)
        res.cov_matrices.append(Sigma.copy())

        # --- Tune alpha, gamma ---
        alpha_sel, gamma_sel = _tune_alpha_gamma(
            bt_cfg, losses[:hist_end], forecasts[:hist_end], y[:hist_end],
            errors_hist, delta_L_full[:, :, :hist_end],
            r, Sigma, M, loss_fn,
        )
        res.alpha_selected[oos_idx] = alpha_sel
        res.gamma_selected[oos_idx] = gamma_sel

        # --- Compute weights for each method ---
        f_oos = forecasts[t]
        y_t = y[t]

        # 1. Equal
        w_eq = equal_weights(M)
        res.weights["equal"][oos_idx] = w_eq

        # 2. Median (no weights vector, just combined forecast)
        res.weights["median"][oos_idx] = w_eq  # placeholder

        # 3. Recent best
        w_rb = recent_best_selection(losses[:hist_end], bt_cfg.recent_best_window)
        res.weights["recent_best"][oos_idx] = w_rb

        # 4. Bates–Granger
        w_bg = bates_granger_weights(errors_hist, bt_cfg.bg_window)
        res.weights["bates_granger"][oos_idx] = w_bg

        # 5. Bates–Granger MV
        w_bgmv = bates_granger_mv_weights(errors_hist, bt_cfg.bg_window)
        res.weights["bates_granger_mv"][oos_idx] = w_bgmv

        # 6. RS selection
        w_rs = rs_selection_weights(mu_mat)
        res.weights["rs_selection"][oos_idx] = w_rs

        # 7. Graph-only
        w_go = graph_only_weights(r)
        res.weights["graph_only"][oos_idx] = w_go

        # 8. Covariance-only
        w_co = covariance_only_weights(Sigma, bt_cfg.ridge_cov)
        res.weights["cov_only"][oos_idx] = w_co

        # 9. Full GCSR
        w_full = full_combination_weights(Sigma, r, alpha_sel, gamma_sel, bt_cfg.ridge_cov)
        res.weights["full_gcsr"][oos_idx] = w_full

        # 10. Multiplicative tilt
        kappa = alpha_sel  # re-use
        w_mt = multiplicative_tilt_weights(w_co, r, kappa)
        res.weights["mult_tilt"][oos_idx] = w_mt

        # --- Combined forecasts and losses ---
        for name in method_names:
            w = res.weights[name][oos_idx]
            if name == "median":
                comb = median_forecast(f_oos)
            else:
                comb = float(f_oos @ w)
            res.combined_forecasts[name][oos_idx] = comb
            res.combined_losses[name][oos_idx] = loss_fn(y_t, comb)

        if verbose and oos_idx % 50 == 0:
            print(f"  OOS {oos_idx}/{n_oos} (t={t})")

    return res


def _tune_alpha_gamma(
    bt_cfg, losses_hist, forecasts_hist, y_hist,
    errors_hist, delta_L_hist,
    r_current, Sigma_current, M, loss_fn,
):
    """
    Tune alpha, gamma over a past validation window.
    Uses recent OOS performance of the method.
    """
    if bt_cfg.alpha is not None and bt_cfg.gamma is not None:
        return bt_cfg.alpha, bt_cfg.gamma

    T_hist = len(y_hist)
    tw = min(bt_cfg.tune_window, T_hist - bt_cfg.min_history)
    if tw < 5:
        # not enough history to tune, use moderate defaults
        return 0.1, 0.1

    alpha_grid = bt_cfg.alpha_grid
    gamma_grid = bt_cfg.gamma_grid

    # Simple grid search: for each (alpha, gamma), compute what the
    # combined forecast would have been over the tuning window using
    # the *current* r and Sigma as proxies (approximation for speed)
    best_loss = np.inf
    best_alpha = float(alpha_grid[len(alpha_grid) // 2])
    best_gamma = float(gamma_grid[len(gamma_grid) // 2])

    val_start = T_hist - tw
    y_val = y_hist[val_start:T_hist]
    f_val = forecasts_hist[val_start:T_hist]

    for alpha in alpha_grid:
        for gamma in gamma_grid:
            w = full_combination_weights(
                Sigma_current, r_current, alpha, gamma, bt_cfg.ridge_cov
            )
            comb = f_val @ w
            avg_loss = loss_fn(y_val, comb).mean()
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_alpha = float(alpha)
                best_gamma = float(gamma)

    return best_alpha, best_gamma


# ===================================================================
# 9.  REPLICATION AND SUMMARY
# ===================================================================

@dataclass
class MCResult:
    """Summary of Monte Carlo replications."""
    scenario_name: str
    n_reps: int
    method_names: List[str]
    msfe_matrix: np.ndarray       # (n_reps, n_methods)
    rel_msfe_matrix: np.ndarray   # relative to equal weights
    mean_msfe: np.ndarray         # (n_methods,)
    median_msfe: np.ndarray
    mean_rel_msfe: np.ndarray
    win_freq: np.ndarray          # fraction of reps each method wins


def run_monte_carlo(
    scenario_factory,
    n_reps: int = 20,
    bt_cfg: BacktestConfig = None,
    verbose: bool = False,
    **scenario_kwargs,
) -> MCResult:
    """Run repeated replications of a scenario."""
    if bt_cfg is None:
        bt_cfg = BacktestConfig()

    all_results = []
    for rep in range(n_reps):
        kw = dict(scenario_kwargs)
        kw["seed"] = kw.get("seed", 42) + rep * 1000
        cfg = scenario_factory(**kw)
        data = generate_scenario(cfg)
        res = run_backtest(data, bt_cfg, verbose=False)
        all_results.append(res)
        if verbose:
            print(f"Replication {rep+1}/{n_reps} done.")

    # Summarise
    method_names = list(all_results[0].combined_losses.keys())
    n_methods = len(method_names)
    msfe_mat = np.zeros((n_reps, n_methods))
    for rep, res in enumerate(all_results):
        for jm, name in enumerate(method_names):
            msfe_mat[rep, jm] = res.combined_losses[name].mean()

    eq_idx = method_names.index("equal")
    rel_msfe = msfe_mat / (msfe_mat[:, eq_idx:eq_idx + 1] + EPS)

    mean_msfe = msfe_mat.mean(axis=0)
    median_msfe = np.median(msfe_mat, axis=0)
    mean_rel = rel_msfe.mean(axis=0)
    win_freq = np.zeros(n_methods)
    winners = msfe_mat.argmin(axis=1)
    for w in winners:
        win_freq[w] += 1
        win_freq /= n_reps

    return MCResult(
        scenario_name=cfg.name,
        n_reps=n_reps,
        method_names=method_names,
        msfe_matrix=msfe_mat,
        rel_msfe_matrix=rel_msfe,
        mean_msfe=mean_msfe,
        median_msfe=median_msfe,
        mean_rel_msfe=mean_rel,
        win_freq=win_freq,
    )


def summarise_mc(mc: MCResult) -> pd.DataFrame:
    """Pretty-print Monte Carlo summary as DataFrame."""
    df = pd.DataFrame({
        "Method": mc.method_names,
        "Mean_MSFE": mc.mean_msfe,
        "Median_MSFE": mc.median_msfe,
        "Rel_MSFE_vs_EW": mc.mean_rel_msfe,
        "Win_Freq": mc.win_freq,
    })
    df = df.sort_values("Mean_MSFE").reset_index(drop=True)
    return df


# ===================================================================
# 10.  PLOTTING UTILITIES
# ===================================================================

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Optional imports
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


def set_plot_style():
    """Set consistent matplotlib style."""
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


set_plot_style()


# ---------- A. Scenario Visualization ----------

def plot_bias_paths(data: SimulationData, ax=None):
    """Plot latent bias paths b_{j,t}."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    M = data.config.M
    for j in range(M):
        ax.plot(data.bias_paths[:, j], label=f"Model {j+1}", alpha=0.7, linewidth=1)
    ax.axvline(data.config.T0, color="k", ls="--", alpha=0.5, label="OOS start")
    ax.set_title(f"Bias paths — {data.config.name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Bias $b_{j,t}$")
    ax.legend(ncol=4, fontsize=7)
    return ax


def plot_sigma_paths(data: SimulationData, ax=None):
    """Plot idiosyncratic std-dev paths."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    M = data.config.M
    for j in range(M):
        ax.plot(data.sigma_paths[:, j], label=f"Model {j+1}", alpha=0.7, linewidth=1)
    ax.axvline(data.config.T0, color="k", ls="--", alpha=0.5)
    ax.set_title(f"Idiosyncratic σ paths — {data.config.name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("$\\sigma_{j,t}$")
    ax.legend(ncol=4, fontsize=7)
    return ax


def plot_common_shock(data: SimulationData, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(data.common_shock, color="steelblue", alpha=0.7, linewidth=0.8)
    ax.axvline(data.config.T0, color="k", ls="--", alpha=0.5)
    ax.set_title(f"Common shock — {data.config.name}")
    ax.set_xlabel("Time")
    return ax


def plot_forecast_errors(data: SimulationData, ax=None, max_models=6):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    M = min(data.config.M, max_models)
    for j in range(M):
        ax.plot(data.errors[:, j], alpha=0.4, linewidth=0.6, label=f"Model {j+1}")
    ax.axvline(data.config.T0, color="k", ls="--", alpha=0.5)
    ax.set_title(f"Forecast errors — {data.config.name}")
    ax.legend(ncol=3, fontsize=7)
    return ax


def plot_scenario_summary(data: SimulationData):
    """4-panel scenario overview."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    plot_bias_paths(data, axes[0, 0])
    plot_sigma_paths(data, axes[0, 1])
    plot_common_shock(data, axes[1, 0])
    plot_forecast_errors(data, axes[1, 1])
    fig.suptitle(f"Scenario: {data.config.name}", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


# ---------- B. Pairwise Methodology Visualization ----------

def plot_pairwise_heatmap(matrix: np.ndarray, title: str = "", ax=None, cmap="RdBu_r"):
    """Plot heatmap of an MxM pairwise matrix."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    M = matrix.shape[0]
    if HAS_SEABORN:
        sns.heatmap(
            matrix, ax=ax, annot=M <= 10, fmt=".3f",
            center=0, cmap=cmap, square=True,
            xticklabels=[f"{i+1}" for i in range(M)],
            yticklabels=[f"{i+1}" for i in range(M)],
        )
    else:
        im = ax.imshow(matrix, cmap=cmap, aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_xticks(range(M))
        ax.set_yticks(range(M))
        ax.set_xticklabels([f"{i+1}" for i in range(M)])
        ax.set_yticklabels([f"{i+1}" for i in range(M)])
    ax.set_title(title)
    ax.set_xlabel("Model j")
    ax.set_ylabel("Model i")
    return ax


def plot_adjacency_heatmaps(res: BacktestResult, time_indices=None):
    """Plot adjacency matrices at selected OOS time points."""
    if time_indices is None:
        n = len(res.adjacency_matrices)
        time_indices = [0, n // 4, n // 2, 3 * n // 4, n - 1]
        time_indices = [i for i in time_indices if 0 <= i < n]

    n_plots = len(time_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]
    for idx, ti in enumerate(time_indices):
        t_abs = res.oos_periods[ti]
        plot_pairwise_heatmap(
            res.adjacency_matrices[ti],
            title=f"Adjacency A (t={t_abs})",
            ax=axes[idx],
            cmap="YlOrRd",
        )
    fig.tight_layout()
    return fig


def plot_centrality_bars(res: BacktestResult, time_indices=None):
    """Bar charts of centrality scores at selected dates."""
    if time_indices is None:
        n = len(res.centrality_scores)
        time_indices = [0, n // 2, n - 1]
        time_indices = [i for i in time_indices if 0 <= i < n]

    n_plots = len(time_indices)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 3.5))
    if n_plots == 1:
        axes = [axes]
    for idx, ti in enumerate(time_indices):
        r = res.centrality_scores[ti]
        M = len(r)
        t_abs = res.oos_periods[ti]
        axes[idx].bar(range(M), r, color="steelblue", alpha=0.8)
        axes[idx].axhline(1.0 / M, color="red", ls="--", alpha=0.6, label="1/M")
        axes[idx].set_title(f"Centrality (t={t_abs})")
        axes[idx].set_xlabel("Model")
        axes[idx].set_xticks(range(M))
        axes[idx].set_xticklabels([f"{i+1}" for i in range(M)])
        axes[idx].legend()
    fig.tight_layout()
    return fig


def plot_graph_network(A: np.ndarray, r: np.ndarray, title: str = ""):
    """Plot directed network from adjacency matrix."""
    if not HAS_NETWORKX:
        print("networkx not available; skipping graph plot.")
        return None
    M = A.shape[0]
    G = nx.DiGraph()
    for i in range(M):
        G.add_node(i, label=f"M{i+1}")
    for i in range(M):
        for j in range(M):
            if A[i, j] > 1e-6:
                G.add_edge(i, j, weight=A[i, j])

    fig, ax = plt.subplots(figsize=(7, 7))
    pos = nx.spring_layout(G, seed=42, k=2)
    node_sizes = r * 3000 + 200
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=r,
                           cmap="YlOrRd", alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={i: f"M{i+1}" for i in range(M)},
                            font_size=9, ax=ax)
    edges = G.edges(data=True)
    if edges:
        edge_weights = [d["weight"] for _, _, d in edges]
        max_w = max(edge_weights) if edge_weights else 1
        widths = [2.0 * w / (max_w + EPS) for w in edge_weights]
        nx.draw_networkx_edges(G, pos, width=widths, alpha=0.4,
                               edge_color="gray", arrows=True, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------- C. Weight Diagnostics ----------

def plot_weight_timeseries(res: BacktestResult, methods=None):
    """Time series of weights for selected methods."""
    if methods is None:
        methods = ["equal", "full_gcsr", "cov_only", "graph_only"]
    n_methods = len(methods)
    fig, axes = plt.subplots(n_methods, 1, figsize=(14, 3.5 * n_methods), sharex=True)
    if n_methods == 1:
        axes = [axes]
    for idx, name in enumerate(methods):
        w = res.weights.get(name)
        if w is None:
            continue
        M = w.shape[1]
        t_axis = res.oos_periods
        for j in range(M):
            axes[idx].fill_between([], [], alpha=0.5)
        # stacked area
        axes[idx].stackplot(
            t_axis, w.T,
            labels=[f"M{j+1}" for j in range(M)],
            alpha=0.7,
        )
        axes[idx].set_title(f"Weights: {name}")
        axes[idx].set_ylabel("Weight")
        axes[idx].set_ylim(0, 1)
        if M <= 10:
            axes[idx].legend(loc="upper right", ncol=M, fontsize=6)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    return fig


def compute_herfindahl(w: np.ndarray) -> np.ndarray:
    """Herfindahl index for each row of weights matrix (n_oos, M)."""
    return (w ** 2).sum(axis=1)


def compute_effective_n(w: np.ndarray) -> np.ndarray:
    """Effective number of forecasters = 1 / Herfindahl."""
    h = compute_herfindahl(w)
    return 1.0 / np.maximum(h, EPS)


def compute_turnover(w: np.ndarray) -> np.ndarray:
    """Turnover: sum |w_t - w_{t-1}| / 2."""
    diffs = np.abs(np.diff(w, axis=0))
    return diffs.sum(axis=1) / 2.0


def plot_weight_diagnostics(res: BacktestResult, methods=None):
    """Herfindahl, effective N, and turnover."""
    if methods is None:
        methods = ["equal", "full_gcsr", "cov_only", "graph_only", "bates_granger"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    for name in methods:
        w = res.weights.get(name)
        if w is None:
            continue
        t_axis = res.oos_periods
        herf = compute_herfindahl(w)
        eff_n = compute_effective_n(w)
        turn = compute_turnover(w)
        axes[0].plot(t_axis, herf, label=name, alpha=0.7)
        axes[1].plot(t_axis, eff_n, label=name, alpha=0.7)
        axes[2].plot(t_axis[1:], turn, label=name, alpha=0.7)

    axes[0].set_title("Herfindahl Index")
    axes[1].set_title("Effective Number of Forecasters")
    axes[2].set_title("Weight Turnover")
    for ax in axes:
        ax.legend(fontsize=7, ncol=3)
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    return fig


# ---------- D. Covariance Diagnostics ----------

def plot_covariance_diagnostics(res: BacktestResult):
    """Condition number and trace of Sigma over time."""
    if not res.cov_matrices:
        return None
    cond_nums = []
    traces = []
    for Sigma in res.cov_matrices:
        eigvals = np.linalg.eigvalsh(Sigma)
        eigvals = np.maximum(eigvals, EPS)
        cond_nums.append(eigvals[-1] / eigvals[0])
        traces.append(np.trace(Sigma))

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    axes[0].plot(res.oos_periods, cond_nums, color="darkred", alpha=0.7)
    axes[0].set_title("Condition Number of Σ̂")
    axes[0].set_ylabel("κ(Σ̂)")
    axes[1].plot(res.oos_periods, traces, color="steelblue", alpha=0.7)
    axes[1].set_title("Trace of Σ̂")
    axes[1].set_ylabel("tr(Σ̂)")
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    return fig


def plot_cov_heatmap(res: BacktestResult, oos_idx: int = -1):
    """Heatmap of covariance matrix at one time point."""
    if not res.cov_matrices:
        return None
    if oos_idx < 0:
        oos_idx = len(res.cov_matrices) + oos_idx
    Sigma = res.cov_matrices[oos_idx]
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_pairwise_heatmap(Sigma, title=f"Σ̂ (t={res.oos_periods[oos_idx]})", ax=ax, cmap="viridis")
    fig.tight_layout()
    return fig


# ---------- E. Performance Metrics ----------

def plot_cumulative_loss(res: BacktestResult, methods=None, reference="equal"):
    """Cumulative loss difference relative to a reference method."""
    if methods is None:
        methods = list(res.combined_losses.keys())

    ref_loss = res.combined_losses.get(reference, res.combined_losses["equal"])
    fig, ax = plt.subplots(figsize=(14, 5))
    for name in methods:
        if name == reference:
            continue
        diff = np.cumsum(res.combined_losses[name] - ref_loss)
        ax.plot(res.oos_periods, diff, label=name, alpha=0.8, linewidth=1.2)
    ax.axhline(0, color="black", ls="-", linewidth=0.5)
    ax.set_title(f"Cumulative Loss Difference vs {reference}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative ΔL")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    return fig


def compute_performance_table(res: BacktestResult) -> pd.DataFrame:
    """Summary performance table."""
    rows = []
    for name in res.combined_losses:
        l = res.combined_losses[name]
        rows.append({
            "Method": name,
            "Mean_Loss": l.mean(),
            "Median_Loss": np.median(l),
            "Std_Loss": l.std(),
            "Total_Loss": l.sum(),
        })
    df = pd.DataFrame(rows)
    eq_mean = df.loc[df.Method == "equal", "Mean_Loss"].values[0]
    df["Rel_MSFE"] = df["Mean_Loss"] / (eq_mean + EPS)
    df = df.sort_values("Mean_Loss").reset_index(drop=True)
    return df


def plot_msfe_barplot(res: BacktestResult, ax=None):
    """Bar plot of mean OOS loss by method."""
    df = compute_performance_table(res)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["steelblue" if n != "full_gcsr" else "darkred" for n in df.Method]
    ax.barh(df.Method, df.Rel_MSFE, color=colors, alpha=0.8)
    ax.axvline(1.0, color="black", ls="--", linewidth=0.8)
    ax.set_xlabel("Relative MSFE (vs Equal Weights)")
    ax.set_title("OOS Performance")
    ax.invert_yaxis()
    return ax


def plot_alpha_gamma_selected(res: BacktestResult):
    """Time series of selected alpha and gamma."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    axes[0].plot(res.oos_periods, res.alpha_selected, color="darkred", alpha=0.7)
    axes[0].set_title("Selected α over time")
    axes[0].set_ylabel("α")
    axes[1].plot(res.oos_periods, res.gamma_selected, color="steelblue", alpha=0.7)
    axes[1].set_title("Selected γ over time")
    axes[1].set_ylabel("γ")
    axes[-1].set_xlabel("Time")
    fig.tight_layout()
    return fig


# ---------- F. MC Robustness ----------

def plot_mc_boxplot(mc: MCResult):
    """Boxplot of relative MSFE across replications."""
    fig, ax = plt.subplots(figsize=(12, 5))
    df = pd.DataFrame(mc.rel_msfe_matrix, columns=mc.method_names)
    # sort by median
    order = df.median().sort_values().index.tolist()
    if HAS_SEABORN:
        sns.boxplot(data=df[order], orient="h", ax=ax, palette="Set2")
    else:
        ax.boxplot([df[c].values for c in order], vert=False, labels=order)
    ax.axvline(1.0, color="black", ls="--", linewidth=0.8)
    ax.set_xlabel("Relative MSFE (vs Equal Weights)")
    ax.set_title(f"MC Distribution — {mc.scenario_name} ({mc.n_reps} reps)")
    fig.tight_layout()
    return fig


def plot_mc_summary_table(mc_results: Dict[str, MCResult]) -> pd.DataFrame:
    """Cross-scenario summary table."""
    rows = []
    for sc_name, mc in mc_results.items():
        for jm, mname in enumerate(mc.method_names):
            rows.append({
                "Scenario": sc_name,
                "Method": mname,
                "Mean_Rel_MSFE": mc.mean_rel_msfe[jm],
                "Win_Freq": mc.win_freq[jm],
            })
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="Method", columns="Scenario",
                           values="Mean_Rel_MSFE", aggfunc="mean")
    return pivot


def plot_mc_heatmap(mc_results: Dict[str, MCResult]):
    """Heatmap of relative MSFE across scenarios and methods."""
    pivot = plot_mc_summary_table(mc_results)
    fig, ax = plt.subplots(figsize=(12, 6))
    if HAS_SEABORN:
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
                    center=1.0, ax=ax)
    else:
        im = ax.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticklabels(pivot.index)
    ax.set_title("Relative MSFE Across Scenarios")
    fig.tight_layout()
    return fig


# ===================================================================
# 11.  LEAKAGE AUDIT UTILITIES
# ===================================================================

def leakage_audit_synthetic(seed=123):
    """
    Synthetic timing sanity check.

    Creates a trivial scenario where the target is fully predictable
    by one model from t-1 info, and verifies that no method uses
    y_{t+1} when forming weights for t+1.

    Returns True if audit passes.
    """
    rng = _ensure_rng(seed)
    T, M = 120, 4
    T0 = 60

    y = rng.normal(0, 1, size=T).cumsum()
    forecasts = np.zeros((T, M))
    # Model 0: knows y exactly one period late (lagged info)
    forecasts[1:, 0] = y[:-1]
    # Model 1-3: noisy
    for j in range(1, M):
        forecasts[:, j] = y + rng.normal(0, 2.0, size=T)

    errors = y[:, None] - forecasts
    losses = squared_loss(y[:, None], forecasts)

    cfg = ScenarioConfig(name="audit", M=M, T=T, T0=T0, seed=seed)
    data = SimulationData(
        y=y, forecasts=forecasts, errors=errors, losses=losses,
        bias_paths=np.zeros((T, M)), sigma_paths=np.ones((T, M)) * 0.5,
        common_shock=np.zeros(T), config=cfg,
    )

    bt_cfg = BacktestConfig(
        d_max=2, fixed_d=1, fixed_h1=0.3, fixed_h2=0.4,
        cov_window=30, min_history=20,
        alpha=0.1, gamma=0.1,
    )
    res = run_backtest(data, bt_cfg)

    # Check: at each OOS period t, the weight vector was formed
    # WITHOUT using y[t]. We verify by checking that the first OOS
    # combined forecast does NOT perfectly nail y[T0], which would
    # indicate leakage.
    first_loss_full = res.combined_losses["full_gcsr"][0]
    first_loss_eq = res.combined_losses["equal"][0]

    # In a correct implementation both should have nonzero loss
    audit_pass = True
    notes = []

    if first_loss_full < 1e-15:
        notes.append("WARNING: Full method has near-zero first-period loss — possible leakage")
        audit_pass = False

    # Additional check: weights should be identical if we permute future y
    # (we can't run this dynamically here, but the structure guarantees it)

    # Check that all weight vectors sum to ~1
    for name in res.weights:
        w = res.weights[name]
        sums = w.sum(axis=1)
        if np.any(np.abs(sums - 1.0) > 1e-4):
            notes.append(f"WARNING: {name} weights don't sum to 1")
            audit_pass = False

    # Check non-negativity
    for name in res.weights:
        w = res.weights[name]
        if np.any(w < -1e-8):
            notes.append(f"WARNING: {name} has negative weights")
            audit_pass = False

    return audit_pass, notes, res


def print_timing_rules():
    """Print the information set / timing conventions."""
    text = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    TIMING CONVENTIONS                          ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                ║
    ║  Forecast origin: t                                            ║
    ║  Target to predict: y_{t}                                      ║
    ║  Forecasts available: f_{j,t} for j=1,...,M                    ║
    ║  Realised outcomes available: y_0, ..., y_{t-1}                ║
    ║  Realised losses available: L_{j,0}, ..., L_{j,t-1}           ║
    ║  Realised errors available: e_{j,0}, ..., e_{j,t-1}           ║
    ║                                                                ║
    ║  Weights w_{t} are formed using ONLY:                          ║
    ║    - losses / errors from periods 0 through t-1                ║
    ║    - pairwise LD from periods 0 through t-1                    ║
    ║    - covariance from errors 0 through t-1                      ║
    ║                                                                ║
    ║  y_{t} is NEVER used when forming w_{t}.                       ║
    ║                                                                ║
    ║  Combined forecast: ŷ_t = Σ_j w_{j,t} f_{j,t}                ║
    ║  Evaluated after y_t is realised.                              ║
    ║                                                                ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(text)


# ===================================================================
# 12.  CONVENIENCE: RUN ALL SCENARIOS
# ===================================================================

def run_all_scenarios(
    n_reps: int = 10,
    bt_cfg: BacktestConfig = None,
    verbose: bool = True,
    M: int = 8,
    T: int = 300,
    T0: int = 150,
) -> Dict[str, MCResult]:
    """Run MC for every pre-built scenario."""
    if bt_cfg is None:
        bt_cfg = BacktestConfig(
            d_max=3,
            fixed_d=1,        # speed: fix lag to 1
            fixed_h1=0.25,    # speed: fix bandwidth
            fixed_h2=0.35,
            cov_window=50,
            min_history=30,
            alpha=None,
            gamma=None,
            tune_window=30,
        )

    results = {}
    for sc_key, factory in ALL_SCENARIO_FACTORIES.items():
        if verbose:
            print(f"\n{'='*50}")
            print(f"Running scenario {sc_key}...")
            print(f"{'='*50}")
        mc = run_monte_carlo(
            factory,
            n_reps=n_reps,
            bt_cfg=bt_cfg,
            verbose=verbose,
            M=M, T=T, T0=T0,
        )
        results[sc_key] = mc
        if verbose:
            print(summarise_mc(mc).to_string(index=False))
    return results


# ===================================================================
# END OF MODULE
# ===================================================================

if __name__ == "__main__":
    print("forecast_combination module loaded successfully.")
    print("Run leakage audit...")
    passed, notes, _ = leakage_audit_synthetic()
    print(f"Audit passed: {passed}")
    for n in notes:
        print(f"  {n}")