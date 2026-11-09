# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 09:50:40 2026

@author: 10798
"""

# -*- coding: utf-8 -*-
"""
mcmc_utils.py
=============
common function for all methods.
"""

import numpy as np
from scipy.special import expit   

# =============================================================================
# log_p
# =============================================================================

def log_p(n_p, o_p, theta, d):
    
    fpart = np.sum(n_p @ theta)
    lin   = o_p @ theta
    spart = np.dot(d, np.logaddexp(0, lin))   
    return fpart - spart

def pi_inv(n_p, beta):
  
    theta = np.log(beta / (1 - beta))
    a     = n_p @ theta
    pi    = expit(a)                           
    pi    = np.clip(pi, 1e-15, 1 - 1e-15)
    return 1.0 / pi

def g_func(n_p, beta):
    a = pi_inv(n_p, beta)
    return sum(a) / sum(a**2) * a

# =============================================================================
# sampler
# =============================================================================

def bisection_sample(log_target_fn, it):
    a, b = 0.0, 1.0
    eps  = 1e-10

    for _ in range(it):
        x_gl, w_gl = np.polynomial.legendre.leggauss(3)
        mid = (a + b) / 2.0
        t1  = 0.5 * (x_gl + 1) * (b - mid) + mid
        t2  = 0.5 * (x_gl + 1) * (mid - a) + a
        t   = np.concatenate((t2, t1))

        # log density
        den = np.array([log_target_fn(ti) for ti in t])

        
        den = np.nan_to_num(den, nan=-1e300, posinf=0, neginf=-1e300)

        # log-sum-exp normalize
        den_max = den.max()
        den     = np.exp(den - den_max)
        den_sum = den.sum()
        if den_sum == 0 or not np.isfinite(den_sum):
            den = np.ones(6) / 6.0   
        else:
            den /= den_sum

        l = float(np.dot(w_gl, den[0:3])) * 0.5 * (mid - a)
        r = float(np.dot(w_gl, den[3:6])) * 0.5 * (b - mid)

        
        lr = l + r
        if lr <= 0 or not np.isfinite(lr):
            pr = 0.5
        else:
            pr = r / lr

        
        pr = float(np.clip(pr, eps, 1.0 - eps))

        if np.random.binomial(1, pr):
            a = mid
        else:
            b = mid

    return float(np.random.uniform(a, b))

# =============================================================================
# mixture-normal（method_1/2/5 ）
# =============================================================================

def logp(p, y, x, beta_y, rho, sigma_2, w):
    mean   = x @ beta_y
    var1   = rho * sigma_2 / w
    var2   = sigma_2 / w
    
    var1   = np.maximum(var1, 1e-15)
    var2   = np.maximum(var2, 1e-15)
    from scipy.stats import norm
    result = np.log(
        np.maximum(
            (1 - p / 2) * norm.pdf(y, mean, np.sqrt(var1)) +
            p / 2       * norm.pdf(y, mean, np.sqrt(var2)),
            1e-300      
        )
    )
    return float(np.sum(result))

def log_rho(rho, z, gi, y, sigma_2, x, beta_y):
    t     = gi / rho - gi * z / rho + gi * z
    mu    = (y - x @ beta_y) ** 2
    spart = float(np.dot(t, mu)) / (2 * sigma_2)
    fpart = (float(np.sum(z)) - len(z)) * np.log(rho) / 2
    return fpart - spart

def sample_p_bi(y, data, beta_y, rho, sigma_2, weights, it=7):
    return bisection_sample(
        lambda t: logp(t, y, data, beta_y, rho, sigma_2, weights),
        it
    )

def sample_rho_bi(z, gi, y, sigma_2, data, beta_y, it=5):
    return bisection_sample(
        lambda t: log_rho(t, z, gi, y, sigma_2, data, beta_y),
        it
    )

# =============================================================================
# HPD （all method ）
# =============================================================================

def calculate_hpd(samples, alpha=0.05):
    sorted_s     = np.sort(samples)
    n            = len(sorted_s)
    interval_len = int(np.floor((1 - alpha) * n))
    min_width    = np.inf
    hpd_l, hpd_u = sorted_s[0], sorted_s[-1]
    for i in range(n - interval_len):
        lower = sorted_s[i]
        upper = sorted_s[i + interval_len]
        if upper - lower < min_width:
            min_width = upper - lower
            hpd_l, hpd_u = lower, upper
    return float(hpd_l), float(hpd_u)


# =============================================================================
# stable_z_prob
# =============================================================================

def stable_z_prob(y, mean1, sd1, sd2, p):
    """pr = p2/(p1+p2)，p1=(1-p)N(y|mean,sd1)，p2=p N(y|mean,sd2)"""
    log_p1 = (np.log(np.maximum(1 - p, 1e-300))
              - 0.5 * ((y - mean1) / sd1) ** 2 - np.log(np.maximum(sd1, 1e-300)))
    log_p2 = (np.log(np.maximum(p, 1e-300))
              - 0.5 * ((y - mean1) / sd2) ** 2 - np.log(np.maximum(sd2, 1e-300)))
    pr = np.exp(log_p2 - np.logaddexp(log_p1, log_p2))
    return np.clip(pr, 1e-10, 1 - 1e-10)

# =============================================================================
# MCMC ：ESS Geweke z-score
# =============================================================================

def ess(x):
    
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4 or np.var(x) == 0:
        return float(n)
    xc = x - x.mean()
    f = np.fft.rfft(xc, 2 * n)
    acov = np.fft.irfft(f * np.conj(f))[:n] / n
    rho = acov / acov[0]
    tau = -1.0
    for k in range(0, n - 1, 2):
        pair = rho[k] + rho[k + 1]
        if pair < 0:
            break
        tau += 2 * pair
    tau = max(tau, 1.0 / n)
    return float(n / tau)

def geweke_z(x, first=0.1, last=0.5):
    
    x = np.asarray(x, dtype=float)
    n = len(x)
    a = x[: int(first * n)]
    b = x[int((1 - last) * n):]
    va = np.var(a, ddof=1) / ess(a)
    vb = np.var(b, ddof=1) / ess(b)
    denom = np.sqrt(va + vb)
    return float((a.mean() - b.mean()) / denom) if denom > 0 else 0.0

def diagnose(samples):
    
    import pandas as pd
    rows = {}
    for c in samples.columns:
        e = ess(samples[c].values)
        rows[c] = {"ess": e, "ess_ratio": e / len(samples),
                   "geweke_z": geweke_z(samples[c].values)}
    return pd.DataFrame(rows).T
