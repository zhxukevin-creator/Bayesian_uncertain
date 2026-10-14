# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 09:50:40 2026

@author: 10798
"""

# -*- coding: utf-8 -*-
"""
mcmc_utils.py
=============
所有 method 文件共用的数值稳定版工具函数。

修复的三个数值问题：
  1. log_p 中 log(1 + exp(x)) 溢出  → 用 np.logaddexp(0, x)
  2. pi_inv 中 exp(a)/(1+exp(a)) 溢出 → 用 scipy.special.expit
  3. bisection_sample 中 pr = NaN   → clip 到 [eps, 1-eps]
"""

import numpy as np
from scipy.special import expit   # 数值稳定的 sigmoid

# =============================================================================
# 核心数学函数（数值稳定版）
# =============================================================================

def log_p(n_p, o_p, theta, d):
    """
    log p(d | theta) for logistic model.
    修复：np.dot(o_p, theta) 可能很大，
          原来 log(1 + exp(x)) 会溢出，
          改用 np.logaddexp(0, x) 等价且稳定。
    """
    fpart = np.sum(n_p @ theta)
    lin   = o_p @ theta
    spart = np.dot(d, np.logaddexp(0, lin))   # 稳定版 log(1+exp(x))
    return fpart - spart

def pi_inv(n_p, beta):
    """
    1 / pi(beta)，其中 pi = sigmoid(n_p @ logit(beta))。
    修复：原来 exp(a)/(1+exp(a)) 在 a 很大时溢出，
          改用 scipy.special.expit(a)，等价且稳定。
    """
    theta = np.log(beta / (1 - beta))
    a     = n_p @ theta
    pi    = expit(a)                           # 稳定版 sigmoid
    # 防止 pi=0 导致除零
    pi    = np.clip(pi, 1e-15, 1 - 1e-15)
    return 1.0 / pi

def g_func(n_p, beta):
    a = pi_inv(n_p, beta)
    return sum(a) / sum(a**2) * a

# =============================================================================
# 通用二分采样器（数值稳定版）
# =============================================================================

def bisection_sample(log_target_fn, it):
    """
    在 (0,1) 上对单变量进行二分采样。
    修复：den 全为 -inf 时 exp(den-max) 产生 NaN，
          pr 变成 NaN 传入 binomial 报错。
          加入三层保护：
            1. den 中有 -inf 时用 nan_to_num 替换
            2. l+r 为 0 时 pr 默认 0.5
            3. pr clip 到 [eps, 1-eps]
    """
    a, b = 0.0, 1.0
    eps  = 1e-10

    for _ in range(it):
        x_gl, w_gl = np.polynomial.legendre.leggauss(3)
        mid = (a + b) / 2.0
        t1  = 0.5 * (x_gl + 1) * (b - mid) + mid
        t2  = 0.5 * (x_gl + 1) * (mid - a) + a
        t   = np.concatenate((t2, t1))

        # 计算 log density
        den = np.array([log_target_fn(ti) for ti in t])

        # 保护1：将 -inf / NaN 替换为有限小值
        den = np.nan_to_num(den, nan=-1e300, posinf=0, neginf=-1e300)

        # log-sum-exp 归一化
        den_max = den.max()
        den     = np.exp(den - den_max)
        den_sum = den.sum()
        if den_sum == 0 or not np.isfinite(den_sum):
            den = np.ones(6) / 6.0   # 退化为均匀分布
        else:
            den /= den_sum

        l = float(np.dot(w_gl, den[0:3])) * 0.5 * (mid - a)
        r = float(np.dot(w_gl, den[3:6])) * 0.5 * (b - mid)

        # 保护2：l+r 为 0 时默认 0.5
        lr = l + r
        if lr <= 0 or not np.isfinite(lr):
            pr = 0.5
        else:
            pr = r / lr

        # 保护3：clip 到合法概率范围
        pr = float(np.clip(pr, eps, 1.0 - eps))

        if np.random.binomial(1, pr):
            a = mid
        else:
            b = mid

    return float(np.random.uniform(a, b))

# =============================================================================
# 混合正态相关函数（method_1/2/5 共用）
# =============================================================================

def logp(p, y, x, beta_y, rho, sigma_2, w):
    mean   = x @ beta_y
    var1   = rho * sigma_2 / w
    var2   = sigma_2 / w
    # 防止 var 为零或负
    var1   = np.maximum(var1, 1e-15)
    var2   = np.maximum(var2, 1e-15)
    from scipy.stats import norm
    result = np.log(
        np.maximum(
            (1 - p / 2) * norm.pdf(y, mean, np.sqrt(var1)) +
            p / 2       * norm.pdf(y, mean, np.sqrt(var2)),
            1e-300      # 防止 log(0)
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
# HPD 区间计算（所有 method 共用）
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