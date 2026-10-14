# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
method_4.py
===========
第4种抽样方法：改写自 uncertainty_logit_noproduct_a.py

与 method_3 的唯一区别：
    mixture_beta 的二分采样使用 log_theta() 作为 log target，
    即额外加入了 log(g)/2 项和 likelihood 项（但无 rho/z，比 method_2 更简单）。
    method_3 只使用 log_p。

与 method_2 的区别：
    无混合模型（无 p、rho、z），log_theta 不含 rho/z 相关项。

对照关系：
    method_3 : method_4  ≈  method_1 : method_2
    （同样的结构，log target 从 log_p 换成 log_theta）

⚠️  variance = sigma2 / N_pop，与 method_4 原始代码一致（N=80000 → 改为 N_pop）。
    同时据此修正了 method_3 的 TODO。

抽样策略：简单随机抽样（SRS）
对外接口：run(pop_data, sim_id) -> dict
"""

import numpy as np
from scipy.stats import norm, gamma, multivariate_normal
from mcmc_utils import (log_p, g_func, bisection_sample,
                        logp, log_rho, sample_p_bi, sample_rho_bi,
                        calculate_hpd)

# =============================================================================
# 方法参数
# =============================================================================

N_SAMPLE = 500      # TODO: 确认每次从population中抽取的样本量
N_TOTAL  = 50000    # MCMC总迭代次数
BURN_IN  = 10000    # burn-in丢弃数
TARGET   = 1000     # 最终保留的后验样本数
BI_ITER  = 5       # 二分采样迭代次数

# =============================================================================
# MCMC 核心函数
# =============================================================================

def log_theta(n_p, o_p, y, beta, d, beta_y, sigma_2):
    """
    method_4 的 log target（用于 mixture_beta 采样）。
    与 method_2 的 log_theta 相比，无 rho/z 项：
        spart = gi @ (y - X*beta_y)^2 / (2*sigma_2)
    而 method_2 是：
        spart = (gi/rho - gi*z/rho + gi*z) @ (...) / (2*sigma_2)
    """
    gi    = g_func(n_p, beta)
    fpart = sum(np.log(gi)) / 2
    spart = gi @ (y - n_p @ beta_y) ** 2 / (2 * sigma_2)
    theta = np.log(beta / (1 - beta))
    tpart = log_p(n_p, o_p, theta, d)
    return fpart - spart + tpart

# =============================================================================
# 二分采样器（通用）
# =============================================================================

def sample_beta_j(j, n_p, o_p, beta_cur, d, y, beta_y, sigma_2, it=10):
    """
    采样 mixture_beta 的第 j 个分量。
    log target 使用 log_theta()，额外传入当前的 beta_y 和 sigma_2。
    """
    def log_target(t):
        beta_tmp = beta_cur.copy()
        beta_tmp[j] = t
        return log_theta(n_p, o_p, y, beta_tmp, d, beta_y, sigma_2)
    return bisection_sample(log_target, it)

# =============================================================================
# MCMC 主循环
# =============================================================================

def run_mcmc(n_p, o_p, y_s, d_s):
    """
    对给定样本数据运行完整MCMC，返回burn-in+thinning后的后验样本。

    返回:
        y_beta_final : shape (TARGET, 5)，列为 [beta(4), sigma2]
    """
    n_comp = n_p.shape[1]   # = 4

    mixture_beta = np.zeros([N_TOTAL, n_comp])
    mixture_beta[0] = 0.5

    y_beta  = np.zeros([N_TOTAL, n_comp + 1])   # beta(4) + sigma2
    sigma_2 = 1.0
    beta_y  = np.array([0.5, 0.5, 0.5, 0.5])   # 初始值（与原始代码一致）

    for i in range(1, N_TOTAL):
        mixture_beta[i] = mixture_beta[i - 1]

        # ── 更新 mixture_beta（log_theta 作为 log target）────────────────
        for j in range(n_comp):
            mixture_beta[i, j] = sample_beta_j(
                j, n_p, o_p, mixture_beta[i], d_s,
                y_s, beta_y, sigma_2, it=BI_ITER
            )

        gi = g_func(n_p, mixture_beta[i])

        # ── 更新 beta（直接用 gi，与 method_3 相同）──────────────────────
        # G = diag(gi): avoid allocating n×n matrix
        Xw       = n_p * gi[:, None]
        fpart    = np.linalg.inv(Xw.T @ n_p)
        spart    = Xw.T @ y_s
        beta_hat = fpart @ spart
        cov      = fpart * sigma_2
        beta_y   = multivariate_normal.rvs(mean=beta_hat, cov=cov)
        y_beta[i, :n_comp] = beta_y

        # ── 更新 sigma2 ───────────────────────────────────────────────────
        rate    = gi @ (y_s - n_p @ beta_hat) ** 2 / 2
        sigma_2 = 1 / gamma.rvs(a=n_p.shape[0] / 2 - 2, scale=1 / rate)
        y_beta[i, n_comp] = sigma_2

    # ── Burn-in + Thinning ────────────────────────────────────────────────
    y_beta_burned = y_beta[BURN_IN:]
    thin          = y_beta_burned.shape[0] // TARGET

    y_beta_final = y_beta_burned[::thin][:TARGET]
    return y_beta_final

# =============================================================================
# 后验 y_bar 抽样与统计量计算
# =============================================================================

def sample_ybar(y_beta_final, x_bar, N_pop):
    """
    variance = sigma2 / N_pop（与原始代码 sigma2/N 一致，N 替换为 N_pop）
    """
    y_bar_samples = np.zeros(TARGET)
    for i in range(TARGET):
        mean_y   = np.dot(x_bar, y_beta_final[i, 0:4])
        variance = y_beta_final[i, 4] / N_pop
        y_bar_samples[i] = np.random.normal(loc=mean_y, scale=np.sqrt(variance))
    return y_bar_samples

def run(pop_data, sim_id):
    """
    参数:
        pop_data : simulation.py 传入的 population 数据字典
        sim_id   : 当前 simulation 编号（用于固定随机种子）
    返回:
        {posterior_mean, posterior_std, hpd_lower, hpd_upper}
    """
    np.random.seed(sim_id)

    # 1. 简单随机抽样
    idx = np.random.choice(pop_data["N_pop"], size=1500, replace=False)
    ids = np.random.choice(pop_data["N_pop"], size=300, replace=False)
    n_p = pop_data["n_p"][idx]
    o_p = pop_data["o_p"][ids]
    y_s = pop_data["y"][idx]
    d_s = pop_data["d_b"][ids]

    # 2. MCMC
    y_beta_final = run_mcmc(n_p, o_p, y_s, d_s)

    # 3. 生成 y_bar 后验样本
    y_bar_samples = sample_ybar(
        y_beta_final,
        pop_data["x_bar"], pop_data["N_pop"]
    )

    # 4. 统计量
    hpd_l, hpd_u = calculate_hpd(y_bar_samples)
    return {
        "posterior_mean": float(np.mean(y_bar_samples)),
        "posterior_std":  float(np.std(y_bar_samples, ddof=1)),
        "hpd_lower":      float(hpd_l),
        "hpd_upper":      float(hpd_u),
    }