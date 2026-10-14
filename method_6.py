# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
method_6.py
===========
第6种抽样方法：两阶段方法

    第一阶段（与 method_5 完全相同）：
        MCMC 抽取 mixture_beta 后验样本 → 后验均值 beta_bar
        → propensity scores → adj_weights

    第二阶段（与 method_5 的区别）：
        使用 single normal 后验，无混合模型（无 p、rho、z）。
        直接以 adj_weights 为权重更新 beta 和 sigma2。
        variance = sigma2 / N_pop（单正态，无混合）

对外接口：run(pop_data, sim_id) -> dict
tr
"""

import numpy as np
from scipy.stats import gamma, multivariate_normal
from mcmc_utils import (log_p, g_func, bisection_sample, calculate_hpd)

# =============================================================================
# 方法参数
# =============================================================================

N_SAMPLE = 500      # TODO: 确认每次从population中抽取的样本量

# 第一阶段（与 method_5 完全相同）
N_TOTAL_1 = 50000
BURN_IN_1  = 10000
TARGET_1   = 1000
BI_ITER_1  = 5

# 第二阶段（single normal）
N_TOTAL_2 = 50000
BURN_IN_2  = 10000
TARGET_2   = 1000

# =============================================================================
# 共用工具函数
# =============================================================================

def stage1_sample_beta_j(j, n_p, o_p, beta_cur, d, it=7):
    def log_target(t):
        beta_tmp = beta_cur.copy()
        beta_tmp[j] = t
        theta = np.log(beta_tmp / (1 - beta_tmp))
        return log_p(n_p, o_p, theta, d)
    return bisection_sample(log_target, it)

def run_stage1(n_p, o_p, d_s, N_pop):
    """
    第一阶段 MCMC（与 method_5 完全相同）：
    返回 adj_weights（shape: n_sample，已 rescale 到 N_pop 尺度）
    """
    n_comp = n_p.shape[1]

    mixture_beta    = np.zeros([N_TOTAL_1, n_comp])
    mixture_beta[0] = 0.5

    for i in range(1, N_TOTAL_1):
        mixture_beta[i] = mixture_beta[i - 1]
        for j in range(n_comp):
            mixture_beta[i, j] = stage1_sample_beta_j(
                j, n_p, o_p, mixture_beta[i], d_s, it=BI_ITER_1
            )

    # Burn-in + Thinning
    burned       = mixture_beta[BURN_IN_1:]
    thin         = burned.shape[0] // TARGET_1
    beta_samples = burned[::thin][:TARGET_1]   # shape (1000, 4)

    # 后验均值 beta_bar → propensity scores → adj_weights
    beta_bar  = beta_samples.mean(axis=0)
    theta_bar = np.log(beta_bar / (1 - beta_bar))
    lin       = n_p @ theta_bar
    ps        = np.exp(lin) / (1 + np.exp(lin))

    raw_w = 1.0 / ps
    adj_w = N_pop * raw_w / raw_w.sum()

    return adj_w

# =============================================================================
# 第二阶段：single normal MCMC（adj_weights 固定，无 p/rho/z）
# =============================================================================

def run_stage2(n_p, y_s, adj_w):
    """
    第二阶段：单正态模型，adj_weights 作为固定权重。
    直接更新 beta 和 sigma2，无混合模型。

    返回:
        result_final : shape (TARGET_2, 5)，列为 [beta(4), sigma2]
    """
    n_s    = n_p.shape[0]
    n_comp = n_p.shape[1]

    result  = np.zeros((N_TOTAL_2, n_comp + 1))
    beta    = np.array([-0.01, 0.5, 0.5, 27])
    sigma_2 = 100.0

    for i in range(N_TOTAL_2):
        # ── 更新 beta（加权最小二乘后验正态）────────────────────────────
        # W = diag(adj_w): avoid allocating n×n matrix
        Xw       = n_p * adj_w[:, None]
        fpart    = np.linalg.inv(Xw.T @ n_p)
        spart    = Xw.T @ y_s
        beta_hat = fpart @ spart
        beta     = multivariate_normal.rvs(mean=beta_hat, cov=fpart * sigma_2)
        result[i, :n_comp] = beta

        # ── 更新 sigma2 ───────────────────────────────────────────────────
        rate    = adj_w * (y_s - n_p @ beta_hat) ** 2
        r       = sum(rate) / 2
        sigma_2 = 1 / gamma.rvs(a=n_s / 2 - 2, scale=1 / r)
        result[i, n_comp] = sigma_2

    # Burn-in + Thinning
    burned = result[BURN_IN_2:]
    thin   = burned.shape[0] // TARGET_2
    return burned[::thin][:TARGET_2]

# =============================================================================
# 后验 y_bar 抽样与统计量计算
# =============================================================================

def sample_ybar(result_final, x_bar, N_pop):
    """
    单正态模型：variance = sigma2 / N_pop（无混合，无 rho）
    """
    y_bar_samples = np.zeros(TARGET_2)
    for i in range(TARGET_2):
        mean_y   = x_bar @ result_final[i, 0:4]
        variance = result_final[i, 4] / N_pop
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

    # 2. 第一阶段：logistic MCMC → adj_weights（与 method_5 相同）
    adj_w = run_stage1(n_p, o_p, d_s, pop_data["N_pop"])

    # 3. 第二阶段：single normal MCMC
    result_final = run_stage2(n_p, y_s, adj_w)

    # 4. 生成 y_bar 后验样本
    y_bar_samples = sample_ybar(
        result_final,
        pop_data["x_bar"], pop_data["N_pop"]
    )

    # 5. 统计量
    hpd_l, hpd_u = calculate_hpd(y_bar_samples)
    return {
        "posterior_mean": float(np.mean(y_bar_samples)),
        "posterior_std":  float(np.std(y_bar_samples, ddof=1)),
        "hpd_lower":      float(hpd_l),
        "hpd_upper":      float(hpd_u),
    }