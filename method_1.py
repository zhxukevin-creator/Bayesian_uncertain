# -*- coding: utf-8 -*-
"""
method_1.py
===========
第1种抽样方法：直接改写自原始代码
抽样策略：简单随机抽样（SRS）
对外接口：run(pop_data, sim_id) -> dict
unp_r
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
N_TOTAL  = 50000   # MCMC总迭代次数
BURN_IN  = 10000    # burn-in丢弃数
TARGET   = 1000     # 最终保留的后验样本数

# =============================================================================
# mixture_beta 分量采样
# =============================================================================

def sample_beta_j(j, n_p, o_p, beta_cur, d, it=10):
    """采样 mixture_beta 的第 j 个分量"""
    def log_target(t):
        beta_tmp = beta_cur.copy()
        beta_tmp[j] = t
        theta = np.log(beta_tmp / (1 - beta_tmp))
        return log_p(n_p, o_p, theta, d)
    return bisection_sample(log_target, it)

# =============================================================================
# MCMC 主循环
# =============================================================================

def run_mcmc(n_p, o_p, y_s, d_s):
    """
    对给定样本数据运行完整MCMC，返回burn-in+thinning后的后验样本。

    返回:
        y_beta_final : shape (TARGET, 6)，列为 [beta(4), sigma2, rho]
        p_final      : shape (TARGET,)
    """
    n_comp = n_p.shape[1]   # = 4

    mixture_beta = np.zeros([N_TOTAL, n_comp])
    mixture_beta[0] = 0.5

    y_beta   = np.zeros([N_TOTAL, n_comp + 2])   # beta(4) + sigma2 + rho
    p_result = np.zeros(N_TOTAL)

    # 初始值（与原始代码一致）
    beta    = np.array([-0.01, 0.5, 0.5, 27])
    rho     = 0.5
    sigma_2 = 100.0
    p       = 0.03

    for i in range(1, N_TOTAL):
        mixture_beta[i] = mixture_beta[i - 1]

        # ── 更新 mixture_beta（逐分量二分采样）────────────────────────────
        for j in range(n_comp):
            mixture_beta[i, j] = sample_beta_j(
                j, n_p, o_p, mixture_beta[i], d_s, it=5
            )

        gi = g_func(n_p, mixture_beta[i])

        # ── 更新 z（潜变量）───────────────────────────────────────────────
        mean1 = n_p @ beta
        p1 = (1 - p) * norm.pdf(y_s, mean1, np.sqrt(rho * sigma_2 / gi))
        p2 =      p  * norm.pdf(y_s, mean1, np.sqrt(sigma_2 / gi))
        pr = p2 / (p1 + p2)
        z  = np.random.binomial(1, pr)

        # ── 更新 p ────────────────────────────────────────────────────────
        p = sample_p_bi(y_s, n_p, beta, rho, sigma_2, gi, it=6) / 2
        p_result[i] = p

        # ── 更新 beta（后验正态）─────────────────────────────────────────
        t_vec    = gi / rho - gi * z / rho + gi * z
        # G = diag(t_vec): avoid allocating n×n matrix, use broadcasting instead
        Xw       = n_p * t_vec[:, None]          # (n,p) weighted X
        fpart    = np.linalg.inv(Xw.T @ n_p)
        spart    = Xw.T @ y_s
        beta_hat = fpart @ spart
        cov      = fpart * sigma_2
        beta     = multivariate_normal.rvs(mean=beta_hat, cov=cov)
        y_beta[i, :n_comp] = beta

        # ── 更新 sigma2 ───────────────────────────────────────────────────
        rate    = t_vec * (y_s - n_p @ beta_hat) ** 2
        r       = sum(rate) / 2
        sigma_2 = 1 / gamma.rvs(a=n_p.shape[0] / 2 - 2, scale=1 / r)
        y_beta[i, n_comp] = sigma_2

        # ── 更新 rho ──────────────────────────────────────────────────────
        rho = sample_rho_bi(z, gi, y_s, sigma_2, n_p, beta, it=5)
        y_beta[i, n_comp + 1] = rho

    # ── Burn-in + Thinning ────────────────────────────────────────────────
    y_beta_burned = y_beta[BURN_IN:]
    p_burned      = p_result[BURN_IN:]
    thin          = y_beta_burned.shape[0] // TARGET

    y_beta_final = y_beta_burned[::thin][:TARGET]
    p_final      = p_burned[::thin][:TARGET]

    return y_beta_final, p_final

# =============================================================================
# 后验 y_bar 抽样与统计量计算
# =============================================================================

def sample_ybar(y_beta_final, p_final, x_bar, N_pop):
    """从1000个后验样本中生成 y_bar 样本（与原始代码一致）"""
    y_bar_samples = np.zeros(TARGET)
    for i in range(TARGET):
        mean_y = np.dot(x_bar, y_beta_final[i, 0:4])
        if np.random.uniform() < p_final[i]:
            variance = y_beta_final[i, 4] / N_pop
        else:
            variance = y_beta_final[i, 5] * y_beta_final[i, 4] / N_pop
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
    np.random.seed(sim_id)   # 固定种子，保证可复现

    # 1. 简单随机抽样
    idx = np.random.choice(pop_data["N_pop"], size=1500, replace=False)
    ids = np.random.choice(pop_data["N_pop"], size=300, replace=False)
    n_p = pop_data["n_p"][idx]
    o_p = pop_data["o_p"][ids]
    y_s = pop_data["y"][idx]
    d_s = pop_data["d_b"][ids]

    # 2. MCMC
    y_beta_final, p_final = run_mcmc(n_p, o_p, y_s, d_s)

    # 3. 生成 y_bar 后验样本
    y_bar_samples = sample_ybar(
        y_beta_final, p_final,
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