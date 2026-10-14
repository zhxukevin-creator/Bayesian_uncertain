# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
method_5.py
===========
第5种抽样方法：两阶段方法
    第一阶段（logistic.py）：
        MCMC 抽取 mixture_beta 的后验样本，
        取1000个独立样本的后验均值得到 beta_bar（4个参数），
        用 logistic 公式计算 propensity scores，
        以 1/ps 为权重并 rescale 得到 adj_weights。

    第二阶段（y_bar_double.py）：
        以 adj_weights 作为固定权重，
        对混合正态模型运行 MCMC，抽取 beta/sigma2/rho/p，
        最终生成 y_bar 后验样本并计算统计量。

对外接口：run(pop_data, sim_id) -> dict
tr_r
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

# 第一阶段（logistic MCMC）参数
N_TOTAL_1 =50000
BURN_IN_1  = 10000
TARGET_1   = 1000
BI_ITER_1  = 5      # 原始代码为7

# 第二阶段（double robust MCMC）参数
N_TOTAL_2 = 50000
BURN_IN_2  = 10000
TARGET_2   = 1000
P_ITER_2   = 6
RHO_ITER_2 = 5

# =============================================================================
# 共用工具函数
# =============================================================================

def stage1_sample_beta_j(j, n_p, o_p, beta_cur, d, it=7):
    """第一阶段：采样 mixture_beta 的第 j 个分量，log target = log_p"""
    def log_target(t):
        beta_tmp = beta_cur.copy()
        beta_tmp[j] = t
        theta = np.log(beta_tmp / (1 - beta_tmp))
        return log_p(n_p, o_p, theta, d)
    return bisection_sample(log_target, it)

def run_stage1(n_p, o_p, d_s, N_pop):
    """
    第一阶段 MCMC：仅抽取 mixture_beta。
    返回 adj_weights（shape: n_sample，已 rescale 到 N_pop 尺度）。
    """
    n_comp = n_p.shape[1]   # = 4
    n_s    = n_p.shape[0]   # 样本量

    mixture_beta = np.zeros([N_TOTAL_1, n_comp])
    mixture_beta[0] = 0.5

    for i in range(1, N_TOTAL_1):
        mixture_beta[i] = mixture_beta[i - 1]
        for j in range(n_comp):
            mixture_beta[i, j] = stage1_sample_beta_j(
                j, n_p, o_p, mixture_beta[i], d_s, it=BI_ITER_1
            )

    # Burn-in + Thinning
    burned = mixture_beta[BURN_IN_1:]
    thin   = burned.shape[0] // TARGET_1
    beta_samples = burned[::thin][:TARGET_1]   # shape (1000, 4)

    # 后验均值 beta_bar
    beta_bar = beta_samples.mean(axis=0)       # shape (4,)

    # 计算 propensity scores（logistic）
    theta_bar = np.log(beta_bar / (1 - beta_bar))
    lin       = n_p @ theta_bar                # shape (n_s,)
    ps        = np.exp(lin) / (1 + np.exp(lin))

    # adj_weights = N_pop * (1/ps) / sum(1/ps)
    raw_w     = 1.0 / ps
    adj_w     = N_pop * raw_w / raw_w.sum()

    return adj_w

# =============================================================================
# 第二阶段：混合正态 MCMC（adj_weights 固定）→ y_bar
# =============================================================================

def run_stage2(n_p, y_s, adj_w):
    """
    第二阶段 MCMC：以 adj_weights 为固定权重，
    抽取 beta/sigma2/rho/p 的后验样本。
    返回:
        result_final : shape (TARGET_2, 8)
                       列为 [beta(4), sigma2, rho, p, sum_z]
    """
    n_s    = n_p.shape[0]
    n_comp = n_p.shape[1]

    result   = np.zeros((N_TOTAL_2, 8))
    beta    = np.array([-0.01, 0.5, 0.5, 27])
    rho      = 0.5
    sigma_2  = 100.0
    p        = 0.3

    for i in range(N_TOTAL_2):
        # 更新 z
        mean1 = n_p @ beta
        sd1   = rho * sigma_2 / adj_w
        sd2   = sigma_2 / adj_w
        p1    = (1 - p) * norm.pdf(y_s, mean1, np.sqrt(sd1))
        p2    =      p  * norm.pdf(y_s, mean1, np.sqrt(sd2))
        pr    = p2 / (p1 + p2)
        z     = np.random.binomial(1, pr)

        # 更新 p
        p = sample_p_bi(y_s, n_p, beta, rho, sigma_2, adj_w, it=P_ITER_2) / 2

        # 更新 beta
        t_vec    = adj_w / rho - adj_w * z / rho + adj_w * z
        # W = diag(t_vec): avoid allocating n×n matrix
        Xw       = n_p * t_vec[:, None]
        fpart    = np.linalg.inv(Xw.T @ n_p)
        spart    = Xw.T @ y_s
        beta_hat = fpart @ spart
        beta     = multivariate_normal.rvs(mean=beta_hat, cov=fpart * sigma_2)

        # 更新 sigma2
        rate    = t_vec * (y_s - n_p @ beta_hat) ** 2
        r       = sum(rate) / 2
        sigma_2 = 1 / gamma.rvs(a=n_s / 2 - 2, scale=1 / r)

        # 更新 rho
        rho = sample_rho_bi(z, adj_w, y_s, sigma_2, n_p, beta, it=RHO_ITER_2)

        result[i, 0:4] = beta
        result[i, 4]   = sigma_2
        result[i, 5]   = rho
        result[i, 6]   = p
        result[i, 7]   = sum(z)

    # Burn-in + Thinning
    burned = result[BURN_IN_2:]
    thin   = burned.shape[0] // TARGET_2
    return burned[::thin][:TARGET_2]

# =============================================================================
# 后验 y_bar 抽样与统计量计算
# =============================================================================

def sample_ybar(result_final, x_bar, N_pop):
    """
    从第二阶段后验样本生成 y_bar。
    result_final 列：[beta(4), sigma2, rho, p, sum_z]
    混合方差：以概率p取 sigma2/N_pop，以概率(1-p)取 rho*sigma2/N_pop
    """
    y_bar_samples = np.zeros(TARGET_2)
    for i in range(TARGET_2):
        mean_y = x_bar @ result_final[i, 0:4]
        p_i    = result_final[i, 6]
        if np.random.uniform() < p_i:
            variance = result_final[i, 4] / N_pop
        else:
            variance = result_final[i, 5] * result_final[i, 4] / N_pop
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

    # 2. 第一阶段：logistic MCMC → adj_weights
    adj_w = run_stage1(n_p, o_p, d_s, pop_data["N_pop"])

    # 3. 第二阶段：混合正态 MCMC（使用 adj_weights）
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