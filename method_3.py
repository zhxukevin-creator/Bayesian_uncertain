# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
method_3.py
===========
第3种抽样方法：改写自 uncertainty_logit_product_rule_a.py

与 method_1/2 的主要区别：
    1. 无混合模型：没有 p、rho、z 参数
    2. y_beta 只有5列：[beta(4), sigma2]，无 rho
    3. beta/sigma2 更新直接用 gi 作为权重，不需要 t_vec
    4. bi_iter=7（method_1 是10）
    5. N_TOTAL=80000

⚠️  y_bar 方差：原始代码中 variance 未定义（疑似bug），
    此处暂用 sigma2/N_pop 作为占位，请确认后修改。

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
N_TOTAL  = 50000    # MCMC总迭代次数（原始代码为80000）
BURN_IN  = 10000    # burn-in丢弃数
TARGET   = 1000     # 最终保留的后验样本数
BI_ITER  = 5        # 二分采样迭代次数（原始代码为7）

# =============================================================================
# MCMC 核心函数
# =============================================================================

def sample_beta_j(j, n_p, o_p, beta_cur, d, it=7):
    """
    采样 mixture_beta 的第 j 个分量。
    log target 与 method_1 相同（log_p），bi_iter=7。
    """
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
        y_beta_final : shape (TARGET, 5)，列为 [beta(4), sigma2]
                       注意：无 rho，比 method_1/2 少一列
    """
    n_comp = n_p.shape[1]   # = 4

    mixture_beta = np.zeros([N_TOTAL, n_comp])
    mixture_beta[0] = 0.5

    y_beta  = np.zeros([N_TOTAL, n_comp + 1])   # beta(4) + sigma2，无 rho
    sigma_2 = 1.0                                # 初始值（原始代码为1）

    for i in range(1, N_TOTAL):
        mixture_beta[i] = mixture_beta[i - 1]

        # ── 更新 mixture_beta（log_p 作为 log target，bi_iter=7）──────────
        for j in range(n_comp):
            mixture_beta[i, j] = sample_beta_j(
                j, n_p, o_p, mixture_beta[i], d_s, it=BI_ITER
            )

        gi = g_func(n_p, mixture_beta[i])

        # ── 更新 beta（直接用 gi，无 t_vec）──────────────────────────────
        # G = diag(gi): avoid allocating n×n matrix
        Xw       = n_p * gi[:, None]
        fpart    = np.linalg.inv(Xw.T @ n_p)
        spart    = Xw.T @ y_s
        beta_hat = fpart @ spart
        cov      = fpart * sigma_2
        beta     = multivariate_normal.rvs(mean=beta_hat, cov=cov)
        y_beta[i, :n_comp] = beta

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
    从后验样本生成 y_bar 样本。
    ⚠️  原始代码中 variance 未定义，暂用 sigma2/N_pop。
        请确认正确的方差公式后修改此处。
    """
    y_bar_samples = np.zeros(TARGET)
    for i in range(TARGET):
        mean_y   = np.dot(x_bar, y_beta_final[i, 0:4])
        variance = y_beta_final[i, 4] / N_pop   # TODO: 确认方差公式
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

    # 2. MCMC（注意：无 p_final，只返回 y_beta_final）
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