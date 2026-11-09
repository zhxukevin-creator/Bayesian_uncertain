# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from scipy.stats import gamma, multivariate_normal
from mcmc_utils import (log_p, g_func, bisection_sample, calculate_hpd, diagnose)

# =============================================================================
# 
# =============================================================================

N_SAMPLE = 500     

# stage1
N_TOTAL_1 = 50000
BURN_IN_1  = 10000
TARGET_1   = 1000
BI_ITER_1  = 5

# stage2
N_TOTAL_2 = 50000
BURN_IN_2  = 10000
TARGET_2   = 1000

# =============================================================================
# 
# =============================================================================

def stage1_sample_beta_j(j, n_p, o_p, beta_cur, d, it=7):
    def log_target(t):
        beta_tmp = beta_cur.copy()
        beta_tmp[j] = t
        theta = np.log(beta_tmp / (1 - beta_tmp))
        return log_p(n_p, o_p, theta, d)
    return bisection_sample(log_target, it)

def run_stage1(n_p, o_p, d_s, N_pop):
    
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

    
    beta_bar  = beta_samples.mean(axis=0)
    theta_bar = np.log(beta_bar / (1 - beta_bar))
    lin       = n_p @ theta_bar
    ps        = np.exp(lin) / (1 + np.exp(lin))

    raw_w = 1.0 / ps
    adj_w = N_pop * raw_w / raw_w.sum()

    return adj_w, beta_samples   # 同时返回 stage1 的1000个后验样本

# =============================================================================
# 
# =============================================================================

def run_stage2(n_p, y_s, adj_w):
    
    n_s    = n_p.shape[0]
    n_comp = n_p.shape[1]

    result  = np.zeros((N_TOTAL_2, n_comp + 1))
    beta    = np.array([-0.01, 0.5, 0.5, 27])
    sigma_2 = 100.0

    for i in range(N_TOTAL_2):
        # ── beta────────────────────────────
        # W = diag(adj_w): avoid allocating n×n matrix
        Xw       = n_p * adj_w[:, None]
        fpart    = np.linalg.inv(Xw.T @ n_p)
        spart    = Xw.T @ y_s
        beta_hat = fpart @ spart
        beta     = multivariate_normal.rvs(mean=beta_hat, cov=fpart * sigma_2)
        result[i, :n_comp] = beta

        # ── sigma2 ───────────────────────────────────────────────────
        rate    = adj_w * (y_s - n_p @ beta_hat) ** 2
        r       = sum(rate) / 2
        sigma_2 = 1 / gamma.rvs(a=n_s / 2 - 2, scale=1 / r)
        result[i, n_comp] = sigma_2

    # Burn-in + Thinning
    burned = result[BURN_IN_2:]
    thin   = burned.shape[0] // TARGET_2
    return burned[::thin][:TARGET_2]

# =============================================================================
# y_bar 
# =============================================================================

def sample_ybar(result_final, x_bar, N_pop):
    
    y_bar_samples = np.zeros(TARGET_2)
    for i in range(TARGET_2):
        mean_y   = x_bar @ result_final[i, 0:4]
        variance = result_final[i, 4] / N_pop
        y_bar_samples[i] = np.random.normal(loc=mean_y, scale=np.sqrt(variance))
    return y_bar_samples

def run(pop_data, seed):
    
    np.random.seed(seed)   # 种子由调用方传入（如 ROUND*1000 + sim_id）

    
    if "ps_nonprop" in pop_data and "ps_prop" in pop_data \
            and pop_data.get("do_sampling", True):
        u_n = np.random.uniform(size=len(pop_data["ps_nonprop"]))
        idx = np.where(u_n < pop_data["ps_nonprop"])[0]
        u_o = np.random.uniform(size=len(pop_data["ps_prop"]))
        ids = np.where(u_o < pop_data["ps_prop"])[0]
        n_p = pop_data["n_p"][idx]
        o_p = pop_data["o_p"][ids]
        y_s = pop_data["y"][idx]
        d_s = pop_data["d_b"][ids]      # d_b = 1/selection probability（权重）
    else:
        n_p = pop_data["n_p"]
        o_p = pop_data["o_p"]
        y_s = pop_data["y"]
        d_s = pop_data["d_b"]

    
    adj_w, stage1_samples = run_stage1(n_p, o_p, d_s, pop_data["N_pop"])

    
    result_final = run_stage2(n_p, y_s, adj_w)

    
    y_bar_samples = sample_ybar(
        result_final,
        pop_data["x_bar"], pop_data["N_pop"]
    )

    
    hpd_l, hpd_u = calculate_hpd(y_bar_samples)

    
    posterior_samples = pd.DataFrame(
        result_final, columns=["beta0","beta1","beta2","beta3","sigma2"]
    )
    stage1_posterior = pd.DataFrame(
        stage1_samples, columns=["beta0","beta1","beta2","beta3"]
    )
    diag = diagnose(posterior_samples)
    return {
        "posterior_mean":    float(np.mean(y_bar_samples)),
        "posterior_std":     float(np.std(y_bar_samples, ddof=1)),
        "hpd_lower":         float(hpd_l),
        "hpd_upper":         float(hpd_u),
        "posterior_samples": posterior_samples,
        "stage1_samples":    stage1_posterior,
        "diagnostics":       diag,
    }
