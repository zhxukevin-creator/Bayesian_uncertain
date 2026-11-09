# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from scipy.stats import norm, gamma, multivariate_normal
from mcmc_utils import (log_p, g_func, bisection_sample,
                        logp, log_rho, sample_p_bi, sample_rho_bi,
                        calculate_hpd, diagnose)

# =============================================================================
# 方法参数
# =============================================================================

N_SAMPLE = 500      
N_TOTAL  = 50000    
BURN_IN  = 10000    
TARGET   = 1000     
BI_ITER  = 5        

# =============================================================================
# 
# =============================================================================

def sample_beta_j(j, n_p, o_p, beta_cur, d, it=7):
    
    def log_target(t):
        beta_tmp = beta_cur.copy()
        beta_tmp[j] = t
        theta = np.log(beta_tmp / (1 - beta_tmp))
        return log_p(n_p, o_p, theta, d)
    return bisection_sample(log_target, it)

# =============================================================================
# 
# =============================================================================

def run_mcmc(n_p, o_p, y_s, d_s):
    
    n_comp = n_p.shape[1]   # = 4

    mixture_beta = np.zeros([N_TOTAL, n_comp])
    mixture_beta[0] = 0.5

    y_beta  = np.zeros([N_TOTAL, n_comp + 1])   
    sigma_2 = 1.0                                

    for i in range(1, N_TOTAL):
        mixture_beta[i] = mixture_beta[i - 1]

        # ───────────
        for j in range(n_comp):
            mixture_beta[i, j] = sample_beta_j(
                j, n_p, o_p, mixture_beta[i], d_s, it=BI_ITER
            )

        gi = g_func(n_p, mixture_beta[i])

        # ── beta──────────────────────────────
        # G = diag(gi): avoid allocating n×n matrix
        Xw       = n_p * gi[:, None]
        fpart    = np.linalg.inv(Xw.T @ n_p)
        spart    = Xw.T @ y_s
        beta_hat = fpart @ spart
        cov      = fpart * sigma_2
        beta     = multivariate_normal.rvs(mean=beta_hat, cov=cov)
        y_beta[i, :n_comp] = beta

        # ── sigma2 ───────────────────────────────────────────────────
        rate    = gi @ (y_s - n_p @ beta_hat) ** 2 / 2
        sigma_2 = 1 / gamma.rvs(a=n_p.shape[0] / 2 - 2, scale=1 / rate)
        y_beta[i, n_comp] = sigma_2

    # ── Burn-in + Thinning ────────────────────────────────────────────────
    y_beta_burned = y_beta[BURN_IN:]
    thin          = y_beta_burned.shape[0] // TARGET

    y_beta_final = y_beta_burned[::thin][:TARGET]

    return y_beta_final

# =============================================================================
# 
# =============================================================================

def sample_ybar(y_beta_final, x_bar, N_pop):
    
    y_bar_samples = np.zeros(TARGET)
    for i in range(TARGET):
        mean_y   = np.dot(x_bar, y_beta_final[i, 0:4])
        variance = y_beta_final[i, 4] / N_pop   # TODO: 确认方差公式
        y_bar_samples[i] = np.random.normal(loc=mean_y, scale=np.sqrt(variance))
    return y_bar_samples

def run(pop_data, seed):
    
    np.random.seed(seed)   

   
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

    # 2.
    y_beta_final = run_mcmc(n_p, o_p, y_s, d_s)

    # 3. 
    y_bar_samples = sample_ybar(
        y_beta_final,
        pop_data["x_bar"], pop_data["N_pop"]
    )

    # 4. 
    hpd_l, hpd_u = calculate_hpd(y_bar_samples)

    # 5. 
    posterior_samples = pd.DataFrame(
        y_beta_final,
        columns=["beta0","beta1","beta2","beta3","sigma2"]
    )
    diag = diagnose(posterior_samples)
    return {
        "posterior_mean":    float(np.mean(y_bar_samples)),
        "posterior_std":     float(np.std(y_bar_samples, ddof=1)),
        "hpd_lower":         float(hpd_l),
        "hpd_upper":         float(hpd_u),
        "posterior_samples": posterior_samples,
        "diagnostics":       diag,
    }
