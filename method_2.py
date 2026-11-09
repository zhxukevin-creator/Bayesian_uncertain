# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from scipy.stats import norm, gamma, multivariate_normal
from mcmc_utils import (log_p, g_func, bisection_sample,
                        logp, log_rho, sample_p_bi, sample_rho_bi,
                        stable_z_prob, calculate_hpd, diagnose)

# =============================================================================
# 
# =============================================================================

N_SAMPLE = 500      
N_TOTAL  = 50000   
BURN_IN  = 10000    
TARGET   = 1000     

# =============================================================================
# MCMC 核心函数
# =============================================================================

def log_theta(n_p, o_p, y, beta, d, y_beta, sigma_2, rho, z):
    
    gi    = g_func(n_p, beta)
    fpart = sum(np.log(np.maximum(gi, 1e-300))) / 2
    spart = ((gi / rho - gi * z / rho + gi * z) @ (y - n_p @ y_beta) ** 2) / (2 * sigma_2)
    theta = np.log(beta / (1 - beta))
    tpart = log_p(n_p, o_p, theta, d)
    return fpart - spart + tpart

def sample_beta_j(j, n_p, o_p, beta_cur, d, y, beta_y, sigma_2, rho, z, it=5):
    
    def log_target(t):
        beta_tmp = beta_cur.copy()
        beta_tmp[j] = t
        return log_theta(n_p, o_p, y, beta_tmp, d, beta_y, sigma_2, rho, z)
    return bisection_sample(log_target, it)

def run_mcmc(n_p, o_p, y_s, d_s):
    
    n_comp = n_p.shape[1]   # = 4

    mixture_beta = np.zeros([N_TOTAL, n_comp])
    mixture_beta[0] = 0.5

    y_beta   = np.zeros([N_TOTAL, n_comp + 2])
    p_result = np.zeros(N_TOTAL)

    # 
    beta    = np.array([-0.01, 0.5, 0.5, 27])
    rho     = 0.5
    sigma_2 = 100.0
    p       = 0.03
   
    z = np.random.binomial(n_p.shape[0], p=0.03)

    for i in range(1, N_TOTAL):
        mixture_beta[i] = mixture_beta[i - 1]

        # ──  mixture_beta────────────────
        for j in range(n_comp):
            mixture_beta[i, j] = sample_beta_j(
                j, n_p, o_p, mixture_beta[i], d_s,
                y_s, beta, sigma_2, rho, z, it=5
            )

        gi = g_func(n_p, mixture_beta[i])

        # ── z ────────────────────────────────────────────────────────
        mean1 = n_p @ beta
        sd1   = np.sqrt(np.maximum(rho * sigma_2 / gi, 1e-15))
        sd2   = np.sqrt(np.maximum(sigma_2 / gi,       1e-15))
        pr    = stable_z_prob(y_s, mean1, sd1, sd2, p)
        z     = np.random.binomial(1, pr)

        # ── p ────────────────────────────────────────────────────────
        p = sample_p_bi(y_s, n_p, beta, rho, sigma_2, gi, it=6) / 2
        p_result[i] = p

        # ──  beta ─────────────────────────────────────────────────────
        t_vec    = gi / rho - gi * z / rho + gi * z
        # G = diag(t_vec): avoid allocating n×n matrix, use broadcasting instead
        Xw       = n_p * t_vec[:, None]          # (n,p) weighted X
        fpart    = np.linalg.inv(Xw.T @ n_p)
        spart    = Xw.T @ y_s
        beta_hat = fpart @ spart
        cov      = fpart * sigma_2
        beta     = multivariate_normal.rvs(mean=beta_hat, cov=cov)
        y_beta[i, :n_comp] = beta

        # ── sigma2 ───────────────────────────────────────────────────
        rate    = t_vec * (y_s - n_p @ beta_hat) ** 2
        r       = sum(rate) / 2
        sigma_2 = 1 / gamma.rvs(a=n_p.shape[0] / 2 - 2, scale=1 / r)
        y_beta[i, n_comp] = sigma_2

        # ── rho ──────────────────────────────────────────────────────
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
# 
# =============================================================================

def sample_ybar(y_beta_final, p_final, x_bar, N_pop):
    y_bar_samples = np.zeros(TARGET)
    for i in range(TARGET):
        mean_y = np.dot(x_bar, y_beta_final[i, 0:4])
        if np.random.uniform() < p_final[i]:
            variance = y_beta_final[i, 4] / N_pop
        else:
            variance = y_beta_final[i, 5] * y_beta_final[i, 4] / N_pop
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

    # 2. MCMC
    y_beta_final, p_final = run_mcmc(n_p, o_p, y_s, d_s)

    # 3. 
    y_bar_samples = sample_ybar(
        y_beta_final, p_final,
        pop_data["x_bar"], pop_data["N_pop"]
    )

    # 4. 
    hpd_l, hpd_u = calculate_hpd(y_bar_samples)

    # 5. 
    posterior_samples = pd.DataFrame(
        np.column_stack([y_beta_final, p_final]),
        columns=["beta0","beta1","beta2","beta3","sigma2","rho","p"]
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
