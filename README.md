# Bayesian Predictive Inference using a Non-Probability Sample with Uncertainty about Propensity Scores

This repository contains the sample code for the six methods in the paper *Bayesian Predictive Inference using a Non-Probability Sample with Uncertainty about Propensity Scores*, the supplemental material (SM), and a sample data.

The code is **sampling and inference code only**: it draws from the posterior distributions and produces posterior summaries of the finite-population mean. **Checking the independence / convergence of the MCMC draws is the user's responsibility** (see [Diagnostics](#diagnostics)).

## Methods

| File | Method | Description |
|---|---|---|
| `method_1.py` | UNP_R | Propensity uncertainty propagated, product-rule, robust model |
| `method_2.py` | UNNP_R | Propensity uncertainty propagated, no-product-rule, robust model |
| `method_3.py` | UNP | As method 1, non-robust model |
| `method_4.py` | UNNP | As method 2, non-robust model |
| `method_5.py` | tr_r | Two-stage: propensity score fixed at its mean, robust model |
| `method_6.py` | tr | Two-stage: propensity score fixed at its mean, non-robust model |

Methods 1–4 propagate the uncertainty of the estimated propensity scores through the weights. Methods 5–6 first estimate the propensity-score parameters (stage 1), form fixed adjusted weights from their posterior mean, and then fit the outcome model (stage 2).

Other files:

- `mcmc_utils.py` – shared functions: bisection sampler, logistic propensity / weight functions, HPD interval, and the diagnostics (`ess`, `geweke_z`, `diagnose`).
- `simlate_parallelv3.py` – runs the simulation study in parallel (resumable).
- `simdata_0.3.csv` – sample data.
- `Supplemental-Material/` – SM of the paper.

## Requirements

Python 3.9+ with `numpy`, `scipy`, `pandas`. No other dependencies.

## Data format

`simdata_0.3.csv` has a header row and one row per population unit:

| Column | Name | Meaning |
|---|---|---|
| 0 | `y1` | outcome |
| 1–4 | `x1`, `x2`, `x3`, `X4` | covariates (design matrix) |
| 5 | `propsenty` | selection probability used to draw the probability sample |
| 6 | `propsenty_n` | selection probability used to draw the non-probability sample |

For each simulation replicate two samples are drawn from the population by **Poisson sampling** (each unit enters independently with its own selection probability, so the expected sample size is the sum of the probabilities):

- non-probability sample: covariates `n_p` and outcome `y`
- probability sample: covariates `o_p` and weights `d_b = 1 / selection probability`

The selection probabilities must therefore sum to the intended expected sample sizes. `simlate_parallelv3.py` prints the expected sizes at start-up; check them.

## Quick start

```bash
python simlate_parallelv3.py
```

Settings at the top of `simlate_parallelv3.py`:

- `ROUND` – simulation round. The seed of replicate `sim_id` is `ROUND*1000 + sim_id`, so changing `ROUND` gives a new, independent set of replicates, and the same `ROUND` reproduces the same results.
- `N_SIM` (in `__main__`) – number of replicates (default 100).
- `N_WORKERS` – number of parallel processes (each method runs in its own process; reduce if memory is limited).

Output: one file per method, `method_k_round{ROUND}_results.csv`, with one row per replicate and written as it finishes (an interrupted run resumes from the last finished replicate):

`sim_id, posterior_mean, posterior_std, hpd_lower, hpd_upper, min_ess, max_abs_geweke`

On Windows keep the `if __name__ == "__main__":` guard; run in the background on Linux with `nohup python simlate_parallelv3.py > simulate.log 2>&1 &`.

## Using a method on your own data

Each method exposes `run(pop_data, seed)`:

```python
import numpy as np, method_1

pop_data = {
    "y": y, "n_p": n_p, "o_p": o_p,      # non-probability data and probability-sample covariates
    "d_b": d_b,                          # weights (1 / selection probability) of o_p
    "x_bar": x_bar, "N_pop": N_pop,      # population covariate means and size
    "do_sampling": False,                # data are already sampled: use them as given
}
out = method_1.run(pop_data, seed=1)

out["posterior_mean"], out["posterior_std"], out["hpd_lower"], out["hpd_upper"]
out["posterior_samples"]   # DataFrame, 1000 draws after burn-in and thinning
out["diagnostics"]         # DataFrame: ESS and Geweke z per parameter
```

With `do_sampling=True` (default) and `ps_nonprop` / `ps_prop` in `pop_data`, `run` first draws the two Poisson samples.

Returned posterior draws (1000 per parameter):

| Methods | `posterior_samples` columns | extra |
|---|---|---|
| 1, 2, 5 | `beta0..beta3, sigma2, rho, p` | |
| 3, 4, 6 | `beta0..beta3, sigma2` | |
| 5, 6 | | `stage1_samples`: `beta0..beta3` (propensity-model parameters) |

## MCMC settings

Set at the top of each `method_k.py`: total iterations (`N_TOTAL`), burn-in (`BURN_IN`), number of retained draws (`TARGET = 1000`, obtained by thinning), and the bisection iterations (`BI_ITER`, `P_ITER`, `RHO_ITER`). The posterior of the finite-population mean is obtained from the retained draws, and the HPD interval is the shortest 95% interval.

## Diagnostics

Thinning to 1000 draws does **not** guarantee that the draws are independent. Before using the results, check:

- **Effective sample size (ESS)** – an ESS far below 1000 means strongly autocorrelated draws; increase `N_TOTAL` or the thinning gap.
- **Geweke z-score** – compares the first 10% and last 50% of the chain; `|z| > 1.96` suggests the chain has not reached stationarity; increase `BURN_IN`.

Both are computed by `diagnose` in `mcmc_utils.py` and returned in `out["diagnostics"]`; the simulation CSVs report the worst case per replicate (`min_ess`, `max_abs_geweke`). They can also be applied to any saved draws:

```python
from mcmc_utils import diagnose
diagnose(out["posterior_samples"])
```

## Citation

If you use this code, please cite the paper above.
