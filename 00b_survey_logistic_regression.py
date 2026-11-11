"""
00b_survey_logistic_regression.py
==================================
Survey-weighted multivariable logistic regression of MOV on the 14 barrier
variables (33 encoded terms), pooled across m=5 MICE-imputed datasets via
Rubin's Rules, with cluster-robust ("sandwich") standard errors clustered on
PSU-within-strata to approximate the complex-survey design (statsmodels /
survey packages are unavailable in this offline environment, so the
weighted-IRLS + cluster-sandwich estimator below is implemented directly in
NumPy; this is the same pseudo-MLE + Binder-type sandwich approach used by
Stata's `svy: logistic` / R's `survey::svyglm`, aside from the finite-
population correction and any additional between-stratum variance component,
which are second-order refinements here).

Output
------
- results/adjusted_ORs_pooled.csv           (pooled OR, 95% CI, p per term)
- figures/fig0_adjusted_OR_coefplot.png/.pdf  (publishable coefplot, OR>1 only)
"""

import os
import numpy as np
import pandas as pd

from config import BARRIER_ORS, BARRIER_LABELS

M_IMPUTATIONS = 5
RNG_SEEDS = [11, 22, 33, 44, 55]


# ─────────────────────────────────────────────────────────────────────────
# 1. Weighted logistic regression (IRLS) with cluster-robust sandwich SEs
# ─────────────────────────────────────────────────────────────────────────

def weighted_logistic_irls(X, y, w, max_iter=100, tol=1e-10):
    """
    Pseudo-MLE logistic regression with case (sampling) weights w, via IRLS.
    X: (n,p) design matrix incl. intercept column. y: (n,) in {0,1}. w: (n,)
    Returns beta (p,), and fitted probabilities p_hat (n,).
    """
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        eta = np.clip(eta, -35, 35)
        mu = 1.0 / (1.0 + np.exp(-eta))
        var = np.clip(mu * (1 - mu), 1e-10, None)
        Wdiag = w * var
        z = eta + (y - mu) / var
        XtW = X.T * Wdiag
        A = XtW @ X
        b = XtW @ z
        beta_new = np.linalg.solve(A + 1e-10 * np.eye(p), b)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = np.clip(X @ beta, -35, 35)
    p_hat = 1.0 / (1.0 + np.exp(-eta))
    return beta, p_hat


def cluster_sandwich_cov(X, y, w, beta, p_hat, cluster_id):
    """
    Binder-type cluster-robust ('sandwich') covariance for weighted logistic
    regression, clustering the summed score contributions by PSU-within-strata.
    """
    n, p = X.shape
    var = np.clip(p_hat * (1 - p_hat), 1e-10, None)
    bread_inv = X.T @ (X * (w * var)[:, None])       # (p,p) = X'WX
    bread = np.linalg.inv(bread_inv + 1e-10 * np.eye(p))

    score_i = X * (w * (y - p_hat))[:, None]          # (n,p) per-observation score
    df = pd.DataFrame(score_i, columns=[f"c{j}" for j in range(p)])
    df["cluster"] = np.asarray(cluster_id)
    cluster_sums = df.groupby("cluster").sum().values  # (n_clusters, p)

    meat = cluster_sums.T @ cluster_sums               # (p,p)
    n_clust = cluster_sums.shape[0]
    # small-sample cluster correction (as in Stata's default)
    correction = (n_clust / (n_clust - 1)) * ((n - 1) / (n - p))
    cov = correction * (bread @ meat @ bread)
    return cov


def fit_one_imputation(df_encoded, terms, outcome="mov", weight_col="psweight",
                        cluster_col="cluster_id"):
    X = df_encoded[terms].values.astype(float)
    X = np.column_stack([np.ones(len(X)), X])
    y = df_encoded[outcome].values.astype(float)
    w = df_encoded[weight_col].values.astype(float)
    # normalize weights to sum to n (numerical stability; does not change beta)
    w = w * (len(w) / w.sum())
    cluster_id = df_encoded[cluster_col].values

    beta, p_hat = weighted_logistic_irls(X, y, w)
    cov = cluster_sandwich_cov(X, y, w, beta, p_hat, cluster_id)
    return beta, cov


# ─────────────────────────────────────────────────────────────────────────
# 2. Multiple imputation (m=5) + Rubin's Rules pooling
# ─────────────────────────────────────────────────────────────────────────

def multiple_impute_and_fit(analytic_df, raw_barriers, cat_order, term_definitions,
                             m=M_IMPUTATIONS, seeds=RNG_SEEDS):
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    terms = list(term_definitions.keys())
    code_maps = {c: {cat: i for i, cat in enumerate(cats)} for c, cats in cat_order.items()}
    inv_code_maps = {c: {i: cat for cat, i in mm.items()} for c, mm in code_maps.items()}
    max_code = {c: len(cats) - 1 for c, cats in cat_order.items()}

    X_num = pd.DataFrame(index=analytic_df.index)
    for c in raw_barriers:
        X_num[c] = analytic_df[c].map(code_maps[c]).astype(float)
    X_num["_mov_aux"] = analytic_df["mov"].astype(float)
    X_num["_logweight_aux"] = np.log(analytic_df["psweight"].astype(float).clip(lower=1e-6))

    betas, covs = [], []
    for m_i, seed in enumerate(seeds[:m]):
        imputer = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True,
                                    max_iter=15, random_state=seed, min_value=0,
                                    max_value=max(max_code.values()), verbose=0)
        arr = imputer.fit_transform(X_num.values)
        X_imp = pd.DataFrame(arr, columns=X_num.columns, index=X_num.index)

        out = analytic_df.copy()
        for c in raw_barriers:
            code = X_imp[c].round().clip(0, max_code[c]).astype(int)
            out[c] = code.map(inv_code_maps[c])
        for term, (col, cat) in term_definitions.items():
            out[term] = (out[col] == cat).astype(int)

        beta, cov = fit_one_imputation(out, terms)
        betas.append(beta)
        covs.append(cov)
        print(f"[survey_logit] Imputation {m_i+1}/{m} fitted "
              f"(n={len(out)}, {len(terms)} terms).")

    betas = np.array(betas)      # (m, p)
    covs = np.array(covs)        # (m, p, p)

    beta_bar = betas.mean(axis=0)
    W_bar = covs.mean(axis=0)                                   # within-imputation var
    B = np.cov(betas.T, bias=False) if betas.shape[0] > 1 else np.zeros_like(W_bar)
    if betas.shape[1] == 1:
        B = np.array([[np.var(betas[:, 0], ddof=1)]])
    total_var = W_bar + (1 + 1.0 / m) * B                        # Rubin's Rules
    se = np.sqrt(np.clip(np.diag(total_var), 0, None))

    return beta_bar, se, terms


def build_pooled_or_table(beta_bar, se, terms):
    from scipy import stats
    or_vals = np.exp(beta_bar[1:])       # drop intercept
    lo = np.exp(beta_bar[1:] - 1.96 * se[1:])
    hi = np.exp(beta_bar[1:] + 1.96 * se[1:])
    z = beta_bar[1:] / se[1:]
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))

    df = pd.DataFrame({
        "term": terms, "OR": or_vals, "CI_lo": lo, "CI_hi": hi,
        "p_value": p, "log_OR": beta_bar[1:], "SE_log_OR": se[1:],
    })
    df["label"] = df["term"].map(lambda t: BARRIER_LABELS.get(t, t).replace("\n", " "))
    df["intercept_log_odds"] = beta_bar[0]
    return df


def run(force=False):
    import importlib.util

    def load(fname, name):
        spec = importlib.util.spec_from_file_location(name, fname)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    dp = load("00_data_preparation.py", "data_prep")
    analytic = dp._load_and_clean()

    beta_bar, se, terms = multiple_impute_and_fit(
        analytic, dp.RAW_BARRIERS, dp.CAT_ORDER, dp.TERM_DEFINITIONS)
    or_table = build_pooled_or_table(beta_bar, se, terms)

    os.makedirs("results", exist_ok=True)
    or_table.to_csv("results/adjusted_ORs_pooled.csv", index=False)
    print(f"\n[survey_logit] Pooled intercept (log-odds): {beta_bar[0]:.4f} "
          f"(implied prevalence at mean covariate profile is NOT 0.34 - the "
          f"ABNM's free intercept is separately calibrated by ABC-SMC to hit "
          f"that target; see 03_abc_smc_calibration.py).")
    print(or_table[["term", "OR", "CI_lo", "CI_hi", "p_value"]].to_string(index=False))
    return or_table


if __name__ == "__main__":
    run()
