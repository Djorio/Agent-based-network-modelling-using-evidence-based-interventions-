"""
01_baseline_prevalence.py
Survey-weighted MOV prevalence with complex survey design variance estimation.
Uses PSU-level bootstrap within strata (Taylor linearization fallback).
"""

import numpy as np
import pandas as pd
from scipy import stats

from config import BASELINE_MOV_CI


def compute_weighted_prevalence(df: pd.DataFrame,
                                 outcome_col: str = "mov",
                                 weight_col: str = "psweight",
                                 strata_col: str = "strata",
                                 psu_col: str = "psu",
                                 n_boot: int = 1000,
                                 seed: int = 42) -> dict:
    """
    Survey-weighted prevalence with PSU bootstrap CI.
    Resamples PSUs within strata to account for clustering.
    """
    rng = np.random.default_rng(seed)
    y = df[outcome_col].values.astype(float)
    w = df[weight_col].values.astype(float)
    strata = df[strata_col].values
    psus   = df[psu_col].values

    # Point estimate
    estimate = np.average(y, weights=w)

    # PSU-level bootstrap within strata
    boot_estimates = []
    unique_strata = np.unique(strata)

    for _ in range(n_boot):
        boot_idx = []
        for s in unique_strata:
            s_mask = strata == s
            s_psus = np.unique(psus[s_mask])
            if len(s_psus) == 0:
                continue
            # Resample PSUs with replacement
            chosen_psus = rng.choice(s_psus, size=len(s_psus), replace=True)
            for psu_id in chosen_psus:
                psu_mask = (strata == s) & (psus == psu_id)
                boot_idx.extend(np.where(psu_mask)[0].tolist())

        if len(boot_idx) == 0:
            continue
        boot_idx = np.array(boot_idx)
        boot_est = np.average(y[boot_idx], weights=w[boot_idx])
        boot_estimates.append(boot_est)

    boot_arr = np.array(boot_estimates)
    se = boot_arr.std()
    ci_lo = np.percentile(boot_arr, 2.5)
    ci_hi = np.percentile(boot_arr, 97.5)

    # Design effect
    srs_se = np.sqrt(estimate * (1 - estimate) / len(df))
    deff = (se / srs_se) ** 2 if srs_se > 0 else np.nan

    print(f"[baseline] Weighted MOV prevalence: {estimate:.3f} "
          f"(95%CI: {ci_lo:.3f}–{ci_hi:.3f})")
    print(f"[baseline] SE={se:.4f}, DEFF={deff:.2f}")

    return {
        "estimate": estimate,
        "CI_lo":    ci_lo,
        "CI_hi":    ci_hi,
        "SE":       se,
        "DEFF":     deff,
        "n":        len(df),
        "boot_samples": boot_arr,
    }


def survey_weighted_crosstabs(df: pd.DataFrame,
                               barrier_cols: list,
                               weight_col: str = "psweight") -> pd.DataFrame:
    """
    For each barrier column, compute weighted MOV prevalence
    separately for agents with barrier=1 vs barrier=0.
    Returns a DataFrame used for Figure 2 subgroup bar chart.
    """
    records = []
    w = df[weight_col].values
    y = df["mov"].values.astype(float)

    for col in barrier_cols:
        x = df[col].values
        # Barrier present
        mask1 = x == 1
        prev1 = np.average(y[mask1], weights=w[mask1]) if mask1.sum() > 0 else np.nan
        # Barrier absent
        mask0 = x == 0
        prev0 = np.average(y[mask0], weights=w[mask0]) if mask0.sum() > 0 else np.nan

        records.append({
            "barrier":    col,
            "prev_barrier_present": prev1,
            "prev_barrier_absent":  prev0,
            "n_present":  int(mask1.sum()),
            "n_absent":   int(mask0.sum()),
            "weighted_prev_present": prev1,
        })

    return pd.DataFrame(records)


def subgroup_prevalence(df: pd.DataFrame,
                         strat_col: str,
                         weight_col: str = "psweight") -> pd.DataFrame:
    """
    Compute weighted MOV prevalence for each category of strat_col.
    Used for subgroup analysis in Figure 2.
    """
    records = []
    w = df[weight_col].values
    y = df["mov"].values.astype(float)
    cats = df[strat_col].values

    for cat in np.unique(cats):
        mask = cats == cat
        n = int(mask.sum())
        if n < 5:
            continue
        prev = np.average(y[mask], weights=w[mask])
        # Simple bootstrap CI
        boot = []
        rng = np.random.default_rng(0)
        for _ in range(500):
            idx = rng.choice(np.where(mask)[0], size=n, replace=True)
            boot.append(np.average(y[idx], weights=w[idx]))
        records.append({
            "category": cat,
            "stratum":  strat_col,
            "prevalence": prev,
            "CI_lo": np.percentile(boot, 2.5),
            "CI_hi": np.percentile(boot, 97.5),
            "n": n,
        })
    return pd.DataFrame(records)


if __name__ == "__main__":
    import mov_utils
    dp = mov_utils.data_prep()
    df_agents, _ = dp.prepare_agents()

    result = compute_weighted_prevalence(df_agents)
    print(f"\nTarget CI: {BASELINE_MOV_CI}")
    print(f"Estimate within CI: {BASELINE_MOV_CI[0] <= result['estimate'] <= BASELINE_MOV_CI[1]}")
