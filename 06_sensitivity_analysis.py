"""
06_sensitivity_analysis.py
One-way sensitivity analysis on every variable in the calibrated ABNM: each of the 33
barrier-term adjusted ORs, plus the ABC-SMC-calibrated intercept and the intervention
RRs, is varied across its 95% CI (holding everything else at its point estimate), and
the resulting change in the simulated weighted MOV prevalence (or, for interventions,
the 2035 package prevalence) is recorded — the standard "tornado diagram" input table.
"""

import numpy as np
import pandas as pd

from config import BARRIER_ORS, INTERVENTION_RRS, PACKAGES


def _weighted_prevalence(fixed_log_odds, intercept, weights):
    from scipy.special import expit
    p = expit(intercept + fixed_log_odds)
    return float(np.average(p, weights=weights))


def one_way_sensitivity(df_agents: pd.DataFrame,
                         fixed_log_odds: np.ndarray,
                         posterior_samples: np.ndarray,
                         posterior_weights: np.ndarray,
                         n_sim: int = 100) -> pd.DataFrame:
    """
    For each of the 33 barrier terms, recompute `fixed_log_odds` with that term's OR
    swapped for its CI_lo / CI_hi (all other terms held at their point-estimate OR),
    and report the resulting change in simulated baseline (no-intervention) prevalence,
    using the posterior-mean calibrated intercept. Also reports the same style of
    sensitivity for the free intercept itself (its ABC-SMC posterior 95% interval) and
    for each intervention RR's 95% CI (effect on the corresponding package's simulated
    prevalence, applied population-wide at the posterior-mean intercept).

    Parameters
    ----------
    n_sim : number of Monte-Carlo prevalence estimates averaged per bound (stability).
    """
    rng = np.random.default_rng(7)
    weights = df_agents["psweight"].values.astype(float)
    theta_mean = float(np.average(posterior_samples, weights=posterior_weights))
    theta_lo, theta_hi = np.quantile(posterior_samples, [0.025, 0.975])

    baseline_prev = _weighted_prevalence(fixed_log_odds, theta_mean, weights)

    records = []

    # ── (1) Barrier term ORs ──────────────────────────────────────────────────────
    terms = list(BARRIER_ORS.keys())
    log_or_point = {t: np.log(BARRIER_ORS[t]["OR"]) for t in terms}
    for term in terms:
        if term not in df_agents.columns:
            continue
        x = df_agents[term].values.astype(float)
        base_contrib = x * log_or_point[term]

        for bound_name, key in [("low", "CI_lo"), ("high", "CI_hi")]:
            or_bound = max(BARRIER_ORS[term][key], 1e-6)
            new_contrib = x * np.log(or_bound)
            flo_new = fixed_log_odds - base_contrib + new_contrib
            prev = np.mean([_weighted_prevalence(flo_new, theta_mean, weights) for _ in range(max(1, n_sim // 50))])
            records.append({
                "parameter": term, "parameter_type": "barrier_OR",
                "bound": bound_name, "bound_value": or_bound,
                "output_prevalence": prev, "delta_vs_baseline": prev - baseline_prev,
            })

    # ── (2) Calibrated intercept (theta) ────────────────────────────────────────
    for bound_name, theta_bound in [("low", theta_lo), ("high", theta_hi)]:
        prev = _weighted_prevalence(fixed_log_odds, theta_bound, weights)
        records.append({
            "parameter": "intercept_theta", "parameter_type": "calibration",
            "bound": bound_name, "bound_value": theta_bound,
            "output_prevalence": prev, "delta_vs_baseline": prev - baseline_prev,
        })

    # ── (3) Intervention RRs — effect on that package's 2035 prevalence ────────────
    from scipy.special import expit
    p_base = expit(theta_mean + fixed_log_odds)
    for int_id, params in INTERVENTION_RRS.items():
        pkg_with_only_this = [pid for pid, ivs in PACKAGES.items() if ivs == [int_id]]
        label = f"intervention_{int_id}_{params['name'].replace(' ', '_')}"
        for bound_name, key in [("low", "CI_lo"), ("high", "CI_hi")]:
            rr = params[key]
            p_new = np.clip(p_base * rr, 0.001, 0.999)
            prev = float(np.average(p_new, weights=weights))
            records.append({
                "parameter": label, "parameter_type": "intervention_RR",
                "bound": bound_name, "bound_value": rr,
                "output_prevalence": prev, "delta_vs_baseline": prev - baseline_prev,
            })

    df = pd.DataFrame(records)
    df["baseline_prevalence"] = baseline_prev
    df["abs_delta"] = df["delta_vs_baseline"].abs()

    # tornado ranking: max absolute swing (high vs low) per parameter
    swing = (df.groupby("parameter")["output_prevalence"].max() -
             df.groupby("parameter")["output_prevalence"].min())
    df["swing_rank"] = df["parameter"].map(swing.rank(ascending=False))
    df = df.sort_values(["swing_rank", "parameter", "bound"]).reset_index(drop=True)

    print(f"[sensitivity] One-way sensitivity computed for {df['parameter'].nunique()} parameters "
          f"({(df['parameter_type']=='barrier_OR').sum()//2} barrier ORs, "
          f"1 intercept, {(df['parameter_type']=='intervention_RR').sum()//2} intervention RRs)")
    return df


if __name__ == "__main__":
    import importlib.util, os

    def load(fname, name):
        spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), fname))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    dp = load("00_data_preparation.py", "data_prep")
    abc = load("03_abc_smc_calibration.py", "abc_smc")
    df_agents, fixed_log_odds = dp.prepare_agents()
    posterior = abc.load_posterior("results/posteriors/")

    sa = one_way_sensitivity(df_agents, fixed_log_odds,
                              posterior["posterior_samples"], posterior["posterior_weights"],
                              n_sim=50)
    print(sa.head(20))
