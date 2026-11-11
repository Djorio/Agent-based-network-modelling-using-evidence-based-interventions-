"""
04_simulation.py
Project MOV prevalence 2019-2035 under no-intervention and 10 packages.

Methodology:
  - Draw N_POSTERIOR_DRAWS intercept values from the ABC-SMC posterior
  - For each draw: sample RRs from log-normal distributions, run N_MC_PER_DRAW
    Monte Carlo Bernoulli replicates per year, average replicate outcomes
  - Aggregate across posterior draws: mean + 95% percentile interval per year
  - COVID adjustment (2020-2023): +COVID_MOV_ADJUSTMENT applied as absolute
    shift to the simulated weighted prevalence (not to individual probabilities)
  - Intervention period (2025-2035): apply package RRs via apply_intervention_rrs
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.special import expit  # noqa: F401 – kept for callers that import it

from config import (
    PACKAGES,
    INTERVENTION_RRS,
    COVID_YEARS,
    COVID_MOV_ADJUSTMENT,
    YEARS,
    INTERVENTION_START,
    SIM_START,
    SIM_END,
    REFERENCE_POPULATION,
    N_POSTERIOR_DRAWS,
    N_MC_PER_DRAW,
)

# ── import helpers from digit-prefixed modules via mov_utils ──────────────────
from mov_utils import network as _net_mod

def _get_nm():
    nm = _net_mod()
    return (
        nm.compute_agent_mov_probabilities,
        nm.simulate_mov_outcomes,
        nm.apply_intervention_rrs,
        nm.sample_rrs_from_distributions,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RR sampling  (re-exported so callers do not need to touch 02_network_model)
# ─────────────────────────────────────────────────────────────────────────────

def sample_rrs_from_distributions(n_samples: int = 1, seed: int = None) -> dict:
    """
    Sample RR values from log-normal distributions for every intervention.

    Parameters
    ----------
    n_samples : int
        Number of samples per intervention.  If 1, returns scalar per key.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        {int_id: float} when n_samples == 1, else {int_id: np.ndarray}.
    """
    _, _, _, _sample_rrs = _get_nm()
    return _sample_rrs(n_samples=n_samples, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# Core simulation
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_one_draw(
    df_agents: pd.DataFrame,
    fixed_log_odds: np.ndarray,
    intercept: float,
    draw_seed: int,
    n_mc: int,
) -> dict:
    """
    Simulate one posterior-intercept draw across all years, all scenarios.

    Returns dict mapping scenario_key → {year: mean_prevalence}.
    Scenario keys: "no_intervention", "P1", …, "P10".
    """
    (compute_probs, sim_outcomes, apply_rrs, sample_rrs) = _get_nm()

    weights = df_agents["psweight"].values.astype(float)
    rng = np.random.default_rng(draw_seed)

    # Sample one RR set for this draw (all interventions)
    rr_samples = sample_rrs(n_samples=1, seed=draw_seed)

    results: dict[str, dict[int, float]] = {
        "no_intervention": {},
        **{pkg: {} for pkg in PACKAGES},
    }

    for year in YEARS:
        # ── baseline probabilities (intercept fixed for this draw) ───────────
        p_base = compute_probs(fixed_log_odds, intercept)

        # ── no-intervention scenario ─────────────────────────────────────────
        if year in COVID_YEARS:
            # Draw outcomes then add COVID adjustment to weighted prevalence
            mc_prevs = []
            for _ in range(n_mc):
                outcomes = sim_outcomes(p_base, rng)
                prev = float(np.average(outcomes, weights=weights))
                mc_prevs.append(prev)
            no_int_prev = float(np.mean(mc_prevs)) + COVID_MOV_ADJUSTMENT
        else:
            mc_prevs = []
            for _ in range(n_mc):
                outcomes = sim_outcomes(p_base, rng)
                prev = float(np.average(outcomes, weights=weights))
                mc_prevs.append(prev)
            no_int_prev = float(np.mean(mc_prevs))

        results["no_intervention"][year] = no_int_prev

        # ── package scenarios (only for intervention years) ──────────────────
        for pkg_name, pkg_interventions in PACKAGES.items():
            if year < INTERVENTION_START:
                # Pre-intervention: same trajectory as no-intervention
                results[pkg_name][year] = no_int_prev
            else:
                # Apply RR multipliers
                p_int = apply_rrs(
                    df_agents=df_agents,
                    fixed_log_odds=fixed_log_odds,
                    package_interventions=pkg_interventions,
                    rr_samples=rr_samples,
                    intercept=intercept,
                )
                mc_prevs_int = []
                for _ in range(n_mc):
                    outcomes = sim_outcomes(p_int, rng)
                    prev = float(np.average(outcomes, weights=weights))
                    mc_prevs_int.append(prev)
                results[pkg_name][year] = float(np.mean(mc_prevs_int))

    return results


def run_simulation(
    df_agents: pd.DataFrame,
    fixed_log_odds: np.ndarray,
    posterior_samples: np.ndarray,
    posterior_weights: np.ndarray,
    n_posterior_draws: int = N_POSTERIOR_DRAWS,
    n_mc_per_draw: int = N_MC_PER_DRAW,
    seed: int = 42,
) -> dict:
    """
    Full projection of MOV prevalence 2019-2035 for no-intervention + 10 packages.

    Parameters
    ----------
    df_agents : pd.DataFrame
        Agent-level data with binary barrier columns and ``psweight``.
    fixed_log_odds : np.ndarray
        Precomputed fixed log-odds array (shape n_agents,).
    posterior_samples : np.ndarray
        Array of posterior intercept draws from ABC-SMC.
    posterior_weights : np.ndarray
        Importance weights corresponding to posterior_samples.
    n_posterior_draws : int
        Number of intercept draws to use (default: 200).
    n_mc_per_draw : int
        Monte Carlo replicates per draw per year (default: 50).
    seed : int
        Master random seed.

    Returns
    -------
    dict
        {"no_intervention": DataFrame, "P1": DataFrame, …, "P10": DataFrame}
        Each DataFrame has columns: year, mean_prev, CI_lo, CI_hi.
    """
    t0 = time.time()
    print(f"[simulation] Starting projection: {n_posterior_draws} posterior draws, "
          f"{n_mc_per_draw} MC reps/draw, years {SIM_START}-{SIM_END}")

    rng_master = np.random.default_rng(seed)

    # Sample intercepts (with replacement, weighted)
    posterior_weights_norm = posterior_weights / posterior_weights.sum()
    draw_indices = rng_master.choice(
        len(posterior_samples),
        size=n_posterior_draws,
        replace=True,
        p=posterior_weights_norm,
    )
    intercepts = posterior_samples[draw_indices]

    # Unique seeds for each draw to ensure reproducibility while avoiding
    # correlation between draws
    draw_seeds = rng_master.integers(0, 2**31, size=n_posterior_draws)

    # Accumulate prevalence per scenario × year across draws
    scenario_keys = ["no_intervention"] + list(PACKAGES.keys())
    # all_draw_prevs[scenario][draw_idx][year]
    all_draw_prevs: dict[str, list] = {k: [] for k in scenario_keys}

    for i, (intercept, dseed) in enumerate(zip(intercepts, draw_seeds)):
        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"[simulation]   Draw {i+1}/{n_posterior_draws} "
                  f"(intercept={intercept:.4f}, elapsed={elapsed:.1f}s)")

        draw_result = _simulate_one_draw(
            df_agents=df_agents,
            fixed_log_odds=fixed_log_odds,
            intercept=intercept,
            draw_seed=int(dseed),
            n_mc=n_mc_per_draw,
        )
        for key in scenario_keys:
            # Store ordered list of (year → prev) for this draw
            all_draw_prevs[key].append(
                [draw_result[key][yr] for yr in YEARS]
            )

    # Build summary DataFrames
    output: dict[str, pd.DataFrame] = {}
    for key in scenario_keys:
        arr = np.array(all_draw_prevs[key])  # shape (n_draws, n_years)
        mean_prev = arr.mean(axis=0)
        ci_lo = np.percentile(arr, 2.5, axis=0)
        ci_hi = np.percentile(arr, 97.5, axis=0)
        output[key] = pd.DataFrame(
            {
                "year": YEARS,
                "mean_prev": mean_prev,
                "CI_lo": ci_lo,
                "CI_hi": ci_hi,
            }
        )

    elapsed = time.time() - t0
    print(f"[simulation] Projection complete in {elapsed:.1f}s")
    _report_summary(output)
    return output


def _report_summary(sim_results: dict) -> None:
    """Print a brief prevalence summary for 2019, 2023, 2025, 2030, 2035."""
    check_years = [y for y in [2019, 2023, 2025, 2030, 2035] if y in YEARS]
    no_int = sim_results["no_intervention"]
    print("\n[simulation] No-intervention prevalence summary:")
    for yr in check_years:
        row = no_int[no_int["year"] == yr].iloc[0]
        print(f"  {yr}: {row.mean_prev:.3f} "
              f"[{row.CI_lo:.3f}, {row.CI_hi:.3f}]")

    print("[simulation] 2035 package prevalences (mean):")
    for pkg in PACKAGES:
        df_pkg = sim_results[pkg]
        row = df_pkg[df_pkg["year"] == SIM_END].iloc[0]
        print(f"  {pkg}: {row.mean_prev:.3f} "
              f"[{row.CI_lo:.3f}, {row.CI_hi:.3f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Derived outputs
# ─────────────────────────────────────────────────────────────────────────────

def compute_averted_children(
    sim_results: dict,
    reference_pop: int = REFERENCE_POPULATION,
) -> pd.DataFrame:
    """
    Compute children with MOV averted relative to no-intervention, per year.

    Only covers years 2025-2035 (INTERVENTION_START to SIM_END).

    Parameters
    ----------
    sim_results : dict
        Output of run_simulation().
    reference_pop : int
        Cohort size to scale prevalence differences.

    Returns
    -------
    pd.DataFrame
        Columns: package, year, averted_mean, averted_CI_lo, averted_CI_hi,
                 cumulative_averted_mean.
    """
    int_years = [y for y in YEARS if y >= INTERVENTION_START]
    no_int_df = sim_results["no_intervention"]

    records = []
    for pkg in PACKAGES:
        pkg_df = sim_results[pkg]
        cumulative = 0.0
        for yr in int_years:
            ni_row = no_int_df[no_int_df["year"] == yr].iloc[0]
            pk_row = pkg_df[pkg_df["year"] == yr].iloc[0]

            diff_mean = ni_row.mean_prev - pk_row.mean_prev
            diff_lo   = ni_row.CI_lo - pk_row.CI_hi   # conservative lower
            diff_hi   = ni_row.CI_hi - pk_row.CI_lo   # conservative upper

            averted_mean = max(0.0, diff_mean * reference_pop)
            averted_lo   = max(0.0, diff_lo   * reference_pop)
            averted_hi   = max(0.0, diff_hi   * reference_pop)

            cumulative += averted_mean
            records.append(
                {
                    "package": pkg,
                    "year": yr,
                    "averted_mean": averted_mean,
                    "averted_CI_lo": averted_lo,
                    "averted_CI_hi": averted_hi,
                    "cumulative_averted_mean": cumulative,
                }
            )

    df_averted = pd.DataFrame(records)
    print(f"[simulation] Averted children computed for {len(PACKAGES)} packages, "
          f"years {int_years[0]}-{int_years[-1]}")
    return df_averted


def save_simulation_results(
    results: dict,
    path: str = "results/",
) -> None:
    """
    Persist each trajectory DataFrame and the averted-children table as CSV.

    Parameters
    ----------
    results : dict
        Output of run_simulation().
    path : str
        Directory for output CSVs (created if absent).
    """
    os.makedirs(path, exist_ok=True)

    for scenario_key, df in results.items():
        fname = os.path.join(path, f"trajectory_{scenario_key}.csv")
        df.to_csv(fname, index=False, float_format="%.6f")
        print(f"[simulation] Saved {fname}")

    # Also save averted-children summary
    averted = compute_averted_children(results)
    averted_path = os.path.join(path, "averted_children.csv")
    averted.to_csv(averted_path, index=False, float_format="%.2f")
    print(f"[simulation] Saved {averted_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from mov_utils import data_prep as _dp, abc_smc as _abc

    print("[simulation] Loading agent data …")
    dp = _dp()
    df_agents, fixed_log_odds = dp.prepare_agents()

    print("[simulation] Running ABC-SMC calibration …")
    abc = _abc()
    posterior = abc.run_abc_smc(df_agents, fixed_log_odds)

    print("[simulation] Running full projection …")
    sim_results = run_simulation(
        df_agents=df_agents,
        fixed_log_odds=fixed_log_odds,
        posterior_samples=posterior["posterior_samples"],
        posterior_weights=posterior["posterior_weights"],
        n_posterior_draws=N_POSTERIOR_DRAWS,
        n_mc_per_draw=N_MC_PER_DRAW,
        seed=42,
    )

    save_simulation_results(sim_results, path="results/")
    print("[simulation] Done.")
