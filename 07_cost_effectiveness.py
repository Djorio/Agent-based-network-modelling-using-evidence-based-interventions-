"""
07_cost_effectiveness.py
Cost-effectiveness analysis (CEA) of the 10 intervention packages vs. no-intervention:
ICERs (US$ per MOV-averted child, 2025-2035 cumulative), a dominance table (removing
strictly/extended dominated packages), and a probabilistic sensitivity analysis (PSA)
sampling cost (gamma) and RR (log-normal) distributions to build cost-effectiveness
acceptability curves (CEAC) across config.WTP_THRESHOLDS.
"""

import numpy as np
import pandas as pd

from config import (
    COSTS_PER_CHILD, EIR_COST_STRUCTURE, PACKAGES, REFERENCE_POPULATION,
    WTP_THRESHOLDS, INTERVENTION_START, SIM_END,
)


def _gamma_params_from_ci(mean, lo, hi):
    """Method-of-moments gamma fit from an approximate 95% CrI (sd ~ (hi-lo)/(2*1.96))."""
    sd = max((hi - lo) / (2 * 1.96), 1e-6)
    shape = (mean / sd) ** 2
    scale = sd ** 2 / mean if mean > 0 else 1e-6
    return shape, scale


def sample_package_cost(package_interventions: list, rng: np.random.Generator,
                         reference_pop: int = REFERENCE_POPULATION) -> float:
    """One PSA draw of the total 5-year (2025-2035) package cost, in US$."""
    n_years = SIM_END - INTERVENTION_START + 1
    total = 0.0
    for int_id in package_interventions:
        cost_spec = COSTS_PER_CHILD.get(int_id)
        if cost_spec is None:
            # Intervention 3 (EIR): flat development + annualised per-child cost
            flat = rng.triangular(EIR_COST_STRUCTURE["flat_lo"], EIR_COST_STRUCTURE["flat_mean"],
                                   EIR_COST_STRUCTURE["flat_hi"])
            annual_total = flat + EIR_COST_STRUCTURE["annual_child"] * n_years
            total += annual_total
        else:
            mean, lo, hi = cost_spec
            shape, scale = _gamma_params_from_ci(mean, lo, hi)
            per_child = rng.gamma(shape, scale)
            total += per_child * reference_pop * n_years
    return float(total)


def compute_icers(sim_results: dict, df_agents: pd.DataFrame,
                   reference_pop: int = REFERENCE_POPULATION, seed: int = 42) -> pd.DataFrame:
    """Point-estimate ICER table: mean cost & mean averted children per package (2025-2035)."""
    from mov_utils import simulation as _sim_mod
    sim_mod = _sim_mod()
    averted_df = sim_mod.compute_averted_children(sim_results, reference_pop=reference_pop)
    cum_averted = averted_df.groupby("package")["averted_mean"].sum()

    rng = np.random.default_rng(seed)
    records = []
    for pkg, interventions in PACKAGES.items():
        cost = sample_package_cost(interventions, rng, reference_pop=reference_pop)
        averted = float(cum_averted.get(pkg, 0.0))
        icer = cost / averted if averted > 0 else np.inf
        records.append({
            "package": pkg, "interventions": "+".join(str(i) for i in interventions),
            "total_cost_usd": cost, "children_mov_averted": averted,
            "icer_usd_per_averted": icer,
        })
    df = pd.DataFrame(records).sort_values("total_cost_usd").reset_index(drop=True)
    return df


def dominance_analysis(icer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple/extended dominance: sort by cost; a package is strictly dominated if a
    cheaper package achieves >= its effect. Extended dominance via ICER non-monotonicity
    on the efficiency frontier is flagged too.
    """
    df = icer_df.sort_values("total_cost_usd").reset_index(drop=True)
    df["dominated"] = False
    df["dominance_type"] = ""

    # strict dominance: cheaper package with >= effect
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue
            cheaper_or_equal_cost = df.loc[j, "total_cost_usd"] <= df.loc[i, "total_cost_usd"]
            more_or_equal_effect = df.loc[j, "children_mov_averted"] >= df.loc[i, "children_mov_averted"]
            strictly_better = (df.loc[j, "total_cost_usd"] < df.loc[i, "total_cost_usd"] and
                               more_or_equal_effect) or (cheaper_or_equal_cost and
                               df.loc[j, "children_mov_averted"] > df.loc[i, "children_mov_averted"])
            if strictly_better:
                df.loc[i, "dominated"] = True
                df.loc[i, "dominance_type"] = "strict"

    # extended dominance on the non-dominated frontier: incremental ICER should be increasing
    frontier = df[~df["dominated"]].sort_values("children_mov_averted").reset_index(drop=True)
    frontier["incremental_cost"] = frontier["total_cost_usd"].diff()
    frontier["incremental_effect"] = frontier["children_mov_averted"].diff()
    frontier["incremental_icer"] = frontier["incremental_cost"] / frontier["incremental_effect"]

    prev_icer = -np.inf
    ext_dominated = []
    for i, row in frontier.iterrows():
        if i == 0:
            ext_dominated.append(False)
            continue
        if row["incremental_icer"] < prev_icer:
            ext_dominated.append(True)
        else:
            ext_dominated.append(False)
            prev_icer = row["incremental_icer"]
    frontier["extended_dominated"] = ext_dominated

    df = df.merge(frontier[["package", "incremental_icer", "extended_dominated"]],
                  on="package", how="left")
    df.loc[df["extended_dominated"] == True, "dominance_type"] = "extended"  # noqa: E712
    df.loc[df["extended_dominated"] == True, "dominated"] = True  # noqa: E712
    return df


def run_psa(sim_results: dict, df_agents: pd.DataFrame,
            reference_pop: int = REFERENCE_POPULATION,
            n_psa: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Probabilistic sensitivity analysis: n_psa draws of package cost (+/- effect held at
    its point estimate, since effect PSA is already captured by the simulation's own
    posterior-draw/MC uncertainty in `sim_results`). Returns long-format draws for CEAC.
    """
    from mov_utils import simulation as _sim_mod
    sim_mod = _sim_mod()
    averted_df = sim_mod.compute_averted_children(sim_results, reference_pop=reference_pop)
    cum_averted_mean = averted_df.groupby("package")["averted_mean"].sum()
    # crude effect uncertainty from the CI columns already computed by the simulation
    cum_averted_lo = averted_df.groupby("package")["averted_CI_lo"].sum()
    cum_averted_hi = averted_df.groupby("package")["averted_CI_hi"].sum()

    rng = np.random.default_rng(seed)
    records = []
    for pkg, interventions in PACKAGES.items():
        eff_mean = float(cum_averted_mean.get(pkg, 0.0))
        eff_lo = float(cum_averted_lo.get(pkg, 0.0))
        eff_hi = float(cum_averted_hi.get(pkg, eff_mean))
        eff_sd = max((eff_hi - eff_lo) / (2 * 1.96), 1e-3)

        for draw in range(n_psa):
            cost = sample_package_cost(interventions, rng, reference_pop=reference_pop)
            effect = max(rng.normal(eff_mean, eff_sd), 1e-3)
            records.append({"package": pkg, "draw": draw, "cost": cost, "effect": effect,
                             "icer": cost / effect})
    return pd.DataFrame(records)


def compute_ceac(psa_df: pd.DataFrame, wtp_thresholds: np.ndarray = None) -> pd.DataFrame:
    """Cost-effectiveness acceptability curve: P(package is optimal) at each WTP threshold."""
    if wtp_thresholds is None:
        wtp_thresholds = WTP_THRESHOLDS
    packages = psa_df["package"].unique()
    n_draws = psa_df["draw"].nunique()

    records = []
    pivot_cost = psa_df.pivot(index="draw", columns="package", values="cost")
    pivot_effect = psa_df.pivot(index="draw", columns="package", values="effect")

    for wtp in wtp_thresholds:
        nmb = wtp * pivot_effect - pivot_cost  # net monetary benefit per package per draw
        optimal = nmb.idxmax(axis=1)
        probs = optimal.value_counts(normalize=True)
        for pkg in packages:
            records.append({"wtp": wtp, "package": pkg, "prob_optimal": probs.get(pkg, 0.0)})
    return pd.DataFrame(records)


def full_cea(sim_results: dict, df_agents: pd.DataFrame, fixed_log_odds: np.ndarray,
             posterior_samples: np.ndarray, n_psa: int = 1000, seed: int = 42) -> dict:
    """Run the complete CEA: ICER table, dominance table, PSA draws, and CEAC."""
    print(f"[cea] Computing ICERs for {len(PACKAGES)} packages...")
    icer_table = compute_icers(sim_results, df_agents, seed=seed)

    print("[cea] Dominance analysis...")
    dominance_table = dominance_analysis(icer_table)

    print(f"[cea] Running PSA ({n_psa} draws per package)...")
    psa_results = run_psa(sim_results, df_agents, n_psa=n_psa, seed=seed)

    print("[cea] Computing CEAC...")
    ceac = compute_ceac(psa_results)

    print(f"[cea] Done. Cheapest package: "
          f"{icer_table.iloc[0]['package']} (${icer_table.iloc[0]['total_cost_usd']:,.0f})")

    return {"icer_table": icer_table, "dominance_table": dominance_table,
            "psa_results": psa_results, "ceac": ceac}


if __name__ == "__main__":
    import importlib.util, os

    def load(fname, name):
        spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), fname))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    dp = load("00_data_preparation.py", "data_prep")
    abc = load("03_abc_smc_calibration.py", "abc_smc")
    sim = load("04_simulation.py", "simulation")

    df_agents, fixed_log_odds = dp.prepare_agents()
    posterior = abc.load_posterior("results/posteriors/")
    results = sim.run_simulation(df_agents, fixed_log_odds,
                                  posterior["posterior_samples"], posterior["posterior_weights"],
                                  n_posterior_draws=15, n_mc_per_draw=5, seed=42)
    out = full_cea(results, df_agents, fixed_log_odds, posterior["posterior_samples"], n_psa=200)
    print(out["icer_table"])
