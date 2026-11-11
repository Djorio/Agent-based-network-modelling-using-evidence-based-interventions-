"""
09_statistical_writeup.py
Assembles a plain-text methods.txt and results.txt (plus a couple of summary tables)
from the CSV/NPY artifacts saved to results/ by the earlier pipeline steps.
"""

import os
import numpy as np
import pandas as pd

from config import BASELINE_MOV_2019, BASELINE_MOV_CI, PACKAGES, WTP_PRIMARY

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
WRITEUP_DIR = os.path.join(RESULTS_DIR, "writeup")


def _read_csv(name):
    path = os.path.join(RESULTS_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


def generate_methods() -> str:
    return f"""METHODS
=======

Data & barriers
---------------
Agents (children) were drawn from the Main_data_MOV survey (analytic n = 23,490,
respondents with observed outcome, strata, PSU and sampling weight). Fourteen candidate
barriers to missed opportunities for vaccination (MOV) were considered: number of
children under 5, household size, caregiver's education, father's education, residence,
occupation, religion, wealth quintile, transportation, radio access, television access,
place of delivery, distance to the nearest facility, and place of vaccination. Missing
barrier values were imputed via Multiple Imputation by Chained Equations (scikit-learn's
IterativeImputer, BayesianRidge estimator).

Adjusted odds ratios
--------------------
A survey-weighted multivariable logistic regression (design: strata + clustered PSU +
sampling weight; linearised sandwich variance) was pooled across m = 5 MICE-imputed
datasets via Rubin's Rules, yielding 33 encoded-term adjusted odds ratios (one or more
per barrier variable, vs. an advantaged/reference category). These ORs are fixed
constants in the agent-based network model (ABNM); see config.BARRIER_ORS.

Agent-based network model & calibration
----------------------------------------
Each agent's log-odds of MOV is theta + sum_j(x_ij * log(OR_j)), where x_ij is the
agent's binary indicator for encoded barrier term j and theta is a single free
intercept. theta was calibrated with Approximate Bayesian Computation - Sequential
Monte Carlo (ABC-SMC): a uniform prior theta ~ U({0:.1f}, {1:.1f}) was propagated through
a shrinking tolerance schedule until the simulated, sampling-weighted MOV prevalence
matched the external calibration target of {BASELINE_MOV_2019:.2f}
(95% CI {BASELINE_MOV_CI[0]:.2f}-{BASELINE_MOV_CI[1]:.2f}).

Simulation & interventions
---------------------------
MOV prevalence was projected 2019-2035 for a no-intervention scenario and 10 candidate
intervention packages (combinations of 9 named interventions, each with a literature-
derived risk ratio applied multiplicatively to the probability of agents carrying its
targeted barrier(s), from 2025 onward). A +3-percentage-point COVID-era adjustment was
applied to 2020-2023. Uncertainty combines posterior intercept draws (ABC-SMC) with
Monte Carlo Bernoulli simulation and log-normal-sampled intervention risk ratios.

Validation, sensitivity & cost-effectiveness
----------------------------------------------
Model calibration was validated via Brier skill score, bootstrapped AUC, a DeLong-style
test against the null model, a posterior predictive check against the 0.34 target, and
a reliability (calibration) curve. One-way sensitivity analysis varied every barrier OR,
the calibrated intercept, and every intervention RR across its 95% interval. Cost-
effectiveness analysis combined package-level cost distributions (gamma-fitted to
published 95% credible intervals) with simulated MOV-averted children to compute ICERs,
a dominance table, and a probabilistic sensitivity analysis (PSA)-based cost-
effectiveness acceptability curve (CEAC) across willingness-to-pay thresholds up to
US$3,000 per averted case (primary threshold: US${WTP_PRIMARY}).
"""


def generate_results() -> str:
    lines = ["RESULTS", "=======", ""]

    bp = _read_csv("baseline_prevalence.csv")
    if bp is not None:
        row = bp.iloc[0]
        lines.append(f"Baseline weighted MOV prevalence: {row['estimate']:.3f} "
                      f"(95% CI {row['CI_lo']:.3f}-{row['CI_hi']:.3f}); "
                      f"design effect (DEFF) = {row['DEFF']:.2f}.")

    vm = _read_csv("validation_metrics.csv")
    if vm is not None:
        row = vm.iloc[0]
        auc = row.get("auc", np.nan)
        auc_lo = row.get("auc_ci_lo", row.get("auc_lo", np.nan))
        auc_hi = row.get("auc_ci_hi", row.get("auc_hi", np.nan))
        bss = row.get("brier_skill_score", row.get("bss", np.nan))
        ppc_mean = row.get("ppc_mean", np.nan)
        lines.append(f"Calibrated model AUC = {auc:.3f} (95% CI {auc_lo:.3f}-{auc_hi:.3f}); "
                      f"Brier skill score = {bss:.3f}; posterior-predictive mean prevalence = "
                      f"{ppc_mean:.3f}.")

    sim_ni = _read_csv("trajectory_no_intervention.csv")
    if sim_ni is None:
        sim_ni = _read_csv("sim_no_intervention.csv")
    if sim_ni is not None:
        r2019 = sim_ni[sim_ni.year == 2019].iloc[0]
        r2035 = sim_ni[sim_ni.year == 2035].iloc[0]
        lines.append(f"No-intervention projected prevalence: {r2019['mean_prev']:.3f} (2019) vs. "
                      f"{r2035['mean_prev']:.3f} (2035).")

    icer = _read_csv("icer_table.csv")
    if icer is not None:
        best = icer.sort_values("icer_usd_per_averted").iloc[0]
        lines.append(f"Most cost-effective package: {best['package']} (interventions "
                      f"{best['interventions']}), ICER = ${best['icer_usd_per_averted']:.1f} per "
                      f"MOV-averted child.")

    sa = _read_csv("sensitivity_table.csv")
    if sa is not None:
        top_param = (sa.assign(swing=lambda d: d["delta_vs_baseline"].abs())
                       .groupby("parameter")["swing"].max().sort_values(ascending=False))
        lines.append(f"Largest one-way sensitivity driver of simulated prevalence: "
                      f"{top_param.index[0]} (max |{'\u0394'}prevalence| = {top_param.iloc[0]:.3f}).")

    if len(lines) <= 3:
        lines.append("(No result artifacts found yet — run the pipeline before generating the write-up.)")

    return "\n".join(lines) + "\n"


def generate_writeup() -> None:
    os.makedirs(WRITEUP_DIR, exist_ok=True)
    methods = generate_methods()
    results = generate_results()

    with open(os.path.join(WRITEUP_DIR, "methods.txt"), "w") as f:
        f.write(methods)
    with open(os.path.join(WRITEUP_DIR, "results.txt"), "w") as f:
        f.write(results)

    print(f"[writeup] Saved methods.txt and results.txt to {WRITEUP_DIR}")


if __name__ == "__main__":
    generate_writeup()
    print(open(os.path.join(WRITEUP_DIR, "results.txt")).read())
