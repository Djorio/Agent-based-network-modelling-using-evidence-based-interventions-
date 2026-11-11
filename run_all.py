"""
run_all.py
Master pipeline runner — runs ALL steps end-to-end.

Usage:
    python run_all.py [--skip-calibration] [--skip-simulation]

Flags:
    --skip-calibration  Skip ABC-SMC (use saved posterior in results/posteriors/)
    --skip-simulation   Skip 2019-2035 simulation (use saved sim_*.csv files)

NOTE ON SPEED SETTINGS: the particle/draw/MC counts below are reduced from the
publication-scale defaults in config.py (ABC_N_PARTICLES=1000, N_POSTERIOR_DRAWS=200,
N_MC_PER_DRAW=50, PSA=1000) so the full pipeline completes in a couple of minutes in an
interactive/offline environment. For a full production run, pass the config constants
directly instead of the FAST_* overrides below.
"""
import os
import time
import argparse
import importlib.util

import numpy as np
import pandas as pd
from scipy.special import expit

# ── fast-run overrides (see note above) ───────────────────────────────────────
FAST_ABC_PARTICLES = 300
FAST_ABC_TOLERANCES = [0.15, 0.10, 0.05, 0.03, 0.02]
FAST_ABC_N_MC = 10
FAST_POSTERIOR_DRAWS = 40
FAST_MC_PER_DRAW = 10
FAST_PSA = 300

os.makedirs("results/posteriors", exist_ok=True)
os.makedirs("results/validation", exist_ok=True)
os.makedirs("results/writeup", exist_ok=True)
os.makedirs("figures", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--skip-calibration", action="store_true")
parser.add_argument("--skip-simulation", action="store_true")
args, _ = parser.parse_known_args()


def load(fname, name):
    spec = importlib.util.spec_from_file_location(name, fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


pipeline_start = time.time()

section("Step 0: Data Preparation")
dp = load("00_data_preparation.py", "data_prep")
df_agents, fixed_log_odds = dp.prepare_agents()

section("Step 1: Baseline Prevalence")
bp = load("01_baseline_prevalence.py", "baseline")
baseline_result = bp.compute_weighted_prevalence(df_agents, n_boot=200)
pd.DataFrame([baseline_result]).to_csv("results/baseline_prevalence.csv", index=False)
print(f"Baseline: {baseline_result['estimate']*100:.1f}% "
      f"[{baseline_result['CI_lo']*100:.1f}%\u2013{baseline_result['CI_hi']*100:.1f}%]")

section("Step 2: Build Network")
nm = load("02_network_model.py", "network_model")
G = nm.build_barrier_network()

section("Step 3: ABC-SMC Calibration")
abc = load("03_abc_smc_calibration.py", "abc_smc")
if args.skip_calibration and os.path.exists("results/posteriors/posterior_intercepts.npy"):
    print("Skipping calibration; loading saved posterior.")
    posterior = abc.load_posterior()
else:
    posterior = abc.run_abc_smc(
        df_agents, fixed_log_odds,
        n_particles=FAST_ABC_PARTICLES, tolerances=FAST_ABC_TOLERANCES,
        n_mc=FAST_ABC_N_MC, seed=42,
    )
    abc.save_posterior(posterior)

theta_mean = float(np.average(posterior["posterior_samples"], weights=posterior["posterior_weights"]))
print(f"Posterior mean theta: {theta_mean:.4f}")

if not os.path.exists("results/posteriors/ppc_prevalences.npy"):
    ppc_prevs = abc.posterior_predictive_prevalences(
        posterior["posterior_samples"], posterior["posterior_weights"],
        fixed_log_odds, df_agents["psweight"].values, n_draws=150, n_mc=15,
    )
    np.save("results/posteriors/ppc_prevalences.npy", ppc_prevs)
else:
    ppc_prevs = np.load("results/posteriors/ppc_prevalences.npy")
print(f"PPC: mean={ppc_prevs.mean():.3f}, "
      f"95%HDPI=[{np.percentile(ppc_prevs, 2.5):.3f}, {np.percentile(ppc_prevs, 97.5):.3f}]")

section("Step 4: Simulation (2019\u20132035)")
sim = load("04_simulation.py", "simulation")
if args.skip_simulation and os.path.exists("results/sim_no_intervention.csv"):
    print("Skipping simulation; loading saved results.")
    results = {}
    for fname in os.listdir("results"):
        if fname.startswith("sim_") and fname.endswith(".csv"):
            key = fname[4:-4]
            results[key] = pd.read_csv(f"results/{fname}")
    print(f"Loaded {len(results)} trajectories.")
else:
    t0 = time.time()
    results = sim.run_simulation(
        df_agents, fixed_log_odds,
        posterior["posterior_samples"], posterior["posterior_weights"],
        n_posterior_draws=FAST_POSTERIOR_DRAWS, n_mc_per_draw=FAST_MC_PER_DRAW,
    )
    print(f"Simulation complete in {time.time()-t0:.1f}s")
    for k, df in results.items():
        df.to_csv(f"results/sim_{k}.csv", index=False)

averted = sim.compute_averted_children(results)
averted.to_csv("results/averted_children.csv", index=False)

section("Step 5: Calibration & Validation")
val = load("05_calibration_validation.py", "validation")
y_pred = expit(theta_mean + fixed_log_odds)
y_true = df_agents["mov"].values.astype(float)
weights = df_agents["psweight"].values

brier = val.compute_brier_score(y_true, y_pred, weights)
auc_r = val.bootstrap_auc(y_true, y_pred, n_boot=500)
p_null = float(np.average(y_true, weights=weights))
delong = val.delong_test(y_true, y_pred, np.full_like(y_pred, p_null))
cal_curve = val.calibration_curve_data(y_true, y_pred)
shap_df = val.compute_shap_importance(df_agents)

pd.DataFrame(cal_curve).to_csv("results/validation/calibration_curve.csv", index=False)
shap_df.to_csv("results/shap_importance.csv", index=False)
pd.DataFrame([{
    "model_brier": brier["model_brier"], "null_brier": brier["null_brier"],
    "brier_skill_score": brier["brier_skill_score"],
    "auc": auc_r["auc"], "auc_ci_lo": auc_r["CI_lo"], "auc_ci_hi": auc_r["CI_hi"],
    "delong_z": delong.get("z_stat"), "delong_p": delong.get("p_value"),
    "ppc_mean": ppc_prevs.mean(), "ppc_ci_lo": np.percentile(ppc_prevs, 2.5),
    "ppc_ci_hi": np.percentile(ppc_prevs, 97.5),
}]).to_csv("results/validation_metrics.csv", index=False)
print(f"AUC: {auc_r['auc']:.3f}, BSS: {brier['brier_skill_score']:.4f}, "
      f"Top barrier by importance: {shap_df.iloc[0]['feature']}")

section("Step 6: Sensitivity Analysis")
t0 = time.time()
sa = load("06_sensitivity_analysis.py", "sensitivity")
sa_results = sa.one_way_sensitivity(
    df_agents, fixed_log_odds,
    posterior["posterior_samples"], posterior["posterior_weights"],
    n_sim=100,
)
sa_results.to_csv("results/sensitivity_table.csv", index=False)
print(f"One-way SA: {sa_results['parameter'].nunique()} parameters in {time.time()-t0:.1f}s")

section("Step 7: Cost-Effectiveness Analysis")
cea_mod = load("07_cost_effectiveness.py", "cea")
t0 = time.time()
cea_out = cea_mod.full_cea(
    results, df_agents, fixed_log_odds,
    posterior["posterior_samples"], n_psa=FAST_PSA, seed=42,
)
cea_out["icer_table"].to_csv("results/icer_table.csv", index=False)
cea_out["dominance_table"].to_csv("results/dominance_table.csv", index=False)
cea_out["psa_results"].to_csv("results/psa_results.csv", index=False)
cea_out["ceac"].to_csv("results/ceac.csv", index=False)
print(f"CEA complete in {time.time()-t0:.1f}s")

section("Step 8: Generate Figures")
t0 = time.time()
figs = load("08_figures.py", "figures")
figs.generate_all_figures(
    G=G, baseline_result=baseline_result, df_agents=df_agents, ppc_prevs=ppc_prevs,
    sim_results=results, posterior=posterior, icer_table=cea_out["icer_table"],
    sa_results=sa_results, validation_results=pd.read_csv("results/validation_metrics.csv").iloc[0].to_dict(),
    psa_results=cea_out["psa_results"], ceac=cea_out["ceac"], cal_curve=cal_curve,
)
print(f"All figures saved to figures/ in {time.time()-t0:.1f}s")

section("Step 9: Statistical Writeup")
writeup = load("09_statistical_writeup.py", "writeup")
writeup.generate_writeup()

total = time.time() - pipeline_start
section(f"Pipeline Complete in {total/60:.1f} minutes")
print("Outputs:")
print(f"  figures/  \u2014 {len([f for f in os.listdir('figures') if f.endswith('.png')])} PNG figures")
print(f"  results/  \u2014 {len([f for f in os.listdir('results') if os.path.isfile(os.path.join('results', f))])} result files")
print("  results/writeup/ \u2014 methods.txt, results.txt")
