"""
run_pipeline.py
Run full simulation pipeline after calibration.
"""
import os, sys, importlib.util, numpy as np, pandas as pd, time

os.makedirs("results/posteriors", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

def load(fname, name):
    spec = importlib.util.spec_from_file_location(name, fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

print("=== Loading Data ===")
dp = load("00_data_preparation.py", "data_prep")
df_agents, fixed_log_odds = dp.prepare_agents()

print("\n=== Loading Posterior ===")
abc = load("03_abc_smc_calibration.py", "abc_smc")
posterior = abc.load_posterior()
print(f"Posterior: n={len(posterior['posterior_samples'])}, mean={np.average(posterior['posterior_samples'], weights=posterior['posterior_weights']):.4f}")

print("\n=== Step 4b: Simulation (2019-2035) ===")
t0 = time.time()
sim = load("04_simulation.py", "simulation")
results = sim.run_simulation(
    df_agents, fixed_log_odds,
    posterior["posterior_samples"],
    posterior["posterior_weights"],
    n_posterior_draws=40, n_mc_per_draw=10,  # reduced for interactive-speed demo runs
)
print(f"Simulation done in {time.time()-t0:.1f}s")
print("Packages:", list(results.keys()))
for k in ["no_intervention", "P1", "P10"]:
    if k in results:
        v = results[k]
        r2019 = v[v.year==2019]["mean_prev"].values[0]
        r2035 = v[v.year==2035]["mean_prev"].values[0]
        print(f"  {k}: 2019={r2019:.3f}, 2035={r2035:.3f}")

# Save simulation results
for k, df in results.items():
    df.to_csv(f"results/sim_{k}.csv", index=False)

averted = sim.compute_averted_children(results)
averted.to_csv("results/averted_children.csv", index=False)
print(f"Averted children saved ({len(averted)} rows)")

print("\n=== Step 5: Calibration Validation ===")
val = load("05_calibration_validation.py", "validation")
# Compute posterior-mean probabilities for validation
theta_mean = float(np.average(posterior["posterior_samples"], weights=posterior["posterior_weights"]))
from scipy.special import expit
y_pred = expit(theta_mean + fixed_log_odds)
y_true = df_agents["mov"].values.astype(float)
weights = df_agents["psweight"].values

brier = val.compute_brier_score(y_true, y_pred, weights)
print(f"Brier score: {brier['model_brier']:.4f} (null: {brier['null_brier']:.4f}, BSS: {brier['brier_skill_score']:.4f})")

auc_result = val.compute_auc_bootstrap(y_true, y_pred, n_boot=500)
print(f"AUC: {auc_result['auc']:.3f} 95%CI=[{auc_result['ci_lo']:.3f}, {auc_result['ci_hi']:.3f}]")

ppc_prevs = np.load("results/posteriors/ppc_prevalences.npy")
ppc_result = val.posterior_predictive_check(ppc_prevs, target=0.34)
print(f"PPC: mean={ppc_result['mean']:.3f}, p-value={ppc_result['p_value']:.3f}, within_95CI={ppc_result['within_95ci']}")

cal_curve = val.calibration_curve(y_true, y_pred)
shap_df = val.compute_shap_importance(df_agents)
print(f"Top SHAP barriers: {shap_df.head(3)['barrier'].tolist()}")

# Save validation metrics
val_metrics = {
    "brier": brier["brier"], "null_brier": brier["null_brier"], "bss": brier["bss"],
    "auc": auc_result["auc"], "auc_ci_lo": auc_result["ci_lo"], "auc_ci_hi": auc_result["ci_hi"],
    "ppc_mean": ppc_result["mean"], "ppc_p_value": ppc_result["p_value"],
}
pd.DataFrame([val_metrics]).to_csv("results/validation_metrics.csv", index=False)
shap_df.to_csv("results/shap_importance.csv", index=False)
print("Validation metrics saved.")

print("\n=== Step 6: Sensitivity Analysis ===")
t0 = time.time()
sa = load("06_sensitivity_analysis.py", "sensitivity")
sa_results = sa.one_way_sensitivity(df_agents, fixed_log_odds, posterior["posterior_samples"], posterior["posterior_weights"])
sa_results.to_csv("results/sensitivity_table.csv", index=False)
print(f"One-way SA done in {time.time()-t0:.1f}s, {len(sa_results)} parameters")

print("\n=== Step 7: Cost-Effectiveness Analysis ===")
cea = load("07_cost_effectiveness.py", "cea")
cea_out = cea.full_cea(results, df_agents, fixed_log_odds, posterior["posterior_samples"], n_psa=500)
icer_df = cea_out["icer_table"]
icer_df.to_csv("results/icer_table.csv", index=False)
print(f"CEA done. ICER table saved ({len(icer_df)} rows)")

print("\n=== Pipeline complete. Run 08_figures.py for figures. ===")
print(f"Results in results/ directory")
