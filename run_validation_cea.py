"""
run_validation_cea.py
Run validation, sensitivity analysis, and CEA using saved simulation results.
"""
import os, sys, importlib.util, numpy as np, pandas as pd, time
from scipy.special import expit

os.makedirs("results/posteriors", exist_ok=True)
os.makedirs("results/validation", exist_ok=True)
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
theta_mean = float(np.average(posterior["posterior_samples"], weights=posterior["posterior_weights"]))
print(f"Posterior mean theta: {theta_mean:.4f}")

print("\n=== Loading Simulation Results ===")
results = {}
for fname in os.listdir("results"):
    if fname.startswith("sim_") and fname.endswith(".csv"):
        key = fname[4:-4]  # strip 'sim_' and '.csv'
        results[key] = pd.read_csv(f"results/{fname}")
print(f"Loaded {len(results)} simulation trajectories: {list(results.keys())}")

print("\n=== Step 5: Calibration Validation ===")
val = load("05_calibration_validation.py", "validation")

y_pred = expit(theta_mean + fixed_log_odds)
y_true = df_agents["mov"].values.astype(float)
weights = df_agents["psweight"].values

# Brier score
brier = val.compute_brier_score(y_true, y_pred, weights)
print(f"Brier: model={brier['model_brier']:.4f}, null={brier['null_brier']:.4f}, BSS={brier['brier_skill_score']:.4f}")

# AUC
auc_result = val.bootstrap_auc(y_true, y_pred, n_boot=500)
print(f"AUC: {auc_result['auc']:.3f} 95%CI=[{auc_result['CI_lo']:.3f}, {auc_result['CI_hi']:.3f}]")

# DeLong test
p_null = float(np.average(y_true, weights=weights))
y_null = np.full_like(y_pred, p_null)
delong = val.delong_test(y_true, y_pred, y_null)
print(f"DeLong test: z={delong.get('z_stat', 'N/A')}, p={delong.get('p_value', 'N/A')}")

# PPC using saved prevalences
ppc_prevs = np.load("results/posteriors/ppc_prevalences.npy")
ppc_mean = ppc_prevs.mean()
ppc_lo, ppc_hi = np.percentile(ppc_prevs, [2.5, 97.5])
within_ci = bool(0.24 <= ppc_mean <= 0.43)
print(f"PPC: mean={ppc_mean:.3f}, 95%CI=[{ppc_lo:.3f}, {ppc_hi:.3f}], within target CI={within_ci}")

# Calibration curve
cal_curve = val.calibration_curve_data(y_true, y_pred)
pd.DataFrame(cal_curve).to_csv("results/validation/calibration_curve.csv", index=False)

# SHAP
print("Computing SHAP importance...")
shap_df = val.compute_shap_importance(df_agents)
feat_col = "feature" if "feature" in shap_df.columns else shap_df.columns[0]
print(f"Top SHAP barriers: {shap_df.head(3)[feat_col].tolist()}")
shap_df.to_csv("results/shap_importance.csv", index=False)

# Save validation metrics
val_metrics = {
    "model_brier": brier["model_brier"],
    "null_brier": brier["null_brier"],
    "brier_skill_score": brier["brier_skill_score"],
    "auc": auc_result["auc"],
    "auc_ci_lo": auc_result["CI_lo"],
    "auc_ci_hi": auc_result["CI_hi"],
    "ppc_mean": ppc_mean,
    "ppc_ci_lo": ppc_lo,
    "ppc_ci_hi": ppc_hi,
}
pd.DataFrame([val_metrics]).to_csv("results/validation_metrics.csv", index=False)
print("Validation metrics saved.")

print("\n=== Step 6: Sensitivity Analysis ===")
t0 = time.time()
sa = load("06_sensitivity_analysis.py", "sensitivity")
sa_results = sa.one_way_sensitivity(
    df_agents, fixed_log_odds,
    posterior["posterior_samples"], posterior["posterior_weights"],
    n_sim=50  # faster for now; use 100+ for publication
)
sa_results.to_csv("results/sensitivity_table.csv", index=False)
print(f"One-way SA done in {time.time()-t0:.1f}s, {len(sa_results)} parameters")

print("\n=== Step 7: Cost-Effectiveness Analysis ===")
cea_mod = load("07_cost_effectiveness.py", "cea")
t0 = time.time()
cea_out = cea_mod.full_cea(
    results, df_agents, fixed_log_odds,
    posterior["posterior_samples"], n_psa=500, seed=42
)
print(f"CEA done in {time.time()-t0:.1f}s")
cea_out["icer_table"].to_csv("results/icer_table.csv", index=False)
if "dominance_table" in cea_out:
    cea_out["dominance_table"].to_csv("results/dominance_table.csv", index=False)
print("CEA tables saved.")

print("\n=== All analyses complete. Run 08_figures.py for figures. ===")
