"""
run_calibration.py
Run ABC-SMC calibration and save posterior. Run this first before full pipeline.
"""
import os, sys, importlib.util, numpy as np

os.makedirs("results/posteriors", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

def load(fname, name):
    spec = importlib.util.spec_from_file_location(name, fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

print("=== Step 1: Data Preparation ===")
dp = load("00_data_preparation.py", "data_prep")
df_agents, fixed_log_odds = dp.prepare_agents()

print("\n=== Step 2: Baseline Prevalence ===")
bp = load("01_baseline_prevalence.py", "baseline")
baseline_result = bp.compute_weighted_prevalence(df_agents, n_boot=200)
import pandas as pd
pd.DataFrame([baseline_result]).to_csv("results/baseline_prevalence.csv", index=False)

print("\n=== Step 3: Build Network ===")
nm = load("02_network_model.py", "network_model")
G = nm.build_barrier_network()

print("\n=== Step 4: ABC-SMC Calibration ===")
abc = load("03_abc_smc_calibration.py", "abc_smc")
posterior = abc.run_abc_smc(
    df_agents, fixed_log_odds,
    n_particles=200,   # reduced for speed; increase to 1000 for publication
    tolerances=[0.15, 0.10, 0.05, 0.03],
    n_mc=10,
    seed=42,
)
abc.save_posterior(posterior)

# Quick PPC
ppc_prevs = abc.posterior_predictive_prevalences(
    posterior["posterior_samples"],
    posterior["posterior_weights"],
    fixed_log_odds,
    df_agents["psweight"].values,
    n_draws=200,
    n_mc=20,
)
print(f"\nPPC: mean={ppc_prevs.mean():.3f}, "
      f"95%HDPI=[{np.percentile(ppc_prevs, 2.5):.3f}, {np.percentile(ppc_prevs, 97.5):.3f}]")
np.save("results/posteriors/ppc_prevalences.npy", ppc_prevs)

print("\n=== Calibration complete. Run run_all.py for full pipeline. ===")
print(f"Posterior mean intercept: {np.average(posterior['posterior_samples'], weights=posterior['posterior_weights']):.4f}")
