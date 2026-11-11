"""
03_abc_smc_calibration.py
ABC-SMC calibration of the logistic intercept theta.
Calibrates so that weighted simulated MOV prevalence ~ 34%.
Custom implementation (no pyabc dependency required).
"""

import numpy as np
import os
import time
from scipy.special import expit
from scipy.stats import gaussian_kde

from config import (
    BASELINE_MOV_2019, BASELINE_MOV_CI,
    ABC_PRIOR_LO, ABC_PRIOR_HI,
    ABC_N_PARTICLES, ABC_TOLERANCES
)
from mov_utils import network as _net_mod

def _get_net():
    nm = _net_mod()
    return nm.compute_agent_mov_probabilities, nm.simulate_mov_outcomes


def _simulate_prevalence(intercept: float,
                          fixed_log_odds: np.ndarray,
                          weights: np.ndarray,
                          rng: np.random.Generator,
                          n_mc: int = 20) -> float:
    """
    Simulate weighted MOV prevalence for a given intercept (theta).
    Averages n_mc Monte Carlo replicates for stability.
    """
    compute_agent_mov_probabilities, simulate_mov_outcomes = _get_net()
    prevs = []
    p = compute_agent_mov_probabilities(fixed_log_odds, intercept)
    for _ in range(n_mc):
        outcomes = simulate_mov_outcomes(p, rng)
        prev = np.average(outcomes, weights=weights)
        prevs.append(prev)
    return float(np.mean(prevs))


def run_abc_smc(df_agents,
                fixed_log_odds: np.ndarray,
                target: float = BASELINE_MOV_2019,
                n_particles: int = ABC_N_PARTICLES,
                tolerances: list = None,
                n_mc: int = 20,
                seed: int = 42) -> dict:
    """
    ABC-SMC with adaptive Gaussian perturbation kernel.

    Prior: theta ~ Uniform(ABC_PRIOR_LO, ABC_PRIOR_HI)
    Summary statistic: weighted mean of simulated MOV
    Distance: |simulated_prev - target|
    Tolerance schedule: progressively tightened

    Returns dict with:
        posterior_samples: accepted particles (final generation)
        posterior_weights: importance weights
        acceptance_rates: list of rates per generation
        all_generations: list of (particles, weights) per generation
        simulated_prevalences: prevalences at final accepted particles
    """
    if tolerances is None:
        tolerances = ABC_TOLERANCES

    weights_obs = df_agents["psweight"].values.astype(float)
    rng = np.random.default_rng(seed)

    t_start = time.time()
    print(f"[abc_smc] Starting ABC-SMC: target={target:.3f}, "
          f"particles={n_particles}, generations={len(tolerances)}")
    print(f"[abc_smc] Tolerance schedule: {tolerances}")

    all_generations = []
    acceptance_rates = []

    # ── Generation 0: sample from prior ──────────────────────────────────────
    particles = []
    sim_prevs = []
    n_tried = 0

    print(f"\n[abc_smc] Generation 0 (eps={tolerances[0]:.3f}): sampling from prior...")

    while len(particles) < n_particles:
        theta = rng.uniform(ABC_PRIOR_LO, ABC_PRIOR_HI)
        sim_prev = _simulate_prevalence(theta, fixed_log_odds, weights_obs, rng, n_mc=5)
        n_tried += 1

        if abs(sim_prev - target) <= tolerances[0]:
            particles.append(theta)
            sim_prevs.append(sim_prev)

        if len(particles) % 100 == 0 and len(particles) > 0:
            rate = len(particles) / n_tried
            print(f"  Accepted {len(particles)}/{n_particles} "
                  f"(rate={rate:.3f}, tried={n_tried})")

    particles = np.array(particles)
    weights_part = np.ones(n_particles) / n_particles
    acc_rate = n_particles / n_tried
    acceptance_rates.append(acc_rate)
    all_generations.append((particles.copy(), weights_part.copy()))
    print(f"[abc_smc] Gen 0 done: acceptance rate={acc_rate:.4f}, "
          f"posterior mean theta={particles.mean():.4f} +/- {particles.std():.4f}")

    # ── Generations 1+: resample → perturb → accept ───────────────────────────
    for gen_idx, tol in enumerate(tolerances[1:], start=1):
        prev_particles = particles.copy()
        prev_weights   = weights_part.copy()

        # Adaptive kernel bandwidth: 2 × posterior SD
        bandwidth = 2.0 * prev_particles.std()
        if bandwidth < 1e-6:
            bandwidth = 0.1

        new_particles = []
        new_sim_prevs = []
        n_tried_gen = 0

        print(f"\n[abc_smc] Generation {gen_idx} (eps={tol:.3f}): "
              f"bandwidth={bandwidth:.4f}...")

        while len(new_particles) < n_particles:
            # Resample from previous particles
            idx = rng.choice(n_particles, p=prev_weights)
            theta_star = prev_particles[idx]
            # Perturb with Gaussian kernel
            theta_prop = theta_star + rng.normal(0, bandwidth)

            # Reject if outside prior
            if theta_prop < ABC_PRIOR_LO or theta_prop > ABC_PRIOR_HI:
                n_tried_gen += 1
                continue

            sim_prev = _simulate_prevalence(
                theta_prop, fixed_log_odds, weights_obs, rng, n_mc=n_mc
            )
            n_tried_gen += 1

            if abs(sim_prev - target) <= tol:
                new_particles.append(theta_prop)
                new_sim_prevs.append(sim_prev)

        new_particles = np.array(new_particles)

        # Compute importance weights
        # w_i ∝ prior(theta_i) / sum_j(w_j * K(theta_i, theta_j))
        # For uniform prior and Gaussian kernel:
        kernel_denom = np.array([
            np.sum(prev_weights * _gaussian_kernel(theta_p - prev_particles, bandwidth))
            for theta_p in new_particles
        ])
        # Prior is uniform → numerator is constant
        new_weights = 1.0 / np.maximum(kernel_denom, 1e-300)
        new_weights /= new_weights.sum()

        particles   = new_particles
        weights_part = new_weights
        sim_prevs   = np.array(new_sim_prevs)

        acc_rate = len(new_particles) / n_tried_gen
        acceptance_rates.append(acc_rate)
        all_generations.append((particles.copy(), weights_part.copy()))

        eff_n = 1.0 / np.sum(weights_part ** 2)  # Effective sample size
        print(f"[abc_smc] Gen {gen_idx} done: rate={acc_rate:.4f}, "
              f"mean theta={np.average(particles, weights=weights_part):.4f}, "
              f"std={np.sqrt(np.average((particles - np.average(particles, weights=weights_part))**2, weights=weights_part)):.4f}, "
              f"ESS={eff_n:.0f}")

        if acc_rate < 0.005:
            print(f"[abc_smc] Warning: acceptance rate < 0.5%. Stopping early.")
            break

    elapsed = time.time() - t_start
    print(f"\n[abc_smc] ABC-SMC completed in {elapsed:.1f}s")
    print(f"[abc_smc] Final posterior: mean={np.average(particles, weights=weights_part):.4f}, "
          f"95%CI=[{np.quantile(particles, 0.025):.4f}, {np.quantile(particles, 0.975):.4f}]")

    return {
        "posterior_samples":     particles,
        "posterior_weights":     weights_part,
        "acceptance_rates":      acceptance_rates,
        "all_generations":       all_generations,
        "simulated_prevalences": np.array(sim_prevs) if hasattr(sim_prevs, '__len__') else np.array([sim_prevs]),
        "final_tolerance":       tolerances[len(acceptance_rates)-1],
        "target":                target,
    }


def _gaussian_kernel(diff: np.ndarray, bandwidth: float) -> np.ndarray:
    """Unnormalized Gaussian kernel K(diff, h)."""
    return np.exp(-0.5 * (diff / bandwidth) ** 2)


def posterior_predictive_prevalences(posterior_samples: np.ndarray,
                                      posterior_weights: np.ndarray,
                                      fixed_log_odds: np.ndarray,
                                      weights_obs: np.ndarray,
                                      n_draws: int = 500,
                                      n_mc: int = 30,
                                      seed: int = 99) -> np.ndarray:
    """
    Draw intercepts from posterior, simulate MOV outcomes,
    return distribution of simulated prevalences.
    Used for posterior predictive check (PPC).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(posterior_samples), size=n_draws,
                     p=posterior_weights)
    thetas = posterior_samples[idx]

    compute_agent_mov_probabilities, simulate_mov_outcomes = _get_net()
    prevs = []
    for theta in thetas:
        p = compute_agent_mov_probabilities(fixed_log_odds, theta)
        for _ in range(n_mc):
            outcomes = simulate_mov_outcomes(p, rng)
            prevs.append(np.average(outcomes, weights=weights_obs))

    return np.array(prevs)


def save_posterior(posterior: dict, path: str = "results/posteriors/") -> None:
    """Save posterior samples to disk."""
    os.makedirs(path, exist_ok=True)
    np.save(os.path.join(path, "posterior_intercepts.npy"),
            posterior["posterior_samples"])
    np.save(os.path.join(path, "posterior_weights.npy"),
            posterior["posterior_weights"])
    np.save(os.path.join(path, "acceptance_rates.npy"),
            np.array(posterior["acceptance_rates"]))
    print(f"[abc_smc] Posterior saved to {path}")


def load_posterior(path: str = "results/posteriors/") -> dict:
    """Load previously saved posterior."""
    return {
        "posterior_samples": np.load(os.path.join(path, "posterior_intercepts.npy")),
        "posterior_weights": np.load(os.path.join(path, "posterior_weights.npy")),
        "acceptance_rates":  np.load(os.path.join(path, "acceptance_rates.npy")).tolist(),
    }


if __name__ == "__main__":
    from mov_utils import data_prep as _dp
    dp = _dp()
    df_agents, fixed_log_odds = dp.prepare_agents()

    # Quick test: what intercept gives ~34% prevalence?
    from scipy.special import expit
    w = df_agents["psweight"].values
    for theta in np.linspace(-20, -10, 20):
        p = expit(theta + fixed_log_odds)
        prev = np.average(p, weights=w)
        print(f"  theta={theta:.1f} -> E[prev]={prev:.3f}")
