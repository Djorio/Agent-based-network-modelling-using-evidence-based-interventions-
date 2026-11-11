"""
05_calibration_validation.py
Validation of the calibrated ABNM: Brier score (+ skill score vs null model), bootstrap
AUC, a DeLong test comparing the model to the null (intercept-only) model, a calibration
(reliability) curve, and a variable-importance ranking across all 33 barrier terms.

Note: the `shap` package is not installable in this offline environment, so
`compute_shap_importance` uses a transparent, deterministic substitute — the mean
absolute log-odds contribution of each barrier term, i.e. |log(OR_j)| * P(term_j = 1) —
which plays the same role (ranking barriers by their typical impact on an agent's
predicted log-odds of MOV) without requiring a black-box surrogate.

Every public function is exposed under the (occasionally inconsistent) names used by the
various run_*.py driver scripts, with generous key aliases in the returned dicts, so all
of them run without modification.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve as _sk_calibration_curve

from config import BARRIER_ORS


def compute_brier_score(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray = None) -> dict:
    """Weighted Brier score for the model vs. a null (sample-prevalence) model."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-6, 1 - 1e-6)
    if weights is None:
        weights = np.ones_like(y_true)
    weights = np.asarray(weights, dtype=float)

    model_brier = float(np.average((y_true - y_pred) ** 2, weights=weights))
    p_null = float(np.average(y_true, weights=weights))
    null_brier = float(np.average((y_true - p_null) ** 2, weights=weights))
    bss = 1.0 - model_brier / null_brier if null_brier > 0 else np.nan

    return {
        "model_brier": model_brier, "null_brier": null_brier, "brier_skill_score": bss,
        # aliases for driver scripts that use shorter key names
        "brier": model_brier, "bss": bss,
    }


def bootstrap_auc(y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap 95% CI for AUC (percentile method)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    auc_point = float(roc_auc_score(y_true, y_pred))

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            continue
        boots.append(roc_auc_score(yt, yp))
    boots = np.array(boots)
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5]) if len(boots) else (np.nan, np.nan)

    return {
        "auc": auc_point, "CI_lo": float(ci_lo), "CI_hi": float(ci_hi),
        "ci_lo": float(ci_lo), "ci_hi": float(ci_hi),  # lowercase aliases
        "boot_aucs": boots,
    }


# alias used by run_pipeline.py
def compute_auc_bootstrap(y_true, y_pred, n_boot: int = 500, seed: int = 0) -> dict:
    return bootstrap_auc(y_true, y_pred, n_boot=n_boot, seed=seed)


def delong_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """
    Simplified DeLong-style test comparing two sets of predictions' AUCs via the
    normal approximation to the difference of (bootstrap-estimated) AUC variances.
    Returns z_stat and a two-sided p_value.
    """
    y_true = np.asarray(y_true)
    auc_a = roc_auc_score(y_true, pred_a)
    auc_b = roc_auc_score(y_true, pred_b)

    rng = np.random.default_rng(1)
    n = len(y_true)
    diffs = []
    for _ in range(500):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        diffs.append(roc_auc_score(yt, pred_a[idx]) - roc_auc_score(yt, pred_b[idx]))
    diffs = np.array(diffs)
    se_diff = diffs.std() if len(diffs) > 1 else np.nan
    z_stat = (auc_a - auc_b) / se_diff if se_diff and se_diff > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat))) if np.isfinite(z_stat) else np.nan

    return {"auc_a": float(auc_a), "auc_b": float(auc_b), "z_stat": float(z_stat) if np.isfinite(z_stat) else None,
            "p_value": float(p_value) if np.isfinite(p_value) else None}


def calibration_curve_data(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> dict:
    frac_pos, mean_pred = _sk_calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="quantile")
    return {"mean_predicted": mean_pred.tolist(), "fraction_positive": frac_pos.tolist()}


# alias used by run_pipeline.py
def calibration_curve(y_true, y_pred, n_bins: int = 10) -> dict:
    return calibration_curve_data(y_true, y_pred, n_bins=n_bins)


def posterior_predictive_check(ppc_prevalences: np.ndarray, target: float = 0.34) -> dict:
    """Posterior predictive check: does the target prevalence fall within the PPC distribution?"""
    ppc_prevalences = np.asarray(ppc_prevalences)
    mean_ppc = float(ppc_prevalences.mean())
    lo, hi = np.percentile(ppc_prevalences, [2.5, 97.5])
    within = bool(lo <= target <= hi)
    # two-sided posterior predictive p-value
    p_value = float(2 * min((ppc_prevalences <= target).mean(), (ppc_prevalences >= target).mean()))
    return {"mean": mean_ppc, "CI_lo": float(lo), "CI_hi": float(hi),
            "within_95ci": within, "p_value": p_value}


def compute_shap_importance(df_agents: pd.DataFrame) -> pd.DataFrame:
    """
    Variable-importance ranking across the 33 barrier terms.

    Substitute for SHAP (unavailable offline): mean absolute log-odds contribution,
    |log(OR_j)| * P(term_j = 1 | data) — the average magnitude by which each barrier
    actually moves an agent's log-odds of MOV in this dataset.
    """
    records = []
    for term, params in BARRIER_ORS.items():
        if term not in df_agents.columns:
            continue
        prevalence = float(df_agents[term].mean())
        log_or = float(np.log(params["OR"]))
        importance = abs(log_or) * prevalence
        records.append({"feature": term, "barrier": term, "log_OR": log_or,
                         "prevalence": prevalence, "importance": importance})
    out = pd.DataFrame(records).sort_values("importance", ascending=False).reset_index(drop=True)
    return out


if __name__ == "__main__":
    import importlib.util, os

    def load(fname, name):
        spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), fname))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    dp = load("00_data_preparation.py", "data_prep")
    df_agents, fixed_log_odds = dp.prepare_agents()
    y_true = df_agents["mov"].values.astype(float)
    weights = df_agents["psweight"].values
    theta_demo = -1.25
    from scipy.special import expit
    y_pred = expit(theta_demo + fixed_log_odds)

    print(compute_brier_score(y_true, y_pred, weights))
    print(bootstrap_auc(y_true, y_pred, n_boot=200))
    print(compute_shap_importance(df_agents).head(10))
