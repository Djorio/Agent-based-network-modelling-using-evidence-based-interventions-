#!/usr/bin/env python3
"""
Surrogate Model Benchmarking for MOV ABM Calibration
===================================================

Compares Logistic Regression, Random Forest, Gradient Boosting, and XGBoost
as candidate surrogates for accelerating ABC-SMC calibration of the MOV
agent-based network model (ABNM).

REVISION NOTE: features are restricted to the ABNM's own 33 encoded barrier
terms (config.BARRIER_ORS / df_agents), i.e. exactly the variables in the
calibrated ABNM DAG - no raw survey columns outside the ABNM (agegr, gender,
insurance, internet, telephone, newspaper, etc.) and no geospatial variables
are used, so the surrogate is learning the same input->output mapping as the
mechanistic model it is meant to approximate.

Calculates 95% confidence intervals for all performance metrics via
bootstrap resampling of the held-out test set.
"""

import os
import warnings
import importlib.util

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, brier_score_loss, roc_curve)
from sklearn.calibration import calibration_curve

# ── Compatibility shim: xgboost is unavailable in this offline environment ──────
try:
    from xgboost import XGBClassifier
    _HAS_REAL_XGBOOST = True
except ImportError:
    _HAS_REAL_XGBOOST = False

    class XGBClassifier(GradientBoostingClassifier):
        """Drop-in-ish substitute for xgboost.XGBClassifier using sklearn's
        GradientBoostingClassifier (predict_proba + feature_importances_ preserved).
        Unsupported xgboost-specific kwargs (colsample_bytree, use_label_encoder,
        eval_metric, n_jobs) are accepted, stored (for sklearn's get_params contract)
        and ignored; scale_pos_weight is emulated via sample_weight at fit time."""

        def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                     subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1.0,
                     random_state=42, use_label_encoder=False, eval_metric="logloss",
                     n_jobs=-1):
            self.colsample_bytree = colsample_bytree
            self.scale_pos_weight = scale_pos_weight
            self.use_label_encoder = use_label_encoder
            self.eval_metric = eval_metric
            self.n_jobs = n_jobs
            super().__init__(n_estimators=n_estimators, max_depth=max_depth,
                              learning_rate=learning_rate, subsample=subsample,
                              random_state=random_state)

        def fit(self, X, y, **kwargs):
            y_arr = np.asarray(y)
            sample_weight = np.where(y_arr == 1, self.scale_pos_weight, 1.0)
            return super().fit(X, y, sample_weight=sample_weight)

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "font.size": 10,
    "font.family": "DejaVu Sans", "errorbar.capsize": 5,
    "lines.linewidth": 2.0, "axes.linewidth": 1.2,
})
COLORS = sns.color_palette("colorblind", n_colors=8).as_hex()
MODELS_ORDER = ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"]


def _load(fname, name):
    spec = importlib.util.spec_from_file_location(name, fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_abnm_features():
    """
    Load the ABNM's own agent-level feature matrix: the 33 binary encoded
    barrier terms (X) and the MOV outcome (y) - i.e. exactly the inputs the
    mechanistic ABNM uses, so the surrogate approximates the same mapping.
    """
    dp = _load("00_data_preparation.py", "data_prep")
    df_agents, _ = dp.prepare_agents()
    terms = list(dp.TERM_DEFINITIONS.keys())
    X = df_agents[terms].copy()
    y = df_agents["mov"].astype(int)
    print(f"[surrogate] Using {X.shape[1]} ABNM barrier terms as features "
          f"(n={len(X)}). Class distribution: {y.value_counts(normalize=True).to_dict()}")
    return X, y


def bootstrap_evaluate(model, X_test, y_test, n_iterations=200, model_name="Model"):
    metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1", "auc_roc", "brier_score"]}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError(f"{model_name} must have predict_proba")
    preds = model.predict(X_test)

    n_samples = len(y_test)
    y_test_arr = np.asarray(y_test)
    rng = np.random.default_rng(0)
    for _ in range(n_iterations):
        idx = rng.integers(0, n_samples, n_samples)
        yt, pr, pb = y_test_arr[idx], preds[idx], probs[idx]
        metrics["accuracy"].append(accuracy_score(yt, pr))
        metrics["precision"].append(precision_score(yt, pr, zero_division=0))
        metrics["recall"].append(recall_score(yt, pr, zero_division=0))
        metrics["f1"].append(f1_score(yt, pr, zero_division=0))
        metrics["auc_roc"].append(roc_auc_score(yt, pb) if len(np.unique(yt)) > 1 else np.nan)
        metrics["brier_score"].append(brier_score_loss(yt, pb))

    out = {}
    for k, v in metrics.items():
        v = np.asarray(v, dtype=float)
        v = v[~np.isnan(v)]
        out[k] = {"mean": float(np.mean(v)), "lower": float(np.percentile(v, 2.5)),
                   "upper": float(np.percentile(v, 97.5)), "std": float(np.std(v))}
    return out, probs, preds


def main():
    os.makedirs("results/writeup", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    X, y = load_abnm_features()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum() if y_train.mean() < 0.5 else 1.0
    print(f"[surrogate] scale_pos_weight for XGBoost: {scale_pos:.3f}"
          f" (using {'real xgboost' if _HAS_REAL_XGBOOST else 'GradientBoosting substitute'})")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42,
                                                 n_jobs=-1, class_weight="balanced"),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                                          learning_rate=0.1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8,
                                  scale_pos_weight=scale_pos, random_state=42,
                                  use_label_encoder=False, eval_metric="logloss", n_jobs=-1),
    }

    all_results, all_probs_preds, all_train_probs = {}, {}, {}
    for name, model in models.items():
        print(f"[surrogate] Fitting {name} ...")
        Xtr = X_train_scaled if name == "Logistic Regression" else X_train
        Xte = X_test_scaled if name == "Logistic Regression" else X_test
        model.fit(Xtr, y_train)
        results, probs, preds = bootstrap_evaluate(model, Xte, y_test, model_name=name)
        all_results[name] = results
        all_probs_preds[name] = (Xte, probs, preds)
        all_train_probs[name] = model.predict_proba(Xtr)[:, 1]

        fname = f"models/surrogate_{name.lower().replace(' ', '_')}.joblib"
        joblib.dump((model, scaler) if name == "Logistic Regression" else model, fname)

    rows = []
    for model_name, metrics in all_results.items():
        for metric_name, values in metrics.items():
            rows.append({"Model": model_name, "Metric": metric_name,
                         "Mean": values["mean"], "Lower": values["lower"],
                         "Upper": values["upper"], "Std": values["std"]})
    results_df = pd.DataFrame(rows)
    results_df.to_csv("results/writeup/surrogate_comparison_results.csv", index=False)

    best_model = max(all_results.keys(), key=lambda m: all_results[m]["auc_roc"]["mean"])
    print(f"[surrogate] Best surrogate by test AUC-ROC: {best_model}")

    # ── Combined metrics bar figure ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, metric in enumerate(["accuracy", "auc_roc", "brier_score", "f1"]):
        ax = axes[i // 2, i % 2]
        subset = results_df[results_df["Metric"] == metric].set_index("Model").reindex(MODELS_ORDER).reset_index()
        ax.bar(subset["Model"], subset["Mean"],
               yerr=[subset["Mean"] - subset["Lower"], subset["Upper"] - subset["Mean"]],
               capsize=5, color=COLORS[:4], alpha=0.85, error_kw={"linewidth": 2, "capthick": 2})
        ax.set_title(f'Model Comparison: {metric.replace("_", " ").title()}', fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_xticklabels(subset["Model"], rotation=15, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/surrogate_comparison_metrics.png", bbox_inches="tight")
    plt.savefig("figures/surrogate_comparison_metrics.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── Precision / recall figure ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for i, metric in enumerate(["precision", "recall"]):
        ax = axes[i]
        subset = results_df[results_df["Metric"] == metric].set_index("Model").reindex(MODELS_ORDER).reset_index()
        ax.bar(subset["Model"], subset["Mean"],
               yerr=[subset["Mean"] - subset["Lower"], subset["Upper"] - subset["Mean"]],
               capsize=5, color=COLORS[:4], alpha=0.85, error_kw={"linewidth": 2, "capthick": 2})
        ax.set_title(f'Model Comparison: {metric.title()}', fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_xticklabels(subset["Model"], rotation=15, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/surrogate_comparison_pr.png", bbox_inches="tight")
    plt.savefig("figures/surrogate_comparison_pr.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── Calibration (reliability) curves, all 4 surrogates ──────────────────
    fig, ax = plt.subplots(figsize=(8, 8))
    for idx, name in enumerate(MODELS_ORDER):
        Xte, probs, _ = all_probs_preds[name]
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
        ax.plot(mean_pred, frac_pos, "s-", color=COLORS[idx], linewidth=2, label=name)
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax.set_ylabel("Fraction of Positives", fontweight="bold")
    ax.set_title("Surrogate Calibration (Reliability) Curves - Test Set", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("figures/surrogate_calibration_curves.png", bbox_inches="tight")
    plt.savefig("figures/surrogate_calibration_curves.pdf", bbox_inches="tight")
    plt.close(fig)

    # ── Clean summary table ─────────────────────────────────────────────────
    metric_order = ["accuracy", "precision", "recall", "f1", "auc_roc", "brier_score"]
    metric_display = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
                       "f1": "F1-score", "auc_roc": "AUC-ROC", "brier_score": "Brier score"}
    summary_table = (results_df[results_df["Metric"].isin(metric_order)]
                      .pivot(index="Model", columns="Metric", values="Mean")
                      .reindex(columns=metric_order).reindex(MODELS_ORDER)
                      .rename(columns=metric_display).round(4))
    summary_table.to_csv("results/writeup/surrogate_metrics_summary.csv")
    print("\n[surrogate] Test-set performance summary (mean of 200 bootstrap resamples):")
    print(summary_table.to_string())

    # ── Training-set ROC curves, all 4 surrogates + random-classifier line ──
    fig, ax = plt.subplots(figsize=(8, 8))
    for idx, name in enumerate(MODELS_ORDER):
        fpr, tpr, _ = roc_curve(y_train, all_train_probs[name])
        train_auc = roc_auc_score(y_train, all_train_probs[name])
        ax.plot(fpr, tpr, color=COLORS[idx], linewidth=2.2,
                label=f"{name} (train AUC = {train_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random classifier (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate", fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontweight="bold")
    ax.set_title("Training-Set ROC Curves: Surrogate Model Comparison", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig("figures/surrogate_roc_curves_train.png", bbox_inches="tight")
    plt.savefig("figures/surrogate_roc_curves_train.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[surrogate] Training-set ROC curves saved to figures/surrogate_roc_curves_train.png/pdf")

    return summary_table, results_df


if __name__ == "__main__":
    main()
