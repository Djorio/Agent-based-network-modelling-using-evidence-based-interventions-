"""
08_figures.py
Generates every publication figure for the MOV ABNM pipeline: the barrier network (DAG),
baseline prevalence, ABC-SMC posterior/PPC, simulation trajectories (no-intervention vs
10 packages), the cost-effectiveness plane + CEAC, the one-way sensitivity tornado plot,
and the calibration (reliability) diagram.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from config import (
    BARRIER_LABELS, CATEGORY_COLORS, PACKAGES, PACKAGE_LABELS, PACKAGE_COLORS,
    COLOR_NO_INT, COLOR_COVID, COLOR_INT_ZONE, COVID_YEARS, INTERVENTION_START,
    BASELINE_MOV_2019, WTP_PRIMARY,
)

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def fig_network(G: nx.DiGraph, path=None):
    """Barrier network DAG, laid out with a layered (multipartite) layout by category."""
    path = path or os.path.join(FIG_DIR, "fig1_barrier_network.png")
    cat_order = ["caregiver", "socioeconomic", "media", "transport", "health_system", "outcome"]
    for n in G.nodes:
        cat = G.nodes[n]["category"]
        G.nodes[n]["layer"] = cat_order.index(cat) if cat in cat_order else len(cat_order)

    pos = {}
    layer_nodes = {}
    for n, d in G.nodes(data=True):
        layer_nodes.setdefault(d["layer"], []).append(n)
    for layer, nodes in layer_nodes.items():
        nodes_sorted = sorted(nodes)
        k = len(nodes_sorted)
        for i, n in enumerate(nodes_sorted):
            pos[n] = (layer * 2.6, (k - 1) / 2 - i)

    fig, ax = plt.subplots(figsize=(16, 11), dpi=200)
    structural = [(u, v) for u, v, d in G.edges(data=True) if d["direction"] == "structural"]
    risk = [(u, v) for u, v, d in G.edges(data=True) if d["direction"] == "risk"]
    prot = [(u, v) for u, v, d in G.edges(data=True) if d["direction"] == "protective"]

    nx.draw_networkx_edges(G, pos, edgelist=structural, ax=ax, edge_color="#c0c0c0",
                            style="dashed", width=0.9, arrows=True, arrowsize=8,
                            connectionstyle="arc3,rad=0.1", alpha=0.6)
    if risk:
        w = [1 + 4 * G.edges[e]["weight"] / max(d["weight"] for _, _, d in G.edges(data=True)) for e in risk]
        nx.draw_networkx_edges(G, pos, edgelist=risk, ax=ax, edge_color="#d73027",
                                width=w, arrows=True, arrowsize=13, connectionstyle="arc3,rad=0.05", alpha=0.9)
    if prot:
        w = [1 + 4 * G.edges[e]["weight"] / max(d["weight"] for _, _, d in G.edges(data=True)) for e in prot]
        nx.draw_networkx_edges(G, pos, edgelist=prot, ax=ax, edge_color="#2166ac",
                                width=w, arrows=True, arrowsize=13, connectionstyle="arc3,rad=0.05", alpha=0.9)

    node_colors = [G.nodes[n]["color"] for n in G.nodes]
    node_sizes = [2600 if n == "MOV" else 1400 for n in G.nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes,
                            edgecolors="black", linewidths=1.0)
    labels = {n: G.nodes[n]["label"] for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6.6, font_weight="bold")

    handles = [mpatches.Patch(color=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
    handles += [mpatches.FancyArrowPatch((0, 0), (1, 0), color="#d73027", label="Risk (OR>1) \u2192 MOV"),
                mpatches.FancyArrowPatch((0, 0), (1, 0), color="#2166ac", label="Protective (OR<1) \u2192 MOV"),
                mpatches.FancyArrowPatch((0, 0), (1, 0), color="#c0c0c0", linestyle="dashed",
                                          label="Conceptual pathway")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, fontsize=8, frameon=False)
    ax.set_title("MOV Agent-Based Network Model: 14 Barriers (33 encoded terms) \u2192 MOV", fontsize=14, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] Saved {path}")


def fig_baseline(baseline_result: dict, path=None):
    path = path or os.path.join(FIG_DIR, "fig2_baseline_prevalence.png")
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    est = float(baseline_result["estimate"])
    lo, hi = float(baseline_result["CI_lo"]), float(baseline_result["CI_hi"])
    ax.bar(["Survey-weighted\nMOV prevalence"], [est], yerr=[[est - lo], [hi - est]],
           color="#4C72B0", capsize=8, width=0.5)
    ax.axhline(BASELINE_MOV_2019, color="black", linestyle="--", label=f"Calibration target ({BASELINE_MOV_2019})")
    ax.set_ylim(0, 0.6)
    ax.set_ylabel("MOV prevalence")
    ax.set_title(f"Baseline MOV Prevalence\n{est:.3f} (95% CI {lo:.3f}\u2013{hi:.3f})", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] Saved {path}")


def fig_calibration_ppc(posterior: dict, ppc_prevs: np.ndarray, path=None):
    path = path or os.path.join(FIG_DIR, "fig3_abc_smc_calibration.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=200)

    axes[0].hist(posterior["posterior_samples"], bins=30, weights=posterior["posterior_weights"],
                 color="#55A868", alpha=0.85)
    axes[0].set_xlabel(r"Intercept $\theta$")
    axes[0].set_ylabel("Posterior density (weighted)")
    axes[0].set_title("ABC-SMC Posterior: Calibrated Intercept", fontweight="bold")

    axes[1].hist(ppc_prevs, bins=30, color="#4C72B0", alpha=0.85)
    axes[1].axvline(BASELINE_MOV_2019, color="black", linestyle="--", label=f"Target = {BASELINE_MOV_2019}")
    axes[1].set_xlabel("Simulated weighted MOV prevalence")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Posterior Predictive Check", fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] Saved {path}")


def fig_simulation_trajectories(sim_results: dict, path=None):
    path = path or os.path.join(FIG_DIR, "fig4_simulation_trajectories.png")
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=200)

    ax.axvspan(min(COVID_YEARS), max(COVID_YEARS), color=COLOR_COVID, zorder=0, label="COVID-19 period")
    ax.axvspan(INTERVENTION_START, max(sim_results["no_intervention"]["year"]), color=COLOR_INT_ZONE,
               zorder=0, label="Intervention period")

    ni = sim_results["no_intervention"]
    ax.plot(ni["year"], ni["mean_prev"], color=COLOR_NO_INT, linewidth=2.5, label="No intervention")
    ax.fill_between(ni["year"], ni["CI_lo"], ni["CI_hi"], color=COLOR_NO_INT, alpha=0.15)

    for i, (pkg, color) in enumerate(zip(PACKAGES, PACKAGE_COLORS)):
        df = sim_results[pkg]
        ax.plot(df["year"], df["mean_prev"], color=color, linewidth=1.6, label=PACKAGE_LABELS[pkg], alpha=0.9)

    ax.axhline(BASELINE_MOV_2019, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Simulated MOV prevalence")
    ax.set_title("Projected MOV Prevalence 2019\u20132035: No Intervention vs. 10 Packages", fontweight="bold")
    ax.legend(loc="upper right", fontsize=6.6, ncol=2, frameon=True ,framealpha=0.9)
    ax.set_ylim(0, max(0.6, ni["CI_hi"].max() * 1.05))
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] Saved {path}")


def fig_ce_plane_ceac(icer_table: pd.DataFrame, psa_results: pd.DataFrame, ceac: pd.DataFrame, path=None):
    path = path or os.path.join(FIG_DIR, "fig5_cost_effectiveness.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

    packages = list(PACKAGES.keys())
    colors = dict(zip(packages, PACKAGE_COLORS))
    for pkg in packages:
        sub = psa_results[psa_results["package"] == pkg]
        axes[0].scatter(sub["effect"], sub["cost"], s=4, alpha=0.15, color=colors[pkg])
        pt = icer_table[icer_table["package"] == pkg].iloc[0]
        axes[0].scatter(pt["children_mov_averted"], pt["total_cost_usd"], s=90, color=colors[pkg],
                         edgecolor="black", zorder=5, label=pkg)
    axes[0].set_xlabel("Cumulative children with MOV averted (2025\u20132035)")
    axes[0].set_ylabel("Total package cost (US$)")
    axes[0].set_title("Cost-Effectiveness Plane", fontweight="bold")
    axes[0].legend(fontsize=7, ncol=2, frameon=False)

    for pkg in packages:
        sub = ceac[ceac["package"] == pkg]
        axes[1].plot(sub["wtp"], sub["prob_optimal"], label=pkg, color=colors[pkg], linewidth=1.8)
    axes[1].axvline(WTP_PRIMARY, color="black", linestyle="--", linewidth=1, label=f"WTP=${WTP_PRIMARY}")
    axes[1].set_xlabel("Willingness-to-pay (US$ per MOV-averted child)")
    axes[1].set_ylabel("Probability cost-effective")
    axes[1].set_title("Cost-Effectiveness Acceptability Curve (CEAC)", fontweight="bold")
    axes[1].legend(fontsize=6.5, ncol=2, frameon=False)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] Saved {path}")


def fig_sensitivity_tornado(sa_results: pd.DataFrame, path=None, top_n: int = 20):
    path = path or os.path.join(FIG_DIR, "fig6_sensitivity_tornado.png")
    piv = sa_results.pivot_table(index="parameter", columns="bound", values="output_prevalence")
    piv["swing"] = (piv["high"] - piv["low"]).abs()
    piv = piv.sort_values("swing", ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(9, max(6, 0.32 * len(piv))), dpi=200)
    y = np.arange(len(piv))
    left = np.minimum(piv["low"], piv["high"])
    width = np.abs(piv["high"] - piv["low"])
    ax.barh(y, width, left=left, color="#DD8452", edgecolor="black", height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(piv.index, fontsize=7.5)
    ax.set_xlabel("Simulated weighted MOV prevalence (parameter at CI bound)")
    ax.set_title("One-Way Sensitivity Analysis (Tornado Diagram)\nTop Drivers of Simulated MOV Prevalence",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] Saved {path}")


def fig_validation_calibration_curve(validation_results: dict, cal_curve: dict = None, path=None):
    path = path or os.path.join(FIG_DIR, "fig7_calibration_reliability.png")
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    if cal_curve is not None:
        ax.plot(cal_curve["mean_predicted"], cal_curve["fraction_positive"], "s-",
                color="#B3202A", label="Calibrated ABNM")
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted P(MOV)")
    ax.set_ylabel("Observed fraction MOV=1")
    title = "Calibration (Reliability) Diagram"
    if validation_results:
        auc = validation_results.get("auc", np.nan)
        bss = validation_results.get("brier_skill_score", validation_results.get("bss", np.nan))
        title += f"\nAUC={auc:.3f}, Brier Skill Score={bss:.3f}"
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] Saved {path}")


def generate_all_figures(G=None, baseline_result=None, df_agents=None, ppc_prevs=None,
                          sim_results=None, posterior=None, icer_table=None, sa_results=None,
                          validation_results=None, psa_results=None, ceac=None, cal_curve=None,
                          **kwargs):
    os.makedirs(FIG_DIR, exist_ok=True)
    if G is not None:
        fig_network(G)
    if baseline_result is not None:
        fig_baseline(baseline_result)
    if posterior is not None and ppc_prevs is not None:
        fig_calibration_ppc(posterior, ppc_prevs)
    if sim_results is not None:
        fig_simulation_trajectories(sim_results)
    if icer_table is not None and psa_results is not None and ceac is not None:
        fig_ce_plane_ceac(icer_table, psa_results, ceac)
    if sa_results is not None:
        fig_sensitivity_tornado(sa_results)
    if validation_results is not None or cal_curve is not None:
        fig_validation_calibration_curve(validation_results or {}, cal_curve)
    print("[figures] All available figures generated.")


if __name__ == "__main__":
    import importlib.util, os as _os

    def load(fname, name):
        spec = importlib.util.spec_from_file_location(name, _os.path.join(_os.path.dirname(__file__), fname))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    nm = load("02_network_model.py", "network_model")
    G = nm.build_barrier_network()
    fig_network(G)
