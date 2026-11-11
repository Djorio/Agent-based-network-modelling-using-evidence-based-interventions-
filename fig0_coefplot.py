"""
fig0_coefplot.py
Publication coefplot of adjusted ORs (barriers with OR > 1 only) from the
pooled survey-weighted multivariable logistic regression (00b).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import CATEGORY_COLORS, BARRIER_CATEGORIES

FIG_PATH = "figures/fig0_adjusted_OR_coefplot.png"


def category_of(term):
    for cat, members in BARRIER_CATEGORIES.items():
        if term in members:
            return cat
    return "other"


def make_coefplot(or_table_path="results/adjusted_ORs_pooled.csv", path=FIG_PATH):
    df = pd.read_csv(or_table_path)
    df = df[df["OR"] > 1.0].copy()                 # only barriers (OR > 1) per spec
    df["category"] = df["term"].map(category_of)
    df = df.sort_values("OR", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(df) + 1.5), dpi=200)
    y = np.arange(len(df))
    colors = [CATEGORY_COLORS.get(c, "#888888") for c in df["category"]]

    ax.errorbar(df["OR"], y,
                xerr=[df["OR"] - df["CI_lo"], df["CI_hi"] - df["OR"]],
                fmt="none", ecolor="black", elinewidth=1.1, capsize=3, zorder=2)
    ax.scatter(df["OR"], y, c=colors, s=70, edgecolor="black", linewidth=0.8, zorder=3)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=8.5)
    ax.set_xlabel("Adjusted Odds Ratio (95% CI)", fontsize=10, fontweight="bold")
    ax.set_title("Adjusted Odd Ratios for MOV Barriers\n(Multivariable Logistic Regression)",
                 fontsize=12, fontweight="bold")

    for i, row in df.iterrows():
        stars = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 \
            else "*" if row["p_value"] < 0.05 else ""
        ax.text(row["CI_hi"] + 0.02, i, f"{row['OR']:.2f}{stars}", va="center", fontsize=7.5)

    import matplotlib.patches as mpatches
    present_cats = df["category"].unique()
    handles = [mpatches.Patch(color=CATEGORY_COLORS[c], label=c) for c in present_cats
               if c in CATEGORY_COLORS]
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True, title="Barrier category")

    ax.set_xlim(0.95, df["CI_hi"].max() * 1.18)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[coefplot] Saved {path} ({len(df)} barrier terms with OR>1 plotted)")
    return df


if __name__ == "__main__":
    make_coefplot()
