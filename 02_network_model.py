"""
02_network_model.py
NetworkX directed graph of the 14 barrier variables (33 encoded terms) + MOV outcome node,
rebuilt to match the actual Main_data_MOV ABNM (no geospatial nodes/edges).
Agent probability engine: logistic model with a free, ABC-SMC-calibrated intercept.
"""

import numpy as np
import networkx as nx
from scipy.special import expit

from config import (
    BARRIER_ORS, BARRIER_COLS_14, BARRIER_LABELS,
    BARRIER_CATEGORIES, CATEGORY_COLORS, BARRIER_INTERVENTION_MAP,
    INTERVENTION_RRS,
)

# conceptual (structural, non-estimated) mediating pathways among barrier VARIABLES,
# expressed at the encoded-term level using each variable's highest-risk term as the
# representative node for edges that originate FROM a variable with >1 term.
_CONCEPTUAL_EDGES_BY_VARIABLE = [
    ("edu_caregiver_none", "occupation_not_working"),
    ("edu_caregiver_none", "edu_father_none"),
    ("edu_caregiver_none", "radio_no"),
    ("edu_caregiver_none", "tv_no"),
    ("wealth_poorest", "residence_rural"),
    ("wealth_poorest", "children_u5_ge4"),
    ("wealth_poorest", "members_household_gt6"),
    ("wealth_poorest", "transport_walking"),
    ("wealth_poorest", "distance_big_problem"),
    ("wealth_poorest", "delivery_home"),
    ("wealth_poorest", "vaccinationplace_other"),
    ("occupation_not_working", "transport_walking"),
    ("occupation_not_working", "distance_big_problem"),
    ("residence_rural", "transport_walking"),
    ("residence_rural", "distance_big_problem"),
    ("residence_rural", "delivery_home"),
    ("residence_rural", "vaccinationplace_other"),
    ("edu_father_none", "radio_no"),
    ("edu_father_none", "tv_no"),
    ("children_u5_ge4", "delivery_home"),
    ("members_household_gt6", "distance_big_problem"),
    ("religion_catholic", "delivery_home"),
    ("religion_catholic", "vaccinationplace_other"),
]


def build_barrier_network() -> nx.DiGraph:
    """
    Build the 33-term-barrier + MOV directed acyclic graph.
    Node attributes: label, category, OR, color
    Edge attributes: weight (|log OR|), log_or, direction (protective/risk), color
    """
    G = nx.DiGraph()

    all_nodes = list(BARRIER_COLS_14) + ["MOV"]

    for node in all_nodes:
        cat = "outcome" if node == "MOV" else None
        for c, members in BARRIER_CATEGORIES.items():
            if node in members:
                cat = c
                break
        if cat is None:
            cat = "other"

        or_val = BARRIER_ORS[node]["OR"] if node in BARRIER_ORS else None

        G.add_node(
            node,
            label=BARRIER_LABELS.get(node, node),
            category=cat,
            color=CATEGORY_COLORS.get(cat, "#aaaaaa"),
            OR=or_val,
        )

    # ── Barrier → MOV edges (every encoded term contributes to the outcome) ──────
    for node in BARRIER_COLS_14:
        log_or = np.log(BARRIER_ORS[node]["OR"])
        direction = "protective" if log_or < 0 else "risk"
        G.add_edge(
            node, "MOV",
            weight=abs(log_or), log_or=log_or, direction=direction,
            color="#2166ac" if direction == "protective" else "#d73027",
        )

    # ── Inter-barrier conceptual edges (structural, not separately estimated) ────
    for src, dst in _CONCEPTUAL_EDGES_BY_VARIABLE:
        if G.has_node(src) and G.has_node(dst) and src != dst:
            G.add_edge(src, dst, weight=0.5, color="#888888",
                       direction="structural", log_or=0)

    assert nx.is_directed_acyclic_graph(G), "Barrier network must remain a DAG"
    print(f"[network] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def compute_agent_mov_probabilities(fixed_log_odds: np.ndarray,
                                     intercept: float) -> np.ndarray:
    """p_i = sigmoid(intercept + fixed_log_odds_i)"""
    return expit(intercept + fixed_log_odds)


def simulate_mov_outcomes(probabilities: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """Bernoulli draw: mov_i ~ Bernoulli(p_i)"""
    return rng.binomial(1, probabilities).astype(float)


def apply_intervention_rrs(df_agents,
                            fixed_log_odds: np.ndarray,
                            package_interventions: list,
                            rr_samples: dict,
                            intercept: float) -> np.ndarray:
    """
    Apply intervention RRs multiplicatively to agent MOV probabilities.

    For each intervention j in the package, agents whose barrier profile includes at
    least one of intervention j's target barriers have their probability reduced:
    p = p * RR_j. Clipped to [0.001, 0.999].
    """
    p = compute_agent_mov_probabilities(fixed_log_odds, intercept).copy()

    for int_id in package_interventions:
        rr = rr_samples.get(int_id, INTERVENTION_RRS[int_id]["RR"])
        target_barriers = BARRIER_INTERVENTION_MAP.get(int_id, [])

        if len(target_barriers) == 0:
            p = p * rr
            continue

        mask = np.zeros(len(df_agents), dtype=bool)
        for barrier in target_barriers:
            if barrier in df_agents.columns:
                mask |= (df_agents[barrier].values == 1)
        p[mask] *= rr

    return np.clip(p, 0.001, 0.999)


def sample_rrs_from_distributions(n_samples: int = 1, seed: int = None) -> dict:
    """
    Sample RR values from log-normal distributions for each intervention.
    mu_log = log(RR_point); sigma_log = (log(CI_hi) - log(CI_lo)) / (2*1.96).
    """
    rng = np.random.default_rng(seed)
    result = {}
    for int_id, params in INTERVENTION_RRS.items():
        rr, lo, hi = params["RR"], max(params["CI_lo"], 1e-6), params["CI_hi"]
        mu_log = np.log(rr)
        sigma_log = (np.log(hi) - np.log(lo)) / (2 * 1.96)
        samples = rng.lognormal(mu_log, max(sigma_log, 0.001), size=n_samples)
        samples = np.clip(samples, 0.001, 1.0)
        result[int_id] = samples[0] if n_samples == 1 else samples
    return result


if __name__ == "__main__":
    G = build_barrier_network()
    print(f"Nodes: {list(G.nodes())[:5]}...")
    print(f"Edges to MOV: {G.in_degree('MOV')}")

    n = 1000
    rng = np.random.default_rng(42)
    test_flo = rng.normal(0.59, 0.43, n)
    p = compute_agent_mov_probabilities(test_flo, intercept=-1.25)
    print(f"Test: mean p = {p.mean():.3f} (target ~0.34 baseline)")
