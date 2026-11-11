"""
00_data_preparation.py
Loads Main_data_MOV.csv, cleans & recodes the 14 barrier variables, imputes missing
barrier values (MICE via scikit-learn's IterativeImputer — single completed dataset,
deterministic mode, for pipeline speed), builds the 33 binary encoded barrier columns
that match config.BARRIER_ORS exactly, and computes each agent's "fixed log-odds"
(the barrier contribution to the logistic MOV model, EXCLUDING the free intercept
theta that is calibrated by 03_abc_smc_calibration.py):

    fixed_log_odds_i = sum_j  x_ij * log(OR_j)      (x_ij in {0,1}, j over 33 terms)

so that p_i(theta) = sigmoid(theta + fixed_log_odds_i) reproduces exactly the adjusted
odds ratios estimated from Main_data_MOV.

Results are cached to disk (data/agents_cache.pkl) so repeated calls across the many
pipeline scripts are fast.
"""

import os
import numpy as np
import pandas as pd

from config import BARRIER_ORS

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "Main_data_MOV.csv")
CACHE_PATH = os.path.join(os.path.dirname(__file__), "data", "agents_cache.pkl")

OUTCOME = "mov"
DESIGN = ["strata", "psu", "psweight"]
RAW_BARRIERS = [
    "children_u5", "members_household", "education_caregiver", "education_father",
    "residence", "occupation", "religion", "wealth", "transportation", "radio",
    "television", "deliveryplace", "distance", "vaccinationplace",
]

CAT_ORDER = {
    "children_u5":          ["One", "2-3", ">4"],
    "members_household":    ["1-3", "4-5", ">6"],
    "education_caregiver":  ["Secondary and beyond", "Primary", "No education"],
    "education_father":     ["secondary or higher", "primary", "no education"],
    "residence":            ["Urban", "Rural"],
    "occupation":           ["Pro/Tec/Man/Cler/Ser/Other", "Sales", "Agriculture", "Manual/HH/Domestic", "Not working"],
    "religion":             ["Muslim", "Other christian", "Catholic", "Protestant", "Other"],
    "wealth":               ["Richest", "Fourth", "Middle", "Second", "Poorest"],
    "transportation":       ["car/truck/bus/taxi/boat_motor", "bicycle/animal-drawn/boat_no_motor", "walking"],
    "radio":                ["Yes", "No"],
    "television":           ["Yes", "No"],
    "deliveryplace":        ["Public", "Private", "NGO/Religious", "Other", "Home"],
    "distance":             ["Not a big problem", "Big problem"],
    "vaccinationplace":     ["Public", "Private", "NGO/Religious", "Other"],
}

# Encoded-term -> (raw barrier column, category value that sets the indicator to 1)
TERM_DEFINITIONS = {
    "children_u5_2to3":           ("children_u5", "2-3"),
    "children_u5_ge4":            ("children_u5", ">4"),
    "members_household_4to5":     ("members_household", "4-5"),
    "members_household_gt6":      ("members_household", ">6"),
    "edu_caregiver_primary":      ("education_caregiver", "Primary"),
    "edu_caregiver_none":         ("education_caregiver", "No education"),
    "edu_father_primary":         ("education_father", "primary"),
    "edu_father_none":            ("education_father", "no education"),
    "residence_rural":            ("residence", "Rural"),
    "occupation_sales":           ("occupation", "Sales"),
    "occupation_agriculture":     ("occupation", "Agriculture"),
    "occupation_manual_domestic": ("occupation", "Manual/HH/Domestic"),
    "occupation_not_working":     ("occupation", "Not working"),
    "religion_other_christian":   ("religion", "Other christian"),
    "religion_catholic":          ("religion", "Catholic"),
    "religion_protestant":        ("religion", "Protestant"),
    "religion_other":             ("religion", "Other"),
    "wealth_fourth":              ("wealth", "Fourth"),
    "wealth_middle":              ("wealth", "Middle"),
    "wealth_second":              ("wealth", "Second"),
    "wealth_poorest":             ("wealth", "Poorest"),
    "transport_bicycle_animal":   ("transportation", "bicycle/animal-drawn/boat_no_motor"),
    "transport_walking":          ("transportation", "walking"),
    "radio_no":                   ("radio", "No"),
    "tv_no":                      ("television", "No"),
    "delivery_private":           ("deliveryplace", "Private"),
    "delivery_ngo":               ("deliveryplace", "NGO/Religious"),
    "delivery_other":             ("deliveryplace", "Other"),
    "delivery_home":              ("deliveryplace", "Home"),
    "distance_big_problem":       ("distance", "Big problem"),
    "vaccinationplace_private":   ("vaccinationplace", "Private"),
    "vaccinationplace_ngo":       ("vaccinationplace", "NGO/Religious"),
    "vaccinationplace_other":     ("vaccinationplace", "Other"),
}
assert set(TERM_DEFINITIONS) == set(BARRIER_ORS), "TERM_DEFINITIONS must match config.BARRIER_ORS exactly"


def _clean_str(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    if s.lower() in {".a", "unknown", "n/a", "na", ""}:
        return np.nan
    return s


def _load_and_clean():
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[["id", OUTCOME] + DESIGN + RAW_BARRIERS].copy()
    for c in RAW_BARRIERS:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype(object).map(_clean_str)
    df["occupation"] = df["occupation"].replace({" Manual/HH/Domestic": "Manual/HH/Domestic"})
    df["education_caregiver"] = df["education_caregiver"].replace({" Primary": "Primary"})
    df["religion"] = df["religion"].replace({" Other christian": "Other christian"})

    analytic = df.dropna(subset=[OUTCOME] + DESIGN).copy()
    analytic[OUTCOME] = analytic[OUTCOME].astype(int)
    analytic["cluster_id"] = (analytic["strata"].astype(int).astype(str) + "_" +
                               analytic["psu"].astype(int).astype(str))
    return analytic


def _impute_barriers(analytic: pd.DataFrame) -> pd.DataFrame:
    """Single-imputation MICE (IterativeImputer, deterministic) for pipeline speed."""
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    code_maps = {c: {cat: i for i, cat in enumerate(cats)} for c, cats in CAT_ORDER.items()}
    inv_code_maps = {c: {i: cat for cat, i in m.items()} for c, m in code_maps.items()}
    max_code = {c: len(cats) - 1 for c, cats in CAT_ORDER.items()}

    X_num = pd.DataFrame(index=analytic.index)
    for c in RAW_BARRIERS:
        X_num[c] = analytic[c].map(code_maps[c]).astype(float)
    X_num["_mov_aux"] = analytic[OUTCOME].astype(float)
    X_num["_logweight_aux"] = np.log(analytic["psweight"].astype(float).clip(lower=1e-6))

    imputer = IterativeImputer(estimator=BayesianRidge(), sample_posterior=False,
                                max_iter=15, random_state=42, min_value=0,
                                max_value=max(max_code.values()), verbose=0)
    arr = imputer.fit_transform(X_num.values)
    X_imp = pd.DataFrame(arr, columns=X_num.columns, index=X_num.index)

    out = analytic.copy()
    for c in RAW_BARRIERS:
        code = X_imp[c].round().clip(0, max_code[c]).astype(int)
        out[c] = code.map(inv_code_maps[c])
    return out


def _encode_binary_barriers(df_imputed: pd.DataFrame) -> pd.DataFrame:
    out = df_imputed.copy()
    for term, (col, cat) in TERM_DEFINITIONS.items():
        out[term] = (out[col] == cat).astype(int)
    return out


def _compute_fixed_log_odds(df_encoded: pd.DataFrame) -> np.ndarray:
    log_or = {term: np.log(params["OR"]) for term, params in BARRIER_ORS.items()}
    terms = list(BARRIER_ORS.keys())
    X = df_encoded[terms].values.astype(float)
    coefs = np.array([log_or[t] for t in terms])
    return X @ coefs


def prepare_agents(force_rebuild: bool = False):
    """
    Returns
    -------
    df_agents : pd.DataFrame
        One row per agent (child), with the 33 binary barrier columns, mov, psweight,
        strata, psu, cluster_id.
    fixed_log_odds : np.ndarray
        Barrier contribution to the log-odds of MOV (intercept NOT included), shape (n,).
    """
    if (not force_rebuild) and os.path.exists(CACHE_PATH):
        cached = pd.read_pickle(CACHE_PATH)
        return cached["df_agents"], cached["fixed_log_odds"]

    analytic = _load_and_clean()
    imputed = _impute_barriers(analytic)
    encoded = _encode_binary_barriers(imputed)
    fixed_log_odds = _compute_fixed_log_odds(encoded)

    keep_cols = ["id", OUTCOME] + DESIGN + ["cluster_id"] + list(BARRIER_ORS.keys())
    df_agents = encoded[keep_cols].reset_index(drop=True)
    fixed_log_odds = np.asarray(fixed_log_odds).reshape(-1)

    pd.to_pickle({"df_agents": df_agents, "fixed_log_odds": fixed_log_odds}, CACHE_PATH)
    print(f"[data_prep] Prepared {len(df_agents)} agents, {len(BARRIER_ORS)} barrier terms.")
    print(f"[data_prep] Fixed log-odds: mean={fixed_log_odds.mean():.3f}, "
          f"sd={fixed_log_odds.std():.3f}, range=[{fixed_log_odds.min():.3f}, {fixed_log_odds.max():.3f}]")
    return df_agents, fixed_log_odds


if __name__ == "__main__":
    df_agents, fixed_log_odds = prepare_agents(force_rebuild=True)
    print(df_agents.head())
    print(f"\nMOV prevalence (unweighted): {df_agents['mov'].mean():.3f}")
    print(f"MOV prevalence (weighted):   {np.average(df_agents['mov'], weights=df_agents['psweight']):.3f}")
