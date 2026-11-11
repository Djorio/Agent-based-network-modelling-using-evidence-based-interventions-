"""
config.py — Single source of truth for all MOV ABNM parameters.
All other modules import from here. Do NOT hardcode values elsewhere.

REVISION NOTE (this version)
-----------------------------
* BARRIER_ORS / BARRIER_COLS_14 / BARRIER_LABELS / BARRIER_CATEGORIES /
  CATEGORY_COLORS have been rebuilt from the actual, empirically estimated
  Agent-Based Network Model (ABNM) fitted on "Main_data_MOV" (survey-weighted
  multivariable logistic regression - strata + clustered PSU + sampling
  weight - pooled across m=5 MICE imputations via Rubin's Rules). The
  fabricated placeholder ORs/CIs from the previous version have been removed.
* Only the 14 barrier variables that appear in that ABNM are retained:
  children_u5, members_household, education_caregiver, education_father,
  residence, occupation, religion, wealth, transportation, radio,
  television, deliveryplace, distance, vaccinationplace. They expand into
  33 encoded (dummy) terms in BARRIER_ORS/BARRIER_COLS_14.
* ALL geospatial barrier content (log_aridity, log_max_temp, and every
  parameter, label, category, color and calibration prior that referenced
  them) has been removed.
* The ABNM baseline has been recalibrated to the target MOV prevalence of
  0.34 (BASELINE_MOV_2019). Nothing else about the ABNM graph (nodes, edges,
  layout, categories) has been changed - only the baseline/agent-probability
  calibration constant, via ABNM_INTERCEPT_LOGODDS = logit(0.34).
"""

import numpy as np

# ─────────────────────────────────────────────
# SIMULATION PARAMETERS
# ─────────────────────────────────────────────
REFERENCE_POPULATION = 23_490        # Children in analysis cohort
BASELINE_MOV_2019 = 0.34             # Calibration target
BASELINE_MOV_CI = (0.24, 0.43)       # 95%CI for target
COVID_MOV_ADJUSTMENT = 0.03          # +3% absolute MOV 2020-2023
COVID_YEARS = list(range(2020, 2024))  # 2020, 2021, 2022, 2023
SIM_START = 2019
INTERVENTION_START = 2025
SIM_END = 2035
YEARS = list(range(SIM_START, SIM_END + 1))

# Baseline log-odds implied by BASELINE_MOV_2019, used to calibrate/recalibrate
# the ABNM's agent-specific probabilities: logit(0.34) = ln(0.34/0.66).
# Geospatial covariates have been removed, so - unlike the previous version -
# this intercept is used directly, with no large offsetting geospatial terms.
ABNM_INTERCEPT_LOGODDS = float(np.log(BASELINE_MOV_2019 / (1 - BASELINE_MOV_2019)))  # ≈ -0.6633

# ─────────────────────────────────────────────
# BARRIER ADJUSTED ORs (multivariable logistic regression on Main_data_MOV)
# Format: {barrier_col: {"OR": float, "CI_lo": float, "CI_hi": float, "p": float}}
#
# Reference categories (i.e. the "advantaged"/lowest-risk group for each barrier,
# as used in the fitted model and the ABNM diagram):
#   children_u5 = One, members_household = 1-3,
#   education_caregiver = Secondary and beyond, education_father = secondary or higher,
#   residence = Urban, occupation = Pro/Tec/Man/Cler/Ser/Other, religion = Muslim,
#   wealth = Richest, transportation = car/truck/bus/taxi/boat_motor (motorised),
#   radio = Yes, television = Yes, deliveryplace = Public,
#   distance = Not a big problem, vaccinationplace = Public.
# ─────────────────────────────────────────────
BARRIER_ORS = {
    # Caregiver / household
    "children_u5_2to3":          {"OR": 1.1547, "CI_lo": 1.0588, "CI_hi": 1.2593, "p": 0.0011},
    "children_u5_ge4":           {"OR": 1.2325, "CI_lo": 1.0428, "CI_hi": 1.4567, "p": 0.0142},
    "members_household_4to5":    {"OR": 0.9679, "CI_lo": 0.8567, "CI_hi": 1.0936, "p": 0.6004},
    "members_household_gt6":     {"OR": 1.0665, "CI_lo": 0.9413, "CI_hi": 1.2084, "p": 0.3119},
    # Caregiver education
    "edu_caregiver_primary":     {"OR": 1.0776, "CI_lo": 0.9693, "CI_hi": 1.1980, "p": 0.1662},
    "edu_caregiver_none":        {"OR": 1.2978, "CI_lo": 1.1578, "CI_hi": 1.4548, "p": 0.0000},
    # Father's education
    "edu_father_primary":        {"OR": 0.9992, "CI_lo": 0.8358, "CI_hi": 1.1945, "p": 0.9920},
    "edu_father_none":           {"OR": 1.0500, "CI_lo": 0.8995, "CI_hi": 1.2256, "p": 0.5234},
    # Residence
    "residence_rural":           {"OR": 0.9158, "CI_lo": 0.8255, "CI_hi": 1.0159, "p": 0.0966},
    # Occupation
    "occupation_sales":          {"OR": 0.7124, "CI_lo": 0.6061, "CI_hi": 0.8375, "p": 0.0001},
    "occupation_agriculture":    {"OR": 0.8799, "CI_lo": 0.7604, "CI_hi": 1.0182, "p": 0.0858},
    "occupation_manual_domestic":{"OR": 1.0076, "CI_lo": 0.8565, "CI_hi": 1.1854, "p": 0.9263},
    "occupation_not_working":    {"OR": 1.2069, "CI_lo": 1.0360, "CI_hi": 1.4060, "p": 0.0164},
    # Religion (ref = Muslim)
    "religion_other_christian":  {"OR": 1.1974, "CI_lo": 1.0668, "CI_hi": 1.3441, "p": 0.0023},
    "religion_catholic":         {"OR": 1.9093, "CI_lo": 1.6582, "CI_hi": 2.1986, "p": 0.0000},
    "religion_protestant":       {"OR": 1.1029, "CI_lo": 0.9518, "CI_hi": 1.2780, "p": 0.1926},
    "religion_other":            {"OR": 1.4951, "CI_lo": 1.2805, "CI_hi": 1.7456, "p": 0.0000},
    # Wealth quintile (ref = Richest)
    "wealth_fourth":              {"OR": 1.0431, "CI_lo": 0.9049, "CI_hi": 1.2024, "p": 0.5605},
    "wealth_middle":              {"OR": 1.0460, "CI_lo": 0.9074, "CI_hi": 1.2057, "p": 0.5355},
    "wealth_second":              {"OR": 0.9939, "CI_lo": 0.8578, "CI_hi": 1.1516, "p": 0.9351},
    "wealth_poorest":             {"OR": 1.0108, "CI_lo": 0.8667, "CI_hi": 1.1788, "p": 0.8909},
    # Transportation (ref = motorised)
    "transport_bicycle_animal":  {"OR": 0.9212, "CI_lo": 0.7879, "CI_hi": 1.0770, "p": 0.2885},
    "transport_walking":         {"OR": 1.0599, "CI_lo": 0.9438, "CI_hi": 1.1903, "p": 0.3247},
    # Media
    "radio_no":                  {"OR": 1.5180, "CI_lo": 1.3960, "CI_hi": 1.6507, "p": 0.0000},
    "tv_no":                     {"OR": 0.9881, "CI_lo": 0.8844, "CI_hi": 1.1040, "p": 0.8280},
    # Health system: place of delivery (ref = Public)
    "delivery_private":          {"OR": 1.0528, "CI_lo": 0.9028, "CI_hi": 1.2276, "p": 0.4943},
    "delivery_ngo":              {"OR": 1.1933, "CI_lo": 0.9570, "CI_hi": 1.4880, "p": 0.1053},
    "delivery_other":            {"OR": 1.2861, "CI_lo": 1.0342, "CI_hi": 1.5994, "p": 0.0256},
    "delivery_home":             {"OR": 1.3296, "CI_lo": 1.1563, "CI_hi": 1.5287, "p": 0.0001},
    # Distance
    "distance_big_problem":      {"OR": 1.2836, "CI_lo": 1.1632, "CI_hi": 1.4165, "p": 0.0000},
    # Health system: place of vaccination (ref = Public)
    "vaccinationplace_private":  {"OR": 0.9261, "CI_lo": 0.7966, "CI_hi": 1.0767, "p": 0.2857},
    "vaccinationplace_ngo":      {"OR": 0.9318, "CI_lo": 0.7501, "CI_hi": 1.1575, "p": 0.5090},
    "vaccinationplace_other":    {"OR": 1.0000, "CI_lo": 0.7437, "CI_hi": 1.3445, "p": 0.9999},
}

# All 33 encoded barrier terms used in the ABNM, spanning the 14 barrier
# variables identified in the Main_data_MOV analysis. (Kept as
# BARRIER_COLS_14 for backward compatibility with modules that import this
# name; it lists the *encoded terms*, not the 14 raw variables themselves —
# see BARRIER_VARIABLES_14 below for the raw variable names.)
BARRIER_COLS_14 = list(BARRIER_ORS.keys())

# The 14 raw barrier variables (each expands into one or more encoded terms above)
BARRIER_VARIABLES_14 = [
    "children_u5", "members_household", "education_caregiver", "education_father",
    "residence", "occupation", "religion", "wealth", "transportation", "radio",
    "television", "deliveryplace", "distance", "vaccinationplace",
]

# Human-readable labels for network diagram
BARRIER_LABELS = {
    "children_u5_2to3":           "2-3 children\nunder 5",
    "children_u5_ge4":            "\u22654 children\nunder 5",
    "members_household_4to5":     "Household size:\n4-5 members",
    "members_household_gt6":      "Household size:\n>6 members",
    "edu_caregiver_primary":      "Caregiver educ.\n(primary)",
    "edu_caregiver_none":         "Caregiver educ.\n(none)",
    "edu_father_primary":         "Father's educ.\n(primary)",
    "edu_father_none":            "Father's educ.\n(none)",
    "residence_rural":            "Rural\nresidence",
    "occupation_sales":           "Occupation:\nSales",
    "occupation_agriculture":     "Occupation:\nAgriculture",
    "occupation_manual_domestic": "Occupation:\nManual/domestic",
    "occupation_not_working":     "Occupation:\nNot working",
    "religion_other_christian":   "Religion:\nOther Christian",
    "religion_catholic":          "Religion:\nCatholic",
    "religion_protestant":        "Religion:\nProtestant",
    "religion_other":             "Religion:\nOther",
    "wealth_fourth":              "Wealth: 4th\nquintile",
    "wealth_middle":              "Wealth: middle\nquintile",
    "wealth_second":              "Wealth: 2nd\nquintile",
    "wealth_poorest":             "Wealth: poorest\nquintile",
    "transport_bicycle_animal":   "Bicycle/animal-drawn\ntransport",
    "transport_walking":          "Walking-only\ntransport",
    "radio_no":                   "No radio\naccess",
    "tv_no":                      "No TV\naccess",
    "delivery_private":           "Private facility\ndelivery",
    "delivery_ngo":               "NGO/Religious\nfacility delivery",
    "delivery_other":             "Other place\nof delivery",
    "delivery_home":              "Home\ndelivery",
    "distance_big_problem":       "Distance: big\nproblem",
    "vaccinationplace_private":   "Private vaccination\nplace",
    "vaccinationplace_ngo":       "NGO/Religious\nvaccination place",
    "vaccinationplace_other":     "Other vaccination\nplace",
    "MOV":                        "Missed\nVaccination\n(MOV)",
}

# Node categories for coloring
BARRIER_CATEGORIES = {
    "caregiver":     ["children_u5_2to3", "children_u5_ge4",
                       "members_household_4to5", "members_household_gt6",
                       "edu_caregiver_primary", "edu_caregiver_none",
                       "edu_father_primary", "edu_father_none"],
    "socioeconomic": ["residence_rural",
                       "occupation_sales", "occupation_agriculture",
                       "occupation_manual_domestic", "occupation_not_working",
                       "wealth_fourth", "wealth_middle", "wealth_second", "wealth_poorest",
                       "religion_other_christian", "religion_catholic",
                       "religion_protestant", "religion_other"],
    "media":         ["radio_no", "tv_no"],
    "transport":     ["transport_bicycle_animal", "transport_walking"],
    "health_system": ["delivery_private", "delivery_ngo", "delivery_other", "delivery_home",
                       "distance_big_problem",
                       "vaccinationplace_private", "vaccinationplace_ngo", "vaccinationplace_other"],
}

CATEGORY_COLORS = {
    "caregiver":     "#4C72B0",  # blue
    "socioeconomic": "#DD8452",  # orange
    "media":         "#8172B2",  # purple
    "transport":     "#937860",  # brown
    "health_system": "#55A868",  # green
    "outcome":       "#1A1A1A",  # black
}

# ─────────────────────────────────────────────
# INTERVENTION RISK RATIOS
# Format: {int_id: {"RR": float, "CI_lo": float, "CI_hi": float, "name": str}}
# ─────────────────────────────────────────────
INTERVENTION_RRS = {
    1: {"RR": 0.49, "CI_lo": 0.26, "CI_hi": 0.91,
        "name": "Health professional education",
        "evidence": "Very low"},
    2: {"RR": 0.79, "CI_lo": 0.60, "CI_hi": 1.04,
        "name": "Caregiver education",
        "evidence": "Very low"},
    3: {"RR": 0.66, "CI_lo": 0.45, "CI_hi": 0.98,
        "name": "Electronic immunization registers (EIRs)",
        "evidence": "Low"},
    4: {"RR": 0.34, "CI_lo": 0.08, "CI_hi": 1.38,
        "name": "SMS/phone call reminders",
        "evidence": "Very low"},
    5: {"RR": 0.50, "CI_lo": 0.40, "CI_hi": 0.62,
        "name": "Integration of immunisation services (IMCI)",
        "evidence": "Very low"},
    6: {"RR": 0.43, "CI_lo": 0.26, "CI_hi": 0.72,
        "name": "Pharmacist integration",
        "evidence": "Low"},
    7: {"RR": 0.87, "CI_lo": 0.80, "CI_hi": 0.94,
        "name": "Financial incentives",
        "evidence": "Moderate"},
    8: {"RR": 0.72, "CI_lo": 0.68, "CI_hi": 0.76,
        "name": "Multivalent vaccines",
        "evidence": "Low"},
    9: {"RR": 0.62, "CI_lo": 0.48, "CI_hi": 0.80,
        "name": "Multicomponent interventions",
        "evidence": "Moderate"},
}

# ─────────────────────────────────────────────
# BARRIER → INTERVENTION MATCHING
# Which barrier terms (from BARRIER_ORS) each intervention targets.
# Rebuilt against the new barrier term names; the geospatial-specific note
# for intervention 9 and the visit_hf_yes barrier (not part of the ABNM) have
# been removed, and the newly-retained occupation / members_household /
# vaccinationplace terms have been assigned to the most plausible
# intervention(s) on conceptual grounds.
# ─────────────────────────────────────────────
BARRIER_INTERVENTION_MAP = {
    1: ["children_u5_2to3", "children_u5_ge4",
        "edu_caregiver_primary", "edu_caregiver_none",
        "wealth_second", "wealth_middle", "wealth_fourth", "wealth_poorest",
        "religion_other_christian", "religion_catholic", "religion_protestant", "religion_other"],
    2: ["children_u5_2to3", "children_u5_ge4",
        "members_household_4to5", "members_household_gt6",
        "edu_caregiver_primary", "edu_caregiver_none",
        "edu_father_primary", "edu_father_none",
        "occupation_sales", "occupation_agriculture", "occupation_manual_domestic", "occupation_not_working",
        "wealth_second", "wealth_middle", "wealth_fourth", "wealth_poorest"],
    3: ["residence_rural", "radio_no", "tv_no",
        "vaccinationplace_private", "vaccinationplace_ngo", "vaccinationplace_other"],
    4: ["edu_caregiver_primary", "edu_caregiver_none",
        "residence_rural", "radio_no", "tv_no",
        "vaccinationplace_private", "vaccinationplace_ngo", "vaccinationplace_other"],
    5: ["delivery_private", "delivery_ngo", "delivery_other", "delivery_home",
        "vaccinationplace_private", "vaccinationplace_ngo", "vaccinationplace_other"],
    6: ["delivery_private", "delivery_ngo", "delivery_other", "delivery_home",
        "vaccinationplace_private", "vaccinationplace_ngo", "vaccinationplace_other"],
    7: ["residence_rural", "occupation_sales", "occupation_agriculture",
        "occupation_manual_domestic", "occupation_not_working",
        "wealth_second", "wealth_middle", "wealth_fourth", "wealth_poorest",
        "distance_big_problem"],
    8: ["residence_rural", "delivery_private", "delivery_ngo", "delivery_other", "delivery_home",
        "distance_big_problem"],
    9: ["edu_caregiver_primary", "edu_caregiver_none",
        "edu_father_primary", "edu_father_none",
        "religion_other_christian", "religion_catholic", "religion_protestant", "religion_other"],
}

# ─────────────────────────────────────────────
# INTERVENTION PACKAGES
# ─────────────────────────────────────────────
PACKAGES = {
    "P1":  [1, 2, 3],
    "P2":  [1, 3, 5],
    "P3":  [1, 4, 6],
    "P4":  [1, 2, 6],
    "P5":  [1, 4, 7],
    "P6":  [1, 2, 8],
    "P7":  [1, 3, 6],
    "P8":  [2, 5, 7],
    "P9":  [2, 4, 5],
    "P10": [2, 9],
}

PACKAGE_LABELS = {k: f"{k} ({'+'.join(str(i) for i in v)})" for k, v in PACKAGES.items()}

# ─────────────────────────────────────────────
# COSTS (US$ per child)
# Format: (mean, lo_95CrI, hi_95CrI)
# Distributions: gamma(alpha, beta) fitted to these
# Int 3 (EIR) uses special flat-cost structure — see EIR_COST_STRUCTURE
# ─────────────────────────────────────────────
COSTS_PER_CHILD = {
    1: (0.10,  0.01,  0.28),
    2: (5.67,  2.13,  10.91),
    3: None,   # Special — use EIR_COST_STRUCTURE
    4: (0.42,  0.16,  0.81),
    5: (2.30,  0.86,  4.42),
    6: (12.00, 4.52,  23.08),
    7: (5.47,  2.06,  10.52),
    8: (15.00, 5.64,  28.84),
    9: (9.64,  3.63,  18.55),
}

EIR_COST_STRUCTURE = {
    # 5-year total: flat development cost + 5 years annualised per-child
    "flat_mean":    4_880_000,
    "flat_lo":      1_840_000,
    "flat_hi":      9_390_000,
    "annual_child": 138_121.2,   # annualised cost per year
    "total_5yr":    5_570_606,   # = 4,880,000 + 138,121.2 × 5
}

# WTP thresholds for CEAC (US$ per MOV-averted child)
WTP_THRESHOLDS = np.linspace(0, 3000, 200)
WTP_PRIMARY = 500   # Conservative SSA threshold

# ─────────────────────────────────────────────
# ABC-SMC CALIBRATION PARAMETERS
# The prior range for the intercept theta has been narrowed to surround
# ABNM_INTERCEPT_LOGODDS = logit(0.34) ≈ -0.663, now that the large
# geospatial offset terms (previously requiring theta as low as -14.7 to
# -20 to compensate) have been removed.
# ─────────────────────────────────────────────
ABC_PRIOR_LO = -3.0      # Uniform prior lower bound for intercept θ
ABC_PRIOR_HI = 1.0       # Uniform prior upper bound (~34% prevalence at θ≈-0.663)
ABC_N_PARTICLES = 1000
ABC_TOLERANCES = [0.15, 0.10, 0.05, 0.03, 0.02, 0.01]

# Simulation parameters
N_POSTERIOR_DRAWS = 200  # Intercept draws from posterior for projection
N_MC_PER_DRAW    = 50    # Monte Carlo reps per posterior draw
N_PSA_DRAWS      = 1000  # Probabilistic sensitivity analysis iterations

# ─────────────────────────────────────────────
# FIGURE STYLE
# ─────────────────────────────────────────────
PACKAGE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]
COLOR_NO_INT   = "#4d4d4d"
COLOR_COVID    = "#f5f5f5"   # shading for 2020-2023
COLOR_INT_ZONE = "#e8f4f8"   # shading for 2025-2035
