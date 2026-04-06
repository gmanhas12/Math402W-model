"""
Agent-Based Model: Homeless Population & Emergency Department Utilization
Sechelt (shíshálh), BC
=========================================================================

Real-world context
------------------
  Hospital     : Sechelt | shíshálh Hospital — 63 acute beds
  Shelter      : RainCity Shelter — 25 permanent beds + 10 seasonal (Oct–Apr)
  Gibsons      : Gibsons Warming Centre — 20 spaces (Oct–Apr)
  Sim start    : April 2026
  Population   : ~129 homeless individuals (Sechelt + Gibsons combined count)

Scenarios
---------
  1. Baseline          — RainCity (25 beds yr-round + 10 seasonal Oct–Apr)
                         + Gibsons Warming Centre (20 spaces Oct–Apr)
  2. Double RainCity   — Two RainCity shelters (each 25 beds + 10 seasonal)
                         + Gibsons Warming Centre
  3. Sechelt Warming   — RainCity + new Sechelt Warming Centre (20 spaces Oct–Apr)
                         + Gibsons Warming Centre

Hospital overflow
-----------------
  Hard capacity = 63 beds.
  When occupancy >= 63, new admissions are still recorded but flagged as overflow
  (i.e. the hospital is over capacity — tracked separately for reporting).

Plug-in points flagged with  # ← SWAP  throughout.
All tunable parameters are in the PARAMETERS block only.
"""

from __future__ import annotations

import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import Counter
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


# =============================================================================
# PARAMETERS  ←  edit here; nowhere else
# =============================================================================

# ── Simulation ────────────────────────────────────────────────────────────────
SIM_MONTHS          = 120         # months to simulate (10-year horizon)
SIM_START_MONTH     = 4           # April (month 1 of sim = April 2026)
RNG_SEED            = 42

# ── Hospital ──────────────────────────────────────────────────────────────────
HOSPITAL_CAPACITY   = 63          # Sechelt | shíshálh Hospital acute beds

# ── Hospital base operating cost ──────────────────────────────────────────────
# Total annual operating cost of Sechelt | shíshálh Hospital (~$151M/yr).
# Fixed cost present in every scenario — not attributable to any single patient
# group. Included so total cost figures reflect the real financial burden and
# shelter savings are shown in proper context.
HOSPITAL_ANNUAL_OPERATING_COST  = 151_000_000          # CAD/yr
HOSPITAL_BASE_MONTHLY           = HOSPITAL_ANNUAL_OPERATING_COST // 12   # ≈ $12,583,333/month

# ── Arrival process ───────────────────────────────────────────────────────────
ARRIVAL_LAMBDA      = 11          # Poisson mean: homeless individuals arriving at ED/month
                                  # ← SWAP with real ED visit counts once data arrives
ARRIVAL_SICKNESS_WEIGHT = 5.0    # scales avg sickness → extra Poisson rate for ED arrivals

# ── Admission process ─────────────────────────────────────────────────────────
ADMISSION_LAMBDA    = 10          # Poisson pool size n for Binomial draw
ADMISSION_PROB      = 0.10        # Binomial p  ← SWAP with Z59/(NFA+Z59) from real data

# ── Sickness index (-1 … 1) & admission coupling ─────────────────────────────
SICKNESS_LAMBDA_WEIGHT          = 2.0
SICKNESS_ADMISSION_DRAW_WEIGHT  = 0.5
SICKNESS_HOUSING_SCORE = {
    "HOMELESS":       0.40,
    "POST_DISCHARGE": 0.22,
    "ADMITTED":       0.0,
    "HOUSED":        -0.45,
    "DECEASED":       0.0,
}
SICKNESS_AGE_SCORE = {
    "18-29": -0.18,
    "30-44":  0.0,
    "45-59":  0.18,
    "60+":    0.32,
}
SICKNESS_GENDER_SCORE = {
    "Male":   0.06,
    "Female": -0.06,
    "Other":  0.0,
}

# ── Length of stay ────────────────────────────────────────────────────────────
LOS_MEAN_DAYS       = 15.4        # ← SWAP with mean(discharge_date - admission_date)
LOS_SD_DAYS         = 2.0         # ← SWAP with std(discharge_date - admission_date)

# ── Starting population ───────────────────────────────────────────────────────
INITIAL_POPULATION  = 129         # Sechelt + Gibsons combined PiT count

# ── Open-population dynamics ──────────────────────────────────────────────────
NEW_HOMELESS_LAMBDA         = 8
MONTHLY_MORTALITY_RATE      = 0.005
MONTHLY_SPONTANEOUS_EXIT    = 0.015

# ── Readmissions ──────────────────────────────────────────────────────────────
READMISSION_PROB            = 0.22
READMISSION_RISK_MONTHS     = 2

# ── Seasonal parameters ───────────────────────────────────────────────────────
WINTER_MONTHS               = {11, 12, 1, 2, 3}
SEASONAL_ADMISSION_BOOST    = 0.05

# ── Demographics ──────────────────────────────────────────────────────────────
GENDER_PROBS = {"Male": 0.70, "Female": 0.26, "Other": 0.04}
AGE_GROUPS   = ["18-29", "30-44", "45-59", "60+"]
AGE_PROBS    = [0.20,    0.35,    0.30,    0.15]

# ── Cost parameters ───────────────────────────────────────────────────────────
# Sechelt | shíshálh Hospital — 63 beds total: 5 ICU + 58 non-ICU
HOSPITAL_ICU_BEDS               = 5
HOSPITAL_NON_ICU_BEDS           = 58       # 63 - 5
COST_ICU_BED_DAY_LOW            = 7_000
COST_ICU_BED_DAY_HIGH           = 10_000
COST_NON_ICU_BED_DAY            = 1_100

_ICU_MID                        = (COST_ICU_BED_DAY_LOW + COST_ICU_BED_DAY_HIGH) / 2
COST_HOSPITAL_BED_DAY           = round(
    (HOSPITAL_ICU_BEDS * _ICU_MID + HOSPITAL_NON_ICU_BEDS * COST_NON_ICU_BED_DAY)
    / (HOSPITAL_ICU_BEDS + HOSPITAL_NON_ICU_BEDS)
)

# RainCity Shelter — 25 permanent beds + 10 seasonal (Oct–Apr)
RAINCITY_CAPACITY_PERMANENT     = 25
RAINCITY_CAPACITY_SEASONAL      = 10
RAINCITY_SEASONAL_MONTHS        = {10, 11, 12, 1, 2, 3, 4}
RAINCITY_FIXED_MONTHLY          = 133_333  # CAD/month (placeholder, same as Driftwood)

# Second RainCity shelter (same structure as first)
RAINCITY2_CAPACITY_PERMANENT    = 25
RAINCITY2_CAPACITY_SEASONAL     = 10
RAINCITY2_FIXED_MONTHLY         = 133_333

# Gibsons Warming Centre (Oct–Apr; 20 spaces)
GIBSONS_WC_CAPACITY             = 20
GIBSONS_WC_MONTHS               = {10, 11, 12, 1, 2, 3, 4}
GIBSONS_WC_FIXED_MONTHLY        = 40_000

# Sechelt Warming Centre (Oct–Apr; 20 spaces — hypothetical scenario 3)
SECHELT_WC_CAPACITY             = 20
SECHELT_WC_MONTHS               = {10, 11, 12, 1, 2, 3, 4}
SECHELT_WC_FIXED_MONTHLY        = 40_000

# ── Shelter diversion effects ─────────────────────────────────────────────────
SHELTER_ADMISSION_REDUCTION     = 0.05
SHELTER_READMISSION_REDUCTION   = 0.08

SHELTER_HOUSING_LAMBDA          = 1.6
SHELTER_DROPOUT_RATE            = 0.05
SHELTER_SICKNESS_REDUCTION      = 0.40

EXTRA_SHELTER_ADMISSION_REDUCTION   = 0.02
EXTRA_SHELTER_READMISSION_REDUCTION = 0.03

WARMING_ADMISSION_REDUCTION     = 0.03
WARMING_READMISSION_REDUCTION   = 0.04


# =============================================================================
# STATE MACHINE
# =============================================================================

class State(Enum):
    HOMELESS       = auto()
    ADMITTED       = auto()
    POST_DISCHARGE = auto()
    HOUSED         = auto()
    DECEASED       = auto()


@dataclass
class Agent:
    agent_id:    int
    entry_month: int
    gender:      str = "Male"
    age_group:   str = "30-44"

    state:                   State         = State.HOMELESS
    in_shelter:              bool          = False
    in_warming_centre:       bool          = False
    admission_count:         int           = 0
    current_admission_month: Optional[int] = None
    los_days:                float         = 0.0
    remaining_days:          float         = 0.0
    discharge_month:         Optional[int] = None
    months_post_discharge:   int           = 0
    total_hospital_bed_days: float         = 0.0

    def admit(self, month: int, los_days: float):
        self.state                   = State.ADMITTED
        self.admission_count        += 1
        self.current_admission_month = month
        self.los_days                = los_days
        self.remaining_days          = los_days
        self.months_post_discharge   = 0

    def discharge(self, month: int):
        self.state             = State.POST_DISCHARGE
        self.discharge_month   = month
        self.months_post_discharge = 0

    def house(self):
        self.state = State.HOUSED
        self.in_shelter = False
        self.in_warming_centre = False

    def die(self):
        self.state = State.DECEASED
        self.in_shelter = False
        self.in_warming_centre = False

    def update(self, current_month: int, days_in_month: int = 30) -> float:
        bed_days = 0.0
        if self.state is State.ADMITTED:
            days_used = min(self.remaining_days, days_in_month)
            bed_days  = days_used
            self.total_hospital_bed_days += days_used
            self.remaining_days          -= days_in_month
            if self.remaining_days <= 0:
                self.discharge(current_month)
        elif self.state is State.POST_DISCHARGE:
            self.months_post_discharge += 1
            if self.months_post_discharge > READMISSION_RISK_MONTHS:
                self.state = State.HOMELESS
        return bed_days

    @property
    def is_active(self)             -> bool: return self.state not in (State.HOUSED, State.DECEASED)
    @property
    def in_bed(self)                -> bool: return self.state is State.ADMITTED
    @property
    def eligible_new_admission(self)-> bool: return self.state is State.HOMELESS
    @property
    def eligible_readmission(self)  -> bool: return self.state is State.POST_DISCHARGE

    @property
    def sickness_index(self) -> float:
        h = SICKNESS_HOUSING_SCORE[self.state.name]
        a = SICKNESS_AGE_SCORE.get(self.age_group, 0.0)
        g = SICKNESS_GENDER_SCORE.get(self.gender, 0.0)
        base = h + a + g
        if self.in_shelter or self.in_warming_centre:
            base -= SHELTER_SICKNESS_REDUCTION
        return float(max(-1.0, min(1.0, base)))


def _monthly_avg_sickness_index(agents: list[Agent]) -> float:
    active = [ag for ag in agents if ag.is_active]
    if not active:
        return 0.0
    return float(np.mean([ag.sickness_index for ag in active]))


def _weighted_admit_sample(
    eligible: list[Agent],
    k: int,
    draw_weight: float,
    rng: np.random.Generator,
) -> list[Agent]:
    if k <= 0 or not eligible:
        return []
    pool = list(range(len(eligible)))
    out: list[Agent] = []
    for _ in range(min(k, len(eligible))):
        weights = [
            max(1e-12, 1.0 + draw_weight * eligible[i].sickness_index)
            for i in pool
        ]
        cum: list[float] = []
        s = 0.0
        for w in weights:
            s += w
            cum.append(s)
        u = rng.random() * s
        j = bisect.bisect_right(cum, u)
        if j >= len(pool):
            j = len(pool) - 1
        pos = pool.pop(j)
        out.append(eligible[pos])
    return out


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def run_simulation(
    months:               int   = SIM_MONTHS,
    initial_agents:       int   = INITIAL_POPULATION,
    arrival_lambda:       float = ARRIVAL_LAMBDA,
    admission_lambda:     float = ADMISSION_LAMBDA,
    admission_prob:       float = ADMISSION_PROB,
    readmission_prob:     float = READMISSION_PROB,
    los_mean:             float = LOS_MEAN_DAYS,
    los_sd:               float = LOS_SD_DAYS,
    seasonal:             bool  = True,
    raincity_open:        bool  = False,
    double_raincity:      bool  = False,
    gibsons_wc:           bool  = False,
    sechelt_wc:           bool  = False,
    seed:                 int   = RNG_SEED,
    start_month:          int   = SIM_START_MONTH,
) -> dict:

    rng         = np.random.default_rng(seed)
    gender_keys = list(GENDER_PROBS.keys())
    gender_vals = list(GENDER_PROBS.values())

    eff_exit_base  = MONTHLY_SPONTANEOUS_EXIT
    eff_adm_base   = admission_prob
    eff_readm_base = readmission_prob

    if raincity_open:
        eff_adm_base   -= SHELTER_ADMISSION_REDUCTION
        eff_readm_base -= SHELTER_READMISSION_REDUCTION

    if double_raincity:
        eff_adm_base   -= EXTRA_SHELTER_ADMISSION_REDUCTION
        eff_readm_base -= EXTRA_SHELTER_READMISSION_REDUCTION

    eff_adm_base   = max(0.0, eff_adm_base)
    eff_readm_base = max(0.0, eff_readm_base)

    agents  = []
    next_id = 0

    def _new_agent(entry_month: int) -> Agent:
        nonlocal next_id
        ag = Agent(
            agent_id    = next_id,
            entry_month = entry_month,
            gender      = rng.choice(gender_keys, p=gender_vals),
            age_group   = rng.choice(AGE_GROUPS,  p=AGE_PROBS),
        )
        next_id += 1
        return ag

    for _ in range(initial_agents):
        agents.append(_new_agent(0))

    initial_shelter_capacity = 0
    if raincity_open:
        initial_shelter_capacity += RAINCITY_CAPACITY_PERMANENT
    if double_raincity:
        initial_shelter_capacity += RAINCITY2_CAPACITY_PERMANENT

    if initial_shelter_capacity > 0:
        n_target = int(initial_shelter_capacity * 0.75)
        init_homeless = [ag for ag in agents if ag.state is State.HOMELESS]
        n_target = min(n_target, len(init_homeless))
        if n_target > 0:
            idxs = rng.choice(len(init_homeless), size=n_target, replace=False)
            for idx in idxs:
                init_homeless[int(idx)].in_shelter = True

    monthly_arrivals                = []
    monthly_new_homeless            = []
    monthly_admissions              = []
    monthly_readmissions            = []
    monthly_overflow                = []
    monthly_discharges              = []
    monthly_deaths                  = []
    monthly_exits                   = []
    monthly_occupancy               = []
    monthly_pop_homeless            = []
    monthly_bed_days                = []
    monthly_cost_hospital_base      = []   # fixed operating cost (same every scenario)
    monthly_cost_hospital_homeless  = []   # homeless-attributable bed-day cost
    monthly_cost_hospital           = []   # total (base + homeless)
    monthly_cost_shelter            = []
    monthly_cost_total              = []

    for sim_month in range(1, months + 1):
        cal_month          = ((start_month - 1 + sim_month - 1) % 12) + 1
        is_winter          = cal_month in WINTER_MONTHS
        is_raincity_season = cal_month in RAINCITY_SEASONAL_MONTHS
        is_gibsons_season  = cal_month in GIBSONS_WC_MONTHS
        is_sechelt_season  = cal_month in SECHELT_WC_MONTHS

        # ── 1. Advance all agents ─────────────────────────────────────────────
        bed_days_this   = 0.0
        discharges_this = 0
        for ag in agents:
            if not ag.is_active:
                continue
            bd = ag.update(sim_month)
            bed_days_this += bd
            if ag.state is State.POST_DISCHARGE and ag.discharge_month == sim_month:
                discharges_this += 1

        # ── 2. Mortality ──────────────────────────────────────────────────────
        deaths_this = 0
        for ag in agents:
            if ag.is_active and rng.random() < MONTHLY_MORTALITY_RATE:
                ag.die()
                deaths_this += 1

        # ── 3. Housing exits ──────────────────────────────────────────────────
        exits_this = 0
        for ag in agents:
            if ag.state is State.HOMELESS and (not ag.in_shelter) and rng.random() < eff_exit_base:
                ag.house()
                exits_this += 1

        sheltered_for_housing = [ag for ag in agents if ag.in_shelter]
        n_housed = rng.poisson(SHELTER_HOUSING_LAMBDA)
        n_housed = min(n_housed, len(sheltered_for_housing))
        if n_housed > 0:
            for idx in rng.choice(len(sheltered_for_housing), size=n_housed, replace=False):
                ag = sheltered_for_housing[int(idx)]
                ag.house()
                ag.in_shelter = False
                exits_this += 1

        for ag in agents:
            if ag.in_shelter and rng.random() < SHELTER_DROPOUT_RATE:
                ag.in_shelter = False

        # ── 4. New homeless entries ───────────────────────────────────────────
        n_new = rng.poisson(NEW_HOMELESS_LAMBDA)
        for _ in range(n_new):
            agents.append(_new_agent(sim_month))

        # ── 4b. Shelter bed capacity this month ───────────────────────────────
        for ag in agents:
            ag.in_warming_centre = False

        active_shelter_capacity = 0
        if raincity_open:
            active_shelter_capacity += RAINCITY_CAPACITY_PERMANENT
            if is_raincity_season:
                active_shelter_capacity += RAINCITY_CAPACITY_SEASONAL
        if double_raincity:
            active_shelter_capacity += RAINCITY2_CAPACITY_PERMANENT
            if is_raincity_season:
                active_shelter_capacity += RAINCITY2_CAPACITY_SEASONAL

        currently_sheltered = sum(1 for ag in agents if ag.in_shelter)
        available_beds = active_shelter_capacity - currently_sheltered
        if available_beds > 0:
            unsheltered = [ag for ag in agents if ag.state is State.HOMELESS and not ag.in_shelter]
            n_to_fill = min(available_beds, len(unsheltered))
            if n_to_fill > 0:
                for idx in rng.choice(len(unsheltered), size=n_to_fill, replace=False):
                    unsheltered[int(idx)].in_shelter = True

        if gibsons_wc and is_gibsons_season:
            elig_g = [ag for ag in agents if ag.state is State.HOMELESS and not ag.in_shelter]
            n_g = min(GIBSONS_WC_CAPACITY, len(elig_g))
            if n_g > 0:
                for idx in rng.choice(len(elig_g), size=n_g, replace=False):
                    elig_g[int(idx)].in_warming_centre = True

        if sechelt_wc and is_sechelt_season:
            elig_s = [ag for ag in agents if ag.state is State.HOMELESS
                      and not ag.in_shelter and not ag.in_warming_centre]
            n_s = min(SECHELT_WC_CAPACITY, len(elig_s))
            if n_s > 0:
                for idx in rng.choice(len(elig_s), size=n_s, replace=False):
                    elig_s[int(idx)].in_warming_centre = True

        # ── 5. ED arrivals ────────────────────────────────────────────────────
        avg_sickness = _monthly_avg_sickness_index(agents)
        arr_lambda_this = max(0.0, arrival_lambda + avg_sickness * ARRIVAL_SICKNESS_WEIGHT)
        n_arrivals = rng.poisson(arr_lambda_this)

        # ── 6. Effective admission probability ────────────────────────────────
        p_adm = eff_adm_base
        p_rdm = eff_readm_base
        if (gibsons_wc and is_gibsons_season) or (sechelt_wc and is_sechelt_season):
            p_adm = max(0.0, p_adm - WARMING_ADMISSION_REDUCTION)
            p_rdm = max(0.0, p_rdm - WARMING_READMISSION_REDUCTION)
        if seasonal and is_winter:
            p_adm = min(1.0, p_adm + SEASONAL_ADMISSION_BOOST)
            p_rdm = min(1.0, p_rdm + SEASONAL_ADMISSION_BOOST)

        # ── 7. First-time admissions ──────────────────────────────────────────
        current_occupancy = sum(1 for ag in agents if ag.in_bed)
        eligible_new      = [ag for ag in agents if ag.eligible_new_admission]
        pool_lambda       = max(0.0, admission_lambda + avg_sickness * SICKNESS_LAMBDA_WEIGHT)
        n_pool            = rng.poisson(pool_lambda)
        n_admit_draw      = rng.binomial(n_pool, p_adm)
        n_admit           = min(n_admit_draw, len(eligible_new))
        n_overflow_this   = 0

        if n_admit > 0:
            picked = _weighted_admit_sample(eligible_new, n_admit, SICKNESS_ADMISSION_DRAW_WEIGHT, rng)
            for ag in picked:
                los = max(1.0, rng.normal(los_mean, los_sd))
                ag.admit(sim_month, los)
                current_occupancy += 1
                if current_occupancy > HOSPITAL_CAPACITY:
                    n_overflow_this += 1

        # ── 8. Readmissions ───────────────────────────────────────────────────
        eligible_readm = [ag for ag in agents if ag.eligible_readmission]
        n_readmit = 0
        if eligible_readm:
            n_readmit_draw = rng.binomial(len(eligible_readm), p_rdm)
            k_readmit = min(n_readmit_draw, len(eligible_readm))
            if k_readmit > 0:
                picked_readm = _weighted_admit_sample(
                    eligible_readm, k_readmit, SICKNESS_ADMISSION_DRAW_WEIGHT, rng)
                for ag in picked_readm:
                    los = max(1.0, rng.normal(los_mean, los_sd))
                    ag.admit(sim_month, los)
                    n_readmit += 1
                    current_occupancy += 1
                    if current_occupancy > HOSPITAL_CAPACITY:
                        n_overflow_this += 1

        # ── 9. Costs ──────────────────────────────────────────────────────────
        shelter_cost_this = 0.0
        if raincity_open:
            shelter_cost_this += RAINCITY_FIXED_MONTHLY
        if double_raincity:
            shelter_cost_this += RAINCITY2_FIXED_MONTHLY
        if gibsons_wc and is_gibsons_season:
            shelter_cost_this += GIBSONS_WC_FIXED_MONTHLY
        if sechelt_wc and is_sechelt_season:
            shelter_cost_this += SECHELT_WC_FIXED_MONTHLY

        # Hospital costs:
        # - base ($151M/yr) tracked separately as context; NOT added to scenario total
        # - PEH-attributable bed-day cost is the scenario-sensitive variable
        hosp_base_this     = HOSPITAL_BASE_MONTHLY
        hosp_homeless_this = bed_days_this * COST_HOSPITAL_BED_DAY
        hospital_cost      = hosp_base_this + hosp_homeless_this  # full hospital cost (reference only)
        total_cost         = hosp_homeless_this + shelter_cost_this  # scenario comparison total (no double-count)

        occupancy    = sum(1 for ag in agents if ag.in_bed)
        pop_homeless = sum(1 for ag in agents if ag.state is State.HOMELESS)

        monthly_arrivals.append(n_arrivals)
        monthly_new_homeless.append(n_new)
        monthly_admissions.append(n_admit)
        monthly_readmissions.append(n_readmit)
        monthly_overflow.append(n_overflow_this)
        monthly_discharges.append(discharges_this)
        monthly_deaths.append(deaths_this)
        monthly_exits.append(exits_this)
        monthly_occupancy.append(occupancy)
        monthly_pop_homeless.append(pop_homeless)
        monthly_bed_days.append(bed_days_this)
        monthly_cost_hospital_base.append(hosp_base_this)
        monthly_cost_hospital_homeless.append(hosp_homeless_this)
        monthly_cost_hospital.append(hospital_cost)
        monthly_cost_shelter.append(shelter_cost_this)
        monthly_cost_total.append(total_cost)

    return dict(
        monthly_arrivals                = monthly_arrivals,
        monthly_new_homeless            = monthly_new_homeless,
        monthly_admissions              = monthly_admissions,
        monthly_readmissions            = monthly_readmissions,
        monthly_overflow                = monthly_overflow,
        monthly_discharges              = monthly_discharges,
        monthly_deaths                  = monthly_deaths,
        monthly_exits                   = monthly_exits,
        monthly_occupancy               = monthly_occupancy,
        monthly_pop_homeless            = monthly_pop_homeless,
        monthly_bed_days                = monthly_bed_days,
        monthly_cost_hospital_base      = monthly_cost_hospital_base,
        monthly_cost_hospital_homeless  = monthly_cost_hospital_homeless,
        monthly_cost_hospital           = monthly_cost_hospital,
        monthly_cost_shelter            = monthly_cost_shelter,
        monthly_cost_total              = monthly_cost_total,
        agents                          = agents,
    )


# =============================================================================
# SCENARIO RUNNER
# =============================================================================

SCENARIO_DEFINITIONS = {
    "Baseline\n(RainCity + Gibsons WC)": dict(
        raincity_open=True,  double_raincity=False, gibsons_wc=True,  sechelt_wc=False),
    "Double RainCity\n(+ Gibsons WC)": dict(
        raincity_open=True,  double_raincity=True,  gibsons_wc=True,  sechelt_wc=False),
    "RainCity + Sechelt WC\n(+ Gibsons WC)": dict(
        raincity_open=True,  double_raincity=False, gibsons_wc=True,  sechelt_wc=True),
}

COLORS = {
    "Baseline\n(RainCity + Gibsons WC)":        "#e74c3c",
    "Double RainCity\n(+ Gibsons WC)":           "#3498db",
    "RainCity + Sechelt WC\n(+ Gibsons WC)":    "#27ae60",
}

def run_all_scenarios(months: int = SIM_MONTHS, seed: int = RNG_SEED) -> dict:
    return {
        name: run_simulation(months=months, seed=seed, **flags)
        for name, flags in SCENARIO_DEFINITIONS.items()
    }


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary(scenarios: dict):
    baseline_cost = sum(list(scenarios.values())[0]["monthly_cost_total"])
    w = 120
    annual_base_M = HOSPITAL_ANNUAL_OPERATING_COST / 1e6
    sim_years = SIM_MONTHS / 12
    print("\n" + "=" * w)
    print(f"  Hospital base operating cost: ${annual_base_M:.0f}M/yr "
          f"(${annual_base_M * sim_years:.0f}M over {sim_years:.0f} yrs) — reference only, not in scenario totals")
    print(f"  Scenario totals = PEH bed-day costs + shelter operating costs only (no double-count)")
    print("=" * w)
    print(f"{'Scenario':<42} {'Admits':>7} {'Re-adm':>7} {'Bed-Days':>10} {'Deaths':>7} "
          f"{'PEH Hosp $M':>12} {'Shelter $M':>11} {'Total $M':>10} {'vs Baseline':>13}")
    print("=" * w)
    for name, res in scenarios.items():
        label    = name.replace("\n", " ")
        admits   = sum(res["monthly_admissions"])
        readmits = sum(res["monthly_readmissions"])
        bed_days = sum(res["monthly_bed_days"])
        deaths   = sum(res["monthly_deaths"])
        h_peh    = sum(res["monthly_cost_hospital_homeless"])
        s_cost   = sum(res["monthly_cost_shelter"])
        t_cost   = sum(res["monthly_cost_total"])
        delta    = baseline_cost - t_cost
        dstr     = f"-${delta/1e6:.3f}M" if delta >= 0 else f"+${abs(delta)/1e6:.3f}M"
        peh_pct  = (h_peh / (HOSPITAL_ANNUAL_OPERATING_COST * sim_years)) * 100
        print(f"{label:<42} {admits:>7} {readmits:>7} {bed_days:>10.0f} {deaths:>7} "
              f"{h_peh/1e6:>10.3f}M ({peh_pct:.1f}%) {s_cost/1e6:>9.3f} {t_cost/1e6:>10.3f} {dstr:>13}")
    print("=" * w + "\n")


# =============================================================================
# BREAK-EVEN ANALYSIS
# NOTE: The fixed $151M/yr base operating cost is identical across all scenarios
# and cancels out in the differential. Break-even is driven purely by
# homeless-attributable bed-day savings vs additional shelter operating costs.
# =============================================================================

def breakeven_analysis(scenarios: dict, months: int = SIM_MONTHS) -> dict:
    baseline           = list(scenarios.values())[0]
    baseline_hosp_peh  = np.array(baseline["monthly_cost_hospital_homeless"])
    baseline_shelter   = np.array(baseline["monthly_cost_shelter"])
    results            = {}

    for name, res in list(scenarios.items())[1:]:
        scen_hosp_peh  = np.array(res["monthly_cost_hospital_homeless"])
        scen_shelter   = np.array(res["monthly_cost_shelter"])
        cum_savings    = np.cumsum(baseline_hosp_peh - scen_hosp_peh)
        cum_extra_shlt = np.cumsum(scen_shelter - baseline_shelter)
        net            = cum_savings - cum_extra_shlt

        breakeven_month = None
        for t, n in enumerate(net, start=1):
            if n >= 0:
                breakeven_month = t
                break

        results[name] = dict(
            breakeven_month  = breakeven_month,
            cum_savings      = cum_savings,
            cum_shelter_cost = cum_extra_shlt,
            net              = net,
        )
    return results


def print_breakeven(scenarios: dict):
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    be = breakeven_analysis(scenarios)

    print("\n" + "=" * 72)
    print("BREAK-EVEN ANALYSIS")
    print("(When PEH-attributable hospital savings exceed additional shelter costs vs Baseline)")
    print("Note: $151M/yr hospital base cost is fixed across scenarios and excluded")
    print("=" * 72)
    print(f"{'Scenario':<42} {'Break-even':>12} {'Calendar date':>16} {'Net saving at end':>18}")
    print("-" * 72)

    for name, data in be.items():
        label   = name.replace("\n", " ")
        bm      = data["breakeven_month"]
        net_end = data["net"][-1]

        if bm is not None:
            cal_idx = ((SIM_START_MONTH - 1 + bm - 1) % 12)
            yr      = 26 + ((SIM_START_MONTH - 1 + bm - 1) // 12)
            cal_str = f"{month_names[cal_idx]} 20{yr}"
            bm_str  = f"Month {bm}"
        else:
            cal_str = "Beyond simulation window"
            bm_str  = f"> {SIM_MONTHS} months"

        net_str = f"${net_end/1e6:.3f}M" if net_end >= 0 else f"-${abs(net_end)/1e6:.3f}M"
        print(f"{label:<42} {bm_str:>12} {cal_str:>16} {net_str:>18}")

    print("=" * 72 + "\n")


def plot_breakeven(scenarios: dict, months: int = SIM_MONTHS):
    be     = breakeven_analysis(scenarios, months)
    labels = _month_labels(months)
    x      = np.arange(1, months + 1)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.axhline(0, color="black", lw=1.8, ls="--", label="Break-even line (net = $0)")

    for name, data in be.items():
        c   = COLORS[name]
        net = data["net"] / 1e6
        ax.plot(x, net, color=c, lw=2.5, marker="o", ms=3.5,
                label=name.replace("\n", " "))

        bm = data["breakeven_month"]
        if bm is not None:
            ax.axvline(bm, color=c, lw=1.2, ls=":", alpha=0.7)
            y_label = data["net"][min(bm, months - 1)] / 1e6
            ax.annotate(
                f"Month {bm}",
                xy=(bm, 0),
                xytext=(bm + 0.4, max(y_label * 0.4, 0.03)),
                fontsize=7.5, color=c, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=c, lw=1),
            )

    tick_every = 3
    ax.set_xticks(x[::tick_every])
    ax.set_xticklabels(labels[::tick_every], rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("Cumulative Net Saving vs Baseline (M CAD)\n"
                  "[PEH hospital savings minus additional shelter costs]")
    ax.set_title(
        "Break-Even Analysis — When Do Additional Shelter Interventions Pay For Themselves?\n"
        "Sechelt | shíshálh Hospital ABM\n"
        "(Hospital $151M/yr base operating cost fixed across all scenarios — excluded from differential)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("outputs/sechelt_breakeven.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: sechelt_breakeven.png")


# =============================================================================
# PLOTTING
# =============================================================================

def _month_labels(months: int, start_month: int = SIM_START_MONTH) -> list[str]:
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    labels = []
    for i in range(months):
        cal = ((start_month - 1 + i) % 12)
        yr  = 26 + ((start_month - 1 + i) // 12)
        labels.append(f"{month_names[cal]} '{yr}")
    return labels


def plot_main(scenarios: dict, months: int = SIM_MONTHS):
    labels     = _month_labels(months)
    x          = np.arange(months)
    tick_every = 3

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes      = axes.flatten()

    panels = [
        ("monthly_occupancy",  "Hospital Occupancy — End of Month",               "# Beds Occupied"),
        ("monthly_admissions", "Monthly Admissions (first-time)",                  "# Admissions"),
        ("monthly_overflow",   "Monthly Overflow Events\n(above 63-bed capacity)", "# Overflow Admissions"),
    ]

    for ax, (key, title, ylabel) in zip(axes[:3], panels):
        for name, res in scenarios.items():
            ax.plot(x, res[key], color=COLORS[name], marker='o', ms=3.5,
                    label=name.replace("\n", " "))
        if key == "monthly_occupancy":
            ax.axhline(HOSPITAL_CAPACITY, color='black', ls='--', lw=1.2,
                       label=f"Capacity ({HOSPITAL_CAPACITY} beds)")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x[::tick_every])
        ax.set_xticklabels(labels[::tick_every], rotation=35, ha='right', fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    ax4 = axes[3]
    for name, res in scenarios.items():
        ax4.plot(x, np.cumsum(res["monthly_cost_total"]) / 1e6,
                 color=COLORS[name], lw=2.5, label=name.replace("\n", " "))
    ax4.set_title("Cumulative Total Cost\n(Hospital Base + PEH Bed-Days + Shelter)", fontsize=10)
    ax4.set_ylabel("Cumulative Cost (M CAD)")
    ax4.set_xticks(x[::tick_every])
    ax4.set_xticklabels(labels[::tick_every], rotation=35, ha='right', fontsize=7)
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.25)

    fig.suptitle(
        "Sechelt | shíshálh Hospital — Homeless ED Utilization ABM\n"
        "Sechelt & Gibsons, BC  |  Simulation starts April 2026  |  Hospital base operating cost: $151M/yr",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("outputs/sechelt_main.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: sechelt_main.png")


def plot_cost_breakdown(scenarios: dict, months: int = SIM_MONTHS):
    """
    Stacked bar: PEH bed-day costs (red) + shelter costs (blue).
    Hospital base operating cost ($151M/yr) shown as reference line only — not stacked,
    to avoid double-counting with the blended bed-day rate.
    """
    names    = [n.replace("\n", " ") for n in scenarios]
    peh_cum  = [sum(s["monthly_cost_hospital_homeless"]) / 1e6 for s in scenarios.values()]
    shlt_cum = [sum(s["monthly_cost_shelter"])           / 1e6 for s in scenarios.values()]
    x        = np.arange(len(names))
    w        = 0.5
    sim_years = months / 12
    base_ref  = HOSPITAL_ANNUAL_OPERATING_COST * sim_years / 1e6

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, peh_cum, w,
           label="PEH-attributable bed-day costs (scenario variable)",
           color="#e74c3c", alpha=0.88)
    ax.bar(x, shlt_cum, w, bottom=peh_cum,
           label="Shelter / Warming-centre operating costs",
           color="#3498db", alpha=0.88)

    baseline_total = peh_cum[0] + shlt_cum[0]
    for i, (p, s) in enumerate(zip(peh_cum, shlt_cum)):
        total   = p + s
        saving  = baseline_total - total
        peh_pct = (p / base_ref) * 100
        label   = f"${total:.2f}M\n(PEH = {peh_pct:.1f}% of hosp. budget)"
        if saving > 0:
            label += f"\nsaves ${saving:.2f}M vs baseline"
        ax.text(i, total + 0.05, label,
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.axhline(base_ref, color='grey', ls='--', lw=1.5, alpha=0.7,
               label=f"Hospital total operating cost over {sim_years:.0f} yrs "
                     f"(${base_ref:.0f}M — reference only, not in bars)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Cumulative Cost (M CAD)")
    ax.set_title(
        f"Cumulative Cost Breakdown Over {months} Months — Sechelt ABM\n"
        f"Bars show PEH-attributable costs only. Hospital base (${HOSPITAL_ANNUAL_OPERATING_COST/1e6:.0f}M/yr) "
        f"shown as reference line.",
        fontsize=11
    )
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig("outputs/sechelt_cost_breakdown.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: sechelt_cost_breakdown.png")


def plot_cost_per_night():
    raincity_per_bed_night = RAINCITY_FIXED_MONTHLY / 30 / RAINCITY_CAPACITY_PERMANENT
    gibsons_per_bed_night  = GIBSONS_WC_FIXED_MONTHLY / 30 / GIBSONS_WC_CAPACITY

    labels  = [
        f"ICU Bed\n($7,000/night)",
        f"ICU Bed\n($10,000/night)",
        f"Non-ICU Bed\n(${COST_NON_ICU_BED_DAY:,}/night)",
        f"RainCity Shelter\nBed (${raincity_per_bed_night:,.0f}/night)",
        f"Gibsons/Sechelt WC\nBed (${gibsons_per_bed_night:,.0f}/night)",
    ]
    values  = [
        COST_ICU_BED_DAY_LOW,
        COST_ICU_BED_DAY_HIGH,
        COST_NON_ICU_BED_DAY,
        raincity_per_bed_night,
        gibsons_per_bed_night,
    ]
    colours = ["#c0392b", "#e74c3c", "#e67e22", "#3498db", "#27ae60"]
    x       = np.arange(len(labels))
    w       = 0.55

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(x, values, width=w, color=colours, alpha=0.88, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 80,
                f"${val:,.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    y_bracket = COST_ICU_BED_DAY_HIGH * 1.08
    ax.annotate("", xy=(x[1], y_bracket), xytext=(x[0], y_bracket),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text((x[0] + x[1]) / 2, y_bracket + 150, "ICU range", ha='center', fontsize=8.5)

    ratio_icu_low  = COST_ICU_BED_DAY_LOW  / raincity_per_bed_night
    ratio_icu_high = COST_ICU_BED_DAY_HIGH / raincity_per_bed_night
    ratio_non_icu  = COST_NON_ICU_BED_DAY  / raincity_per_bed_night
    ax.text(0.98, 0.97,
            f"ICU is {ratio_icu_low:.0f}–{ratio_icu_high:.0f}× more costly than RainCity shelter\n"
            f"Non-ICU is {ratio_non_icu:.0f}× more costly than RainCity shelter\n"
            f"Hospital total operating cost: ${HOSPITAL_ANNUAL_OPERATING_COST/1e6:.0f}M/yr",
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            bbox=dict(boxstyle='round', facecolor='#fffbcc', alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("CAD per bed-night")
    ax.set_ylim(0, COST_ICU_BED_DAY_HIGH * 1.30)
    ax.set_title(
        "Cost per Bed-Night by Care Setting — Sechelt | shíshálh Hospital & RainCity Shelter\n"
        f"(5 ICU beds + 58 non-ICU beds; RainCity $1.6M/yr ÷ {RAINCITY_CAPACITY_PERMANENT} beds ÷ 365 days; Hospital base: $151M/yr)",
        fontsize=10
    )
    ax.grid(axis='y', alpha=0.25, zorder=0)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#c0392b", alpha=0.88, label="ICU — low estimate ($7,000)"),
        Patch(facecolor="#e74c3c", alpha=0.88, label="ICU — high estimate ($10,000)"),
        Patch(facecolor="#e67e22", alpha=0.88, label=f"Non-ICU (${COST_NON_ICU_BED_DAY:,})"),
        Patch(facecolor="#3498db", alpha=0.88, label="RainCity shelter bed"),
        Patch(facecolor="#27ae60", alpha=0.88, label="Gibsons / Sechelt warming centre bed"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.savefig("outputs/sechelt_cost_per_night.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: sechelt_cost_per_night.png")


def plot_demographics(scenarios: dict):
    admitted = [ag for ag in list(scenarios.values())[0]["agents"]
                if ag.admission_count > 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    gc    = Counter(ag.gender for ag in admitted)
    axes[0].bar(gc.keys(), gc.values(), color=["#3498db","#e74c3c","#95a5a6"])
    axes[0].set_title("Admitted Agents by Gender (Baseline)")
    axes[0].set_ylabel("Count")

    ac    = Counter(ag.age_group for ag in admitted)
    order = ["18-29","30-44","45-59","60+"]
    axes[1].bar(order, [ac.get(g, 0) for g in order], color="#27ae60")
    axes[1].set_title("Admitted Agents by Age Group (Baseline)")
    axes[1].set_ylabel("Count")

    rc      = Counter(ag.admission_count for ag in admitted)
    max_adm = max(rc.keys(), default=1)
    axes[2].bar(range(1, max_adm + 1),
                [rc.get(i, 0) for i in range(1, max_adm + 1)],
                color="#9b59b6")
    axes[2].set_title("Admission Frequency per Agent (Baseline)")
    axes[2].set_xlabel("Total Admissions per Individual")
    axes[2].set_ylabel("Count")

    fig.suptitle(
        "Demographics — Admitted Homeless Individuals, Sechelt ABM (Baseline)",
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("outputs/sechelt_demographics.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: sechelt_demographics.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Sechelt | shíshálh Homeless ED Utilization ABM")
    print(f"  Hospital capacity : {HOSPITAL_CAPACITY} beds (Sechelt | shíshálh Hospital)")
    print(f"    ICU beds        : {HOSPITAL_ICU_BEDS} @ ${COST_ICU_BED_DAY_LOW:,}–${COST_ICU_BED_DAY_HIGH:,}/day")
    print(f"    Non-ICU beds    : {HOSPITAL_NON_ICU_BEDS} @ ${COST_NON_ICU_BED_DAY:,}/day")
    print(f"    Blended cost    : ${COST_HOSPITAL_BED_DAY:,}/day (used in simulation)")
    print(f"  Hospital base ops : ${HOSPITAL_ANNUAL_OPERATING_COST/1e6:.0f}M/yr → ${HOSPITAL_BASE_MONTHLY:,}/month (fixed)")
    print(f"  RainCity shelter  : $1.6M/yr → ${RAINCITY_FIXED_MONTHLY:,}/month")
    print(f"    Permanent beds  : {RAINCITY_CAPACITY_PERMANENT}")
    print(f"    Seasonal beds   : {RAINCITY_CAPACITY_SEASONAL} (Oct–Apr)")
    print(f"  Gibsons WC        : ${GIBSONS_WC_FIXED_MONTHLY:,}/month (Oct–Apr, {GIBSONS_WC_CAPACITY} spaces)")
    print(f"  Sechelt WC        : ${SECHELT_WC_FIXED_MONTHLY:,}/month (Oct–Apr, {SECHELT_WC_CAPACITY} spaces) [scenario 3 only]")
    print(f"  Start             : April 2026  |  Duration: {SIM_MONTHS} months")
    print(f"  Initial population: {INITIAL_POPULATION}")
    print(f"  Admission prob    : {ADMISSION_PROB:.0%}  ← SWAP with Z59/(NFA+Z59)")
    print(f"  LOS mean          : {LOS_MEAN_DAYS} days  ← SWAP with real data")
    print("=" * 60 + "\n")

    scenarios = run_all_scenarios(months=SIM_MONTHS, seed=RNG_SEED)

    print_summary(scenarios)
    print_breakeven(scenarios)
    plot_cost_per_night()
    plot_main(scenarios, months=SIM_MONTHS)
    plot_cost_breakdown(scenarios, months=SIM_MONTHS)
    plot_breakeven(scenarios, months=SIM_MONTHS)
    plot_demographics(scenarios)
