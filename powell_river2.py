"""
Agent-Based Model: Homeless Population & Emergency Department Utilization
Powell River (qathet), BC
=========================================================================

Real-world context
------------------
  Hospital     : qathet General Hospital — 42 acute beds
  Shelter hist : Joyce Ave (20 beds) CLOSED March 2025
                 → zero shelter beds from March 2025 onward
  Sim start    : April 2026
  Driftwood    : 40-bed shelter at 7104 Barnet St opens April 2026 (month 1)
  Population   : ~126 homeless individuals (2023 PiT count)

Scenarios
---------
  1. Baseline          — no shelter (reflects post-March 2025 reality)
  2. Driftwood Opens   — 40-bed year-round shelter active from month 1 (April 2026)
  3. Double Shelter    — Driftwood + additional 40-bed shelter (hypothetical expansion)
  4. Warming Centre    — Driftwood + 20-bed warming centre (Oct–Apr only)

Hospital overflow
-----------------
  Hard capacity = 42 beds.
  When occupancy >= 42, new admissions are still recorded but flagged as overflow
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
HOSPITAL_CAPACITY   = 42          # qathet General Hospital acute beds

# ── Hospital base operating cost ──────────────────────────────────────────────
# Total annual operating cost of qathet General Hospital (~$101M/yr).
# This is a fixed cost present in every scenario — not attributable to any
# single patient group. Included so total cost figures reflect the real
# financial burden and shelter savings are shown in proper context.
HOSPITAL_ANNUAL_OPERATING_COST  = 101_000_000          # CAD/yr
HOSPITAL_BASE_MONTHLY           = HOSPITAL_ANNUAL_OPERATING_COST // 12   # ≈ $8,416,667/month

# ── Arrival process ───────────────────────────────────────────────────────────
ARRIVAL_LAMBDA      = 11          # Poisson mean: homeless individuals arriving at ED/month
                                  # ← SWAP with real ED visit counts once data arrives
ARRIVAL_SICKNESS_WEIGHT = 5.0    # scales avg sickness → extra Poisson rate for ED arrivals

# ── Admission process ─────────────────────────────────────────────────────────
ADMISSION_LAMBDA    = 10          # Poisson pool size n for Binomial draw
ADMISSION_PROB      = 0.10        # Binomial p  ← SWAP with Z59/(NFA+Z59) from real data

# ── Sickness index (-1 … 1) & admission coupling ─────────────────────────────
# Index: 0 = average risk, +1 = much more likely ill, -1 = much less likely ill.
# Poisson pool rate each month: ADMISSION_LAMBDA + (monthly avg sickness index) * weight
SICKNESS_LAMBDA_WEIGHT          = 2.0     # scales avg sickness → extra Poisson rate
# Who gets picked for first-time admission: weight_i = 1 + w * sickness_index(agent_i)
SICKNESS_ADMISSION_DRAW_WEIGHT  = 0.5     # keep < 1 if s=-1 must stay positive (1+w*s>0)
# Component scores (sum clamped to [-1, 1]); ← SWAP with calibration / literature
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
INITIAL_POPULATION  = 80          # ← SWAP with confirmed PiT count when available

# ── Open-population dynamics ──────────────────────────────────────────────────
NEW_HOMELESS_LAMBDA         = 8   # Poisson mean: new entries into homelessness/month
                                  # ← SWAP with local inflow estimates
MONTHLY_MORTALITY_RATE      = 0.005  # ~6% annual; elevated vs general pop  ← SWAP
MONTHLY_SPONTANEOUS_EXIT    = 0.015  # ~18% annual housed/left area          ← SWAP

# ── Readmissions ──────────────────────────────────────────────────────────────
READMISSION_PROB            = 0.22   # prob of readmission per eligible month ← SWAP
READMISSION_RISK_MONTHS     = 2      # months post-discharge at elevated risk

# ── Seasonal parameters ───────────────────────────────────────────────────────
WINTER_MONTHS               = {11, 12, 1, 2, 3}
SEASONAL_ADMISSION_BOOST    = 0.05   # extra admission prob in winter months

# ── Demographics ──────────────────────────────────────────────────────────────
# BC Coroners data: 79% male, skewed 30-49
# ← SWAP with real age/gender from qathet hospital data
GENDER_PROBS = {"Male": 0.70, "Female": 0.26, "Other": 0.04}
AGE_GROUPS   = ["18-29", "30-44", "45-59", "60+"]
AGE_PROBS    = [0.20,    0.35,    0.30,    0.15]

# ── Cost parameters ───────────────────────────────────────────────────────────
# qathet General Hospital — 42 beds total: 3 ICU + 39 non-ICU
HOSPITAL_ICU_BEDS               = 3
HOSPITAL_NON_ICU_BEDS           = 39      # 42 - 3
COST_ICU_BED_DAY_LOW            = 7_000   # CAD; ICU bed-day lower bound
COST_ICU_BED_DAY_HIGH           = 10_000  # CAD; ICU bed-day upper bound
COST_NON_ICU_BED_DAY            = 1_100   # CAD; non-ICU acute bed-day

# Weighted average bed-day cost used in simulation:
# (3 ICU × midpoint $8,500 + 39 non-ICU × $1,100) / 42  ≈ $1,624/day blended
_ICU_MID                        = (COST_ICU_BED_DAY_LOW + COST_ICU_BED_DAY_HIGH) / 2
COST_HOSPITAL_BED_DAY           = round(
    (HOSPITAL_ICU_BEDS * _ICU_MID + HOSPITAL_NON_ICU_BEDS * COST_NON_ICU_BED_DAY)
    / (HOSPITAL_ICU_BEDS + HOSPITAL_NON_ICU_BEDS)
)

# Driftwood shelter — confirmed $1.6M/yr operating cost
DRIFTWOOD_CAPACITY              = 40      # beds
DRIFTWOOD_FIXED_MONTHLY         = 133_333 # CAD; $1,600,000 / 12

# Additional 40-bed shelter (same cost structure as Driftwood)
ADDITIONAL_SHELTER_CAPACITY     = 40
ADDITIONAL_SHELTER_FIXED_MONTHLY = 133_333

# Warming centre (Oct–Apr; 20 beds; lower cost than year-round shelter)
WARMING_CENTRE_CAPACITY         = 20
WARMING_CENTRE_MONTHS           = {10, 11, 12, 1, 2, 3, 4}   # Oct–Apr
WARMING_CENTRE_FIXED_MONTHLY    = 40_000  # CAD/month while open
                                          # ← SWAP with actual contract value

# ── Shelter diversion effects ─────────────────────────────────────────────────
SHELTER_ADMISSION_REDUCTION     = 0.05
SHELTER_READMISSION_REDUCTION   = 0.08
SHELTER_HOUSING_LAMBDA          = 1.6
SHELTER_DROPOUT_RATE            = 0.05
SHELTER_SICKNESS_REDUCTION       = 0.40

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
    driftwood_open:       bool  = False,
    extra_shelter:        bool  = False,
    warming_centre:       bool  = False,
    seed:                 int   = RNG_SEED,
    start_month:          int   = SIM_START_MONTH,
) -> dict:

    rng         = np.random.default_rng(seed)
    gender_keys = list(GENDER_PROBS.keys())
    gender_vals = list(GENDER_PROBS.values())

    eff_exit_base  = MONTHLY_SPONTANEOUS_EXIT
    eff_adm_base   = admission_prob
    eff_readm_base = readmission_prob

    if driftwood_open:
        eff_adm_base    -= SHELTER_ADMISSION_REDUCTION
        eff_readm_base  -= SHELTER_READMISSION_REDUCTION

    if extra_shelter:
        eff_adm_base    -= EXTRA_SHELTER_ADMISSION_REDUCTION
        eff_readm_base  -= EXTRA_SHELTER_READMISSION_REDUCTION
    eff_adm_base    = max(0.0, eff_adm_base)
    eff_readm_base  = max(0.0, eff_readm_base)

    agents   = []
    next_id  = 0

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

    active_year_round_shelter_capacity = 0
    if driftwood_open:
        active_year_round_shelter_capacity += DRIFTWOOD_CAPACITY
    if extra_shelter:
        active_year_round_shelter_capacity += ADDITIONAL_SHELTER_CAPACITY

    if active_year_round_shelter_capacity > 0:
        n_target_sheltered = int(active_year_round_shelter_capacity * 0.75)
        initial_homeless = [ag for ag in agents if ag.state is State.HOMELESS]
        n_target_sheltered = min(n_target_sheltered, len(initial_homeless))
        if n_target_sheltered > 0:
            sheltered_idxs = rng.choice(
                len(initial_homeless),
                size=n_target_sheltered,
                replace=False,
            )
            for idx in sheltered_idxs:
                initial_homeless[int(idx)].in_shelter = True

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
    monthly_cost_hospital           = []   # total hospital cost (base + homeless)
    monthly_cost_shelter            = []
    monthly_cost_total              = []

    for sim_month in range(1, months + 1):
        cal_month = ((start_month - 1 + sim_month - 1) % 12) + 1
        is_winter = cal_month in WINTER_MONTHS
        is_warming_season = cal_month in WARMING_CENTRE_MONTHS

        bed_days_this = 0.0
        discharges_this = 0
        for ag in agents:
            if not ag.is_active:
                continue
            bd = ag.update(sim_month)
            bed_days_this += bd
            if ag.state is State.POST_DISCHARGE and ag.discharge_month == sim_month:
                discharges_this += 1

        deaths_this = 0
        for ag in agents:
            if ag.is_active and rng.random() < MONTHLY_MORTALITY_RATE:
                ag.die()
                deaths_this += 1

        exits_this = 0
        for ag in agents:
            if ag.state is State.HOMELESS and (not ag.in_shelter) and rng.random() < eff_exit_base:
                ag.house()
                exits_this += 1

        sheltered_for_housing = [ag for ag in agents if ag.in_shelter]
        n_housed = rng.poisson(SHELTER_HOUSING_LAMBDA)
        n_housed = min(n_housed, len(sheltered_for_housing))
        if n_housed > 0:
            housed_idxs = rng.choice(
                len(sheltered_for_housing),
                size=n_housed,
                replace=False,
            )
            for idx in housed_idxs:
                ag = sheltered_for_housing[int(idx)]
                ag.house()
                ag.in_shelter = False
                exits_this += 1

        for ag in agents:
            if ag.in_shelter and rng.random() < SHELTER_DROPOUT_RATE:
                ag.in_shelter = False

        n_new = rng.poisson(NEW_HOMELESS_LAMBDA)
        for _ in range(n_new):
            agents.append(_new_agent(sim_month))

        for ag in agents:
            ag.in_warming_centre = False

        active_year_round_shelter_capacity = 0
        if driftwood_open:
            active_year_round_shelter_capacity += DRIFTWOOD_CAPACITY
        if extra_shelter:
            active_year_round_shelter_capacity += ADDITIONAL_SHELTER_CAPACITY

        currently_sheltered = sum(1 for ag in agents if ag.in_shelter)
        available_beds = active_year_round_shelter_capacity - currently_sheltered
        if available_beds > 0:
            unsheltered_homeless = [ag for ag in agents if ag.state is State.HOMELESS and (not ag.in_shelter)]
            n_to_fill = min(available_beds, len(unsheltered_homeless))
            if n_to_fill > 0:
                fill_idxs = rng.choice(
                    len(unsheltered_homeless),
                    size=n_to_fill,
                    replace=False,
                )
                for idx in fill_idxs:
                    unsheltered_homeless[int(idx)].in_shelter = True

        if warming_centre and is_warming_season:
            eligible_warming = [
                ag for ag in agents
                if ag.state is State.HOMELESS and (not ag.in_shelter)
            ]
            n_warming = min(WARMING_CENTRE_CAPACITY, len(eligible_warming))
            if n_warming > 0:
                warming_idxs = rng.choice(
                    len(eligible_warming),
                    size=n_warming,
                    replace=False,
                )
                for idx in warming_idxs:
                    eligible_warming[int(idx)].in_warming_centre = True

        avg_sickness = _monthly_avg_sickness_index(agents)
        arr_lambda_this = max(0.0, arrival_lambda + avg_sickness * ARRIVAL_SICKNESS_WEIGHT)
        n_arrivals = rng.poisson(arr_lambda_this)

        p_adm = eff_adm_base
        p_rdm = eff_readm_base
        if warming_centre and is_warming_season:
            p_adm = max(0.0, p_adm - WARMING_ADMISSION_REDUCTION)
            p_rdm = max(0.0, p_rdm - WARMING_READMISSION_REDUCTION)
        if seasonal and is_winter:
            p_adm = min(1.0, p_adm + SEASONAL_ADMISSION_BOOST)
            p_rdm = min(1.0, p_rdm + SEASONAL_ADMISSION_BOOST)

        current_occupancy  = sum(1 for ag in agents if ag.in_bed)
        eligible_new       = [ag for ag in agents if ag.eligible_new_admission]
        pool_lambda        = max(0.0, admission_lambda + avg_sickness * SICKNESS_LAMBDA_WEIGHT)
        n_pool             = rng.poisson(pool_lambda)
        n_admit_draw       = rng.binomial(n_pool, p_adm)
        n_admit            = min(n_admit_draw, len(eligible_new))
        n_overflow_this    = 0

        if n_admit > 0:
            picked = _weighted_admit_sample(
                eligible_new, n_admit, SICKNESS_ADMISSION_DRAW_WEIGHT, rng
            )
            for ag in picked:
                los = max(1.0, rng.normal(los_mean, los_sd))
                ag.admit(sim_month, los)
                current_occupancy += 1
                if current_occupancy > HOSPITAL_CAPACITY:
                    n_overflow_this += 1

        eligible_readm = [ag for ag in agents if ag.eligible_readmission]
        n_readmit = 0
        if eligible_readm:
            n_readmit_draw = rng.binomial(len(eligible_readm), p_rdm)
            k_readmit = min(n_readmit_draw, len(eligible_readm))
            if k_readmit > 0:
                picked_readm = _weighted_admit_sample(
                    eligible_readm, k_readmit, SICKNESS_ADMISSION_DRAW_WEIGHT, rng
                )
                for ag in picked_readm:
                    los = max(1.0, rng.normal(los_mean, los_sd))
                    ag.admit(sim_month, los)
                    n_readmit += 1
                    current_occupancy += 1
                    if current_occupancy > HOSPITAL_CAPACITY:
                        n_overflow_this += 1

        # ── Costs ─────────────────────────────────────────────────────────────
        shelter_cost_this = 0.0
        if driftwood_open:
            shelter_cost_this += DRIFTWOOD_FIXED_MONTHLY
        if extra_shelter:
            shelter_cost_this += ADDITIONAL_SHELTER_FIXED_MONTHLY
        if warming_centre and is_warming_season:
            shelter_cost_this += WARMING_CENTRE_FIXED_MONTHLY

        # Hospital costs:
        # - base ($101M/yr) tracked separately as context; NOT added to scenario total
        # - PEH-attributable bed-day cost is the scenario-sensitive variable
        hosp_base_this     = HOSPITAL_BASE_MONTHLY
        hosp_homeless_this = bed_days_this * COST_HOSPITAL_BED_DAY
        hospital_cost      = hosp_base_this + hosp_homeless_this  # full hospital cost (reference only)
        total_cost         = hosp_homeless_this + shelter_cost_this  # scenario comparison total (no double-count)

        occupancy       = sum(1 for ag in agents if ag.in_bed)
        pop_homeless    = sum(1 for ag in agents if ag.state is State.HOMELESS)

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
    "Baseline\n(no shelter)": dict(
        driftwood_open=False, extra_shelter=False, warming_centre=False),
    "Driftwood Opens\n(40 beds, Apr 2026)": dict(
        driftwood_open=True,  extra_shelter=False, warming_centre=False),
    "Double Shelter\n(+40 beds)": dict(
        driftwood_open=True,  extra_shelter=True,  warming_centre=False),
    "Driftwood +\nWarming Centre\n(20 beds Oct–Apr)": dict(
        driftwood_open=True,  extra_shelter=False, warming_centre=True),
}

COLORS = {
    "Baseline\n(no shelter)":                          "#e74c3c",
    "Driftwood Opens\n(40 beds, Apr 2026)":            "#3498db",
    "Double Shelter\n(+40 beds)":                      "#9b59b6",
    "Driftwood +\nWarming Centre\n(20 beds Oct–Apr)":  "#27ae60",
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
    print(f"{'Scenario':<38} {'Admits':>7} {'Re-adm':>7} {'Bed-Days':>10} {'Deaths':>7} "
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
        # PEH costs as % of annual hospital budget
        peh_pct  = (h_peh / (HOSPITAL_ANNUAL_OPERATING_COST * sim_years)) * 100
        print(f"{label:<38} {admits:>7} {readmits:>7} {bed_days:>10.0f} {deaths:>7} "
              f"{h_peh/1e6:>10.3f}M ({peh_pct:.1f}%) {s_cost/1e6:>9.3f} {t_cost/1e6:>10.3f} {dstr:>13}")
    print("=" * w + "\n")


# =============================================================================
# BREAK-EVEN ANALYSIS
# NOTE: The fixed $101M/yr base operating cost is identical across all scenarios
# and therefore cancels out in the differential calculation. Break-even is driven
# purely by homeless-attributable bed-day savings vs shelter operating costs.
# =============================================================================

def breakeven_analysis(scenarios: dict, months: int = SIM_MONTHS) -> dict:
    baseline          = list(scenarios.values())[0]
    baseline_hosp_peh = np.array(baseline["monthly_cost_hospital_homeless"])
    results           = {}

    for name, res in list(scenarios.items())[1:]:
        scen_hosp_peh = np.array(res["monthly_cost_hospital_homeless"])
        scen_shelter  = np.array(res["monthly_cost_shelter"])
        cum_savings   = np.cumsum(baseline_hosp_peh - scen_hosp_peh)
        cum_shelter   = np.cumsum(scen_shelter)
        net           = cum_savings - cum_shelter

        breakeven_month = None
        for t, n in enumerate(net, start=1):
            if n >= 0:
                breakeven_month = t
                break

        results[name] = dict(
            breakeven_month  = breakeven_month,
            cum_savings      = cum_savings,
            cum_shelter_cost = cum_shelter,
            net              = net,
        )
    return results


def print_breakeven(scenarios: dict):
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    be = breakeven_analysis(scenarios)

    print("\n" + "=" * 72)
    print("BREAK-EVEN ANALYSIS")
    print("(When PEH-attributable hospital savings exceed shelter operating costs)")
    print("Note: $101M/yr hospital base cost is fixed across scenarios and excluded")
    print("=" * 72)
    print(f"{'Scenario':<40} {'Break-even':>12} {'Calendar date':>16} {'Net saving at end':>18}")
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
        print(f"{label:<40} {bm_str:>12} {cal_str:>16} {net_str:>18}")

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
                  "[PEH hospital savings  minus  shelter operating costs]")
    ax.set_title(
        "Break-Even Analysis — When Do Shelter Interventions Pay For Themselves?\n"
        "Positive values = shelter has more than paid for itself through reduced homeless ED costs\n"
        "(Hospital $101M/yr base operating cost fixed across all scenarios — excluded from differential)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("outputs/powell_river_breakeven.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: powell_river_breakeven.png")


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
        ("monthly_occupancy",    "Hospital Occupancy — End of Month",      "# Beds Occupied"),
        ("monthly_admissions",   "Monthly Admissions (first-time)",         "# Admissions"),
        ("monthly_overflow",     "Monthly Overflow Events\n(above 42-bed capacity)", "# Overflow Admissions"),
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

    # ── Cumulative total cost (base + PEH + shelter) ──────────────────────────
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
        "qathet General Hospital — Homeless ED Utilization ABM\n"
        "Powell River, BC  |  Simulation starts April 2026  |  Hospital base operating cost: $101M/yr",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("outputs/powell_river_main.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: powell_river_main.png")


def plot_cost_breakdown(scenarios: dict, months: int = SIM_MONTHS):
    """
    Stacked bar: PEH bed-day costs (red) + shelter costs (blue).
    Hospital base operating cost ($101M/yr) shown as annotation only — not stacked,
    to avoid double-counting with the blended bed-day rate.
    """
    names    = [n.replace("\n", " ") for n in scenarios]
    peh_cum  = [sum(s["monthly_cost_hospital_homeless"]) / 1e6 for s in scenarios.values()]
    shlt_cum = [sum(s["monthly_cost_shelter"])           / 1e6 for s in scenarios.values()]
    x        = np.arange(len(names))
    w        = 0.5
    sim_years = months / 12
    base_ref  = HOSPITAL_ANNUAL_OPERATING_COST * sim_years / 1e6  # reference only

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, peh_cum, w,
           label="PEH-attributable bed-day costs (scenario variable)",
           color="#e74c3c", alpha=0.88)
    ax.bar(x, shlt_cum, w, bottom=peh_cum,
           label="Shelter / Warming-centre operating costs",
           color="#3498db", alpha=0.88)

    baseline_total = peh_cum[0] + shlt_cum[0]
    for i, (p, s) in enumerate(zip(peh_cum, shlt_cum)):
        total  = p + s
        saving = baseline_total - total
        peh_pct = (p / base_ref) * 100
        label  = f"${total:.2f}M\n(PEH = {peh_pct:.1f}% of hosp. budget)"
        if saving > 0:
            label += f"\nsaves ${saving:.2f}M vs baseline"
        ax.text(i, total + 0.05, label,
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Reference line for hospital base — annotated, not stacked
    ax.axhline(base_ref, color='grey', ls='--', lw=1.5, alpha=0.7,
               label=f"Hospital total operating cost over {sim_years:.0f} yrs "
                     f"(${base_ref:.0f}M — reference only, not in bars)")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Cumulative Cost (M CAD)")
    ax.set_title(
        f"Cumulative Cost Breakdown Over {months} Months — Powell River ABM\n"
        f"Bars show PEH-attributable costs only. Hospital base (${HOSPITAL_ANNUAL_OPERATING_COST/1e6:.0f}M/yr) "
        f"shown as reference line.",
        fontsize=11
    )
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig("outputs/powell_river_cost_breakdown.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: powell_river_cost_breakdown.png")


def plot_cost_per_night():
    driftwood_per_bed_night = DRIFTWOOD_FIXED_MONTHLY / 30 / DRIFTWOOD_CAPACITY
    warming_per_bed_night   = WARMING_CENTRE_FIXED_MONTHLY / 30 / WARMING_CENTRE_CAPACITY

    labels  = [
        f"ICU Bed\n($7,000/night)",
        f"ICU Bed\n($10,000/night)",
        f"Non-ICU Bed\n(${COST_NON_ICU_BED_DAY:,}/night)",
        f"Driftwood Shelter\nBed (${driftwood_per_bed_night:,.0f}/night)",
        f"Warming Centre\nBed (${warming_per_bed_night:,.0f}/night)",
    ]
    values  = [
        COST_ICU_BED_DAY_LOW,
        COST_ICU_BED_DAY_HIGH,
        COST_NON_ICU_BED_DAY,
        driftwood_per_bed_night,
        warming_per_bed_night,
    ]
    colours = ["#c0392b", "#e74c3c", "#e67e22", "#3498db", "#27ae60"]
    x       = np.arange(len(labels))
    w       = 0.55

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(x, values, width=w, color=colours, alpha=0.88, zorder=3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 80,
                f"${val:,.0f}", ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    y_bracket = COST_ICU_BED_DAY_HIGH * 1.08
    ax.annotate("", xy=(x[1], y_bracket), xytext=(x[0], y_bracket),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text((x[0] + x[1]) / 2, y_bracket + 150,
            "ICU range", ha='center', fontsize=8.5, color='black')

    ratio_icu_low_vs_shelter  = COST_ICU_BED_DAY_LOW  / driftwood_per_bed_night
    ratio_icu_high_vs_shelter = COST_ICU_BED_DAY_HIGH / driftwood_per_bed_night
    ratio_non_icu_vs_shelter  = COST_NON_ICU_BED_DAY  / driftwood_per_bed_night
    ax.text(0.98, 0.97,
            f"ICU is {ratio_icu_low_vs_shelter:.0f}–{ratio_icu_high_vs_shelter:.0f}× "
            f"more costly than Driftwood shelter\n"
            f"Non-ICU is {ratio_non_icu_vs_shelter:.0f}× more costly than Driftwood shelter\n"
            f"Hospital total operating cost: ${HOSPITAL_ANNUAL_OPERATING_COST/1e6:.0f}M/yr",
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            bbox=dict(boxstyle='round', facecolor='#fffbcc', alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("CAD per bed-night")
    ax.set_ylim(0, COST_ICU_BED_DAY_HIGH * 1.30)
    ax.set_title(
        "Cost per Bed-Night by Care Setting — qathet General Hospital & Powell River Shelter\n"
        "(3 ICU beds + 39 non-ICU beds; Driftwood $1.6M/yr ÷ 40 beds ÷ 365 days; Hospital base: $101M/yr)",
        fontsize=10
    )
    ax.grid(axis='y', alpha=0.25, zorder=0)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#c0392b", alpha=0.88, label="ICU — low estimate ($7,000)"),
        Patch(facecolor="#e74c3c", alpha=0.88, label="ICU — high estimate ($10,000)"),
        Patch(facecolor="#e67e22", alpha=0.88, label=f"Non-ICU (${COST_NON_ICU_BED_DAY:,})"),
        Patch(facecolor="#3498db", alpha=0.88, label="Driftwood shelter bed"),
        Patch(facecolor="#27ae60", alpha=0.88, label="Warming centre bed"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc='upper left')

    plt.tight_layout()
    plt.savefig("outputs/powell_river_cost_per_night.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: powell_river_cost_per_night.png")


def plot_demographics(scenarios: dict):
    admitted = [ag for ag in scenarios["Baseline\n(no shelter)"]["agents"]
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

    fig.suptitle("Demographics — Admitted Homeless Individuals, Powell River ABM (Baseline)",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/powell_river_demographics.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: powell_river_demographics.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Powell River Homeless ED Utilization ABM")
    print(f"  Hospital capacity : {HOSPITAL_CAPACITY} beds (qathet General)")
    print(f"    ICU beds        : {HOSPITAL_ICU_BEDS} @ ${COST_ICU_BED_DAY_LOW:,}–${COST_ICU_BED_DAY_HIGH:,}/day")
    print(f"    Non-ICU beds    : {HOSPITAL_NON_ICU_BEDS} @ ${COST_NON_ICU_BED_DAY:,}/day")
    print(f"    Blended cost    : ${COST_HOSPITAL_BED_DAY:,}/day (used in simulation)")
    print(f"  Hospital base ops : ${HOSPITAL_ANNUAL_OPERATING_COST/1e6:.0f}M/yr → ${HOSPITAL_BASE_MONTHLY:,}/month (fixed)")
    print(f"  Driftwood shelter : $1.6M/yr → ${DRIFTWOOD_FIXED_MONTHLY:,}/month")
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
