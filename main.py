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
  Population   : 80 homeless individuals (2023 PiT count)

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
SIM_MONTHS          = 24          # months to simulate
SIM_START_MONTH     = 4           # April (month 1 of sim = April 2026)
RNG_SEED            = 42

# ── Hospital ──────────────────────────────────────────────────────────────────
HOSPITAL_CAPACITY   = 42          # qathet General Hospital acute beds

# ── Arrival process ───────────────────────────────────────────────────────────
ARRIVAL_LAMBDA      = 11          # Poisson mean: homeless individuals arriving at ED/month
                                  # ← SWAP with real ED visit counts once data arrives

# ── Admission process ─────────────────────────────────────────────────────────
ADMISSION_LAMBDA    = 10          # Poisson pool size n for Binomial draw
ADMISSION_PROB      = 0.10        # Binomial p  ← SWAP with Z59/(NFA+Z59) from real data

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
# ← SWAP with real age/gender from qathet hospital data
GENDER_PROBS = {"Male": 0.70, "Female": 0.26, "Other": 0.04}
AGE_GROUPS   = ["18-29", "30-44", "45-59", "60+"]
AGE_PROBS    = [0.20,    0.35,    0.30,    0.15]

# ── Cost parameters ───────────────────────────────────────────────────────────
COST_HOSPITAL_BED_DAY           = 1_500   # CAD; BC acute-care bed-day (CIHI estimate)

# Driftwood / year-round shelter (based on Sechelt RainCity ~$900k/yr for 35 beds)
# Scaled to 40 beds → ~$1.03M/yr → ~$85,700/month total
DRIFTWOOD_CAPACITY              = 40      # beds
DRIFTWOOD_FIXED_MONTHLY         = 85_700  # CAD; total operating cost/month
                                          # ← SWAP with actual BC Housing contract value

# Additional 40-bed shelter (same cost structure as Driftwood)
ADDITIONAL_SHELTER_CAPACITY     = 40
ADDITIONAL_SHELTER_FIXED_MONTHLY = 85_700

# Warming centre (Oct–Apr; 20 beds; lower cost than year-round shelter)
WARMING_CENTRE_CAPACITY         = 20
WARMING_CENTRE_MONTHS           = {10, 11, 12, 1, 2, 3, 4}   # Oct–Apr
WARMING_CENTRE_FIXED_MONTHLY    = 40_000  # CAD/month while open
                                          # ← SWAP with actual contract value

# ── Shelter diversion effects ─────────────────────────────────────────────────
# How much each shelter type reduces ED pressure
# These are conservative estimates; ← SWAP with literature values or calibrated data

# Year-round shelter (Driftwood)
SHELTER_ARRIVAL_REDUCTION       = 0.25   # 25% of ED arrivals diverted
SHELTER_ADMISSION_REDUCTION     = 0.05   # absolute drop in admission probability
SHELTER_READMISSION_REDUCTION   = 0.08   # absolute drop in readmission probability
SHELTER_EXIT_BOOST              = 0.01   # extra monthly housing exit probability

# Second 40-bed shelter (additive on top of Driftwood)
EXTRA_SHELTER_ARRIVAL_REDUCTION     = 0.10   # additional 10% diversion
EXTRA_SHELTER_ADMISSION_REDUCTION   = 0.02
EXTRA_SHELTER_READMISSION_REDUCTION = 0.03

# Warming centre (winter-only, smaller effect)
WARMING_ARRIVAL_REDUCTION       = 0.08   # 8% additional winter diversion
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

    def house(self):  self.state = State.HOUSED
    def die(self):    self.state = State.DECEASED

    def update(self, current_month: int, days_in_month: int = 30) -> float:
        """Advance one month. Returns bed-days consumed."""
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
    # Scenario flags
    driftwood_open:       bool  = False,   # 40-bed year-round shelter from month 1
    extra_shelter:        bool  = False,   # additional 40-bed shelter
    warming_centre:       bool  = False,   # 20-bed warming centre Oct–Apr
    seed:                 int   = RNG_SEED,
    start_month:          int   = SIM_START_MONTH,
) -> dict:

    rng         = np.random.default_rng(seed)
    gender_keys = list(GENDER_PROBS.keys())
    gender_vals = list(GENDER_PROBS.values())

    # ── Effective parameters ──────────────────────────────────────────────────
    eff_exit_base  = MONTHLY_SPONTANEOUS_EXIT
    eff_adm_base   = admission_prob
    eff_readm_base = readmission_prob
    eff_arr_lambda = arrival_lambda

    if driftwood_open:
        eff_arr_lambda  *= (1 - SHELTER_ARRIVAL_REDUCTION)
        eff_adm_base    -= SHELTER_ADMISSION_REDUCTION
        eff_readm_base  -= SHELTER_READMISSION_REDUCTION
        eff_exit_base   += SHELTER_EXIT_BOOST

    if extra_shelter:
        eff_arr_lambda  *= (1 - EXTRA_SHELTER_ARRIVAL_REDUCTION)
        eff_adm_base    -= EXTRA_SHELTER_ADMISSION_REDUCTION
        eff_readm_base  -= EXTRA_SHELTER_READMISSION_REDUCTION

    eff_arr_lambda  = max(0.0, eff_arr_lambda)
    eff_adm_base    = max(0.0, eff_adm_base)
    eff_readm_base  = max(0.0, eff_readm_base)

    # ── Initialise population ─────────────────────────────────────────────────
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

    # ── Monthly output lists ──────────────────────────────────────────────────
    monthly_arrivals       = []
    monthly_new_homeless   = []
    monthly_admissions     = []
    monthly_readmissions   = []
    monthly_overflow       = []    # admissions that exceeded 42-bed capacity
    monthly_discharges     = []
    monthly_deaths         = []
    monthly_exits          = []
    monthly_occupancy      = []
    monthly_pop_homeless   = []
    monthly_bed_days       = []
    monthly_cost_hospital  = []
    monthly_cost_shelter   = []
    monthly_cost_total     = []

    for sim_month in range(1, months + 1):
        cal_month = ((start_month - 1 + sim_month - 1) % 12) + 1   # 1–12
        is_winter = cal_month in WINTER_MONTHS
        is_warming_season = cal_month in WARMING_CENTRE_MONTHS

        # ── 1. Advance all agents ─────────────────────────────────────────────
        bed_days_this = 0.0
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

        # ── 3. Spontaneous exits ──────────────────────────────────────────────
        exits_this = 0
        for ag in agents:
            if ag.state is State.HOMELESS and rng.random() < eff_exit_base:
                ag.house()
                exits_this += 1

        # ── 4. New homeless entries ───────────────────────────────────────────
        n_new = rng.poisson(NEW_HOMELESS_LAMBDA)
        for _ in range(n_new):
            agents.append(_new_agent(sim_month))

        # ── 5. ED arrivals ────────────────────────────────────────────────────
        # Warming centre provides additional winter diversion on top of shelter effects
        arr_lambda_this = eff_arr_lambda
        if warming_centre and is_warming_season:
            arr_lambda_this *= (1 - WARMING_ARRIVAL_REDUCTION)
        arr_lambda_this = max(0.0, arr_lambda_this)
        n_arrivals = rng.poisson(arr_lambda_this)

        # ── 6. Effective admission probability ────────────────────────────────
        p_adm = eff_adm_base
        p_rdm = eff_readm_base
        if warming_centre and is_warming_season:
            p_adm = max(0.0, p_adm - WARMING_ADMISSION_REDUCTION)
            p_rdm = max(0.0, p_rdm - WARMING_READMISSION_REDUCTION)
        if seasonal and is_winter:
            p_adm = min(1.0, p_adm + SEASONAL_ADMISSION_BOOST)
            p_rdm = min(1.0, p_rdm + SEASONAL_ADMISSION_BOOST)

        # ── 7. First-time admissions ──────────────────────────────────────────
        current_occupancy  = sum(1 for ag in agents if ag.in_bed)
        eligible_new       = [ag for ag in agents if ag.eligible_new_admission]
        n_pool             = rng.poisson(admission_lambda)
        n_admit_draw       = rng.binomial(n_pool, p_adm)
        n_admit            = min(n_admit_draw, len(eligible_new))
        n_overflow_this    = 0

        if n_admit > 0:
            chosen = rng.choice(len(eligible_new), size=n_admit, replace=False)
            for idx in chosen:
                los = max(1.0, rng.normal(los_mean, los_sd))
                eligible_new[idx].admit(sim_month, los)
                current_occupancy += 1
                if current_occupancy > HOSPITAL_CAPACITY:
                    n_overflow_this += 1   # flag: over 42-bed capacity

        # ── 8. Readmissions ───────────────────────────────────────────────────
        eligible_readm = [ag for ag in agents if ag.eligible_readmission]
        n_readmit = 0
        for ag in eligible_readm:
            if rng.random() < p_rdm:
                los = max(1.0, rng.normal(los_mean, los_sd))
                ag.admit(sim_month, los)
                n_readmit      += 1
                current_occupancy += 1
                if current_occupancy > HOSPITAL_CAPACITY:
                    n_overflow_this += 1

        # ── 9. Shelter / warming-centre operating costs ───────────────────────
        shelter_cost_this = 0.0

        if driftwood_open:
            shelter_cost_this += DRIFTWOOD_FIXED_MONTHLY

        if extra_shelter:
            shelter_cost_this += ADDITIONAL_SHELTER_FIXED_MONTHLY

        if warming_centre and is_warming_season:
            shelter_cost_this += WARMING_CENTRE_FIXED_MONTHLY

        # ── 10. Collect tallies ───────────────────────────────────────────────
        occupancy       = sum(1 for ag in agents if ag.in_bed)
        pop_homeless    = sum(1 for ag in agents if ag.state is State.HOMELESS)
        hospital_cost   = bed_days_this * COST_HOSPITAL_BED_DAY
        total_cost      = hospital_cost + shelter_cost_this

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
        monthly_cost_hospital.append(hospital_cost)
        monthly_cost_shelter.append(shelter_cost_this)
        monthly_cost_total.append(total_cost)

    return dict(
        monthly_arrivals      = monthly_arrivals,
        monthly_new_homeless  = monthly_new_homeless,
        monthly_admissions    = monthly_admissions,
        monthly_readmissions  = monthly_readmissions,
        monthly_overflow      = monthly_overflow,
        monthly_discharges    = monthly_discharges,
        monthly_deaths        = monthly_deaths,
        monthly_exits         = monthly_exits,
        monthly_occupancy     = monthly_occupancy,
        monthly_pop_homeless  = monthly_pop_homeless,
        monthly_bed_days      = monthly_bed_days,
        monthly_cost_hospital = monthly_cost_hospital,
        monthly_cost_shelter  = monthly_cost_shelter,
        monthly_cost_total    = monthly_cost_total,
        agents                = agents,
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
    w = 110
    print("\n" + "=" * w)
    print(f"{'Scenario':<35} {'Admits':>7} {'Re-adm':>7} {'Overflow':>9} "
          f"{'Bed-Days':>10} {'Deaths':>7} "
          f"{'Hosp $M':>9} {'Shelter $M':>11} {'Total $M':>10} {'vs Baseline':>13}")
    print("=" * w)
    for name, res in scenarios.items():
        label    = name.replace("\n", " ")
        admits   = sum(res["monthly_admissions"])
        readmits = sum(res["monthly_readmissions"])
        overflow = sum(res["monthly_overflow"])
        bed_days = sum(res["monthly_bed_days"])
        deaths   = sum(res["monthly_deaths"])
        h_cost   = sum(res["monthly_cost_hospital"])
        s_cost   = sum(res["monthly_cost_shelter"])
        t_cost   = sum(res["monthly_cost_total"])
        delta    = baseline_cost - t_cost
        dstr     = f"-${delta/1e6:.3f}M" if delta >= 0 else f"+${abs(delta)/1e6:.3f}M"
        print(f"{label:<35} {admits:>7} {readmits:>7} {overflow:>9} "
              f"{bed_days:>10.0f} {deaths:>7} "
              f"{h_cost/1e6:>9.3f} {s_cost/1e6:>11.3f} {t_cost/1e6:>10.3f} {dstr:>13}")
    print("=" * w + "\n")


# =============================================================================
# PLOTTING
# =============================================================================

def _month_labels(months: int, start_month: int = SIM_START_MONTH) -> list[str]:
    """Generate 'Apr 26', 'May 26' … labels for the x-axis."""
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    labels = []
    for i in range(months):
        cal = ((start_month - 1 + i) % 12)
        yr  = 26 + ((start_month - 1 + i) // 12)
        labels.append(f"{month_names[cal]} '{yr}")
    return labels


def plot_main(scenarios: dict, months: int = SIM_MONTHS):
    """4-panel summary: occupancy, admissions, overflow, cumulative cost."""
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
        # Hospital capacity line on occupancy chart
        if key == "monthly_occupancy":
            ax.axhline(HOSPITAL_CAPACITY, color='black', ls='--', lw=1.2,
                       label=f"Capacity ({HOSPITAL_CAPACITY} beds)")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x[::tick_every])
        ax.set_xticklabels(labels[::tick_every], rotation=35, ha='right', fontsize=7)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)

    # ── Cumulative cost ───────────────────────────────────────────────────────
    ax4 = axes[3]
    for name, res in scenarios.items():
        ax4.plot(x, np.cumsum(res["monthly_cost_total"]) / 1e6,
                 color=COLORS[name], lw=2.5, label=name.replace("\n", " "))
    ax4.set_title("Cumulative Total Cost (Hospital + Shelter)", fontsize=10)
    ax4.set_ylabel("Cumulative Cost (M CAD)")
    ax4.set_xticks(x[::tick_every])
    ax4.set_xticklabels(labels[::tick_every], rotation=35, ha='right', fontsize=7)
    ax4.legend(fontsize=7)
    ax4.grid(alpha=0.25)

    fig.suptitle(
        "qathet General Hospital — Homeless ED Utilization ABM\n"
        "Powell River, BC  |  Simulation starts April 2026",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("outputs/powell_river_main.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: powell_river_main.png")


def plot_cost_breakdown(scenarios: dict, months: int = SIM_MONTHS):
    """Stacked bar: cumulative hospital vs shelter cost per scenario."""
    names     = [n.replace("\n", " ") for n in scenarios]
    hosp_cum  = [sum(s["monthly_cost_hospital"]) / 1e6 for s in scenarios.values()]
    shlt_cum  = [sum(s["monthly_cost_shelter"])  / 1e6 for s in scenarios.values()]
    x         = np.arange(len(names))
    w         = 0.5

    fig, ax = plt.subplots(figsize=(11, 5))
    bars_h  = ax.bar(x, hosp_cum, w, label="Hospital bed-day costs", color="#e74c3c", alpha=0.88)
    bars_s  = ax.bar(x, shlt_cum, w, bottom=hosp_cum,
                     label="Shelter / Warming-centre operating costs", color="#3498db", alpha=0.88)

    baseline_total = hosp_cum[0] + shlt_cum[0]
    for i, (h, s) in enumerate(zip(hosp_cum, shlt_cum)):
        total   = h + s
        saving  = baseline_total - total
        label   = f"${total:.2f}M"
        if saving > 0:
            label += f"\n(saves ${saving:.2f}M)"
        ax.text(i, total + 0.02, label, ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.axhline(baseline_total, color='grey', ls='--', lw=1, alpha=0.6, label="Baseline total")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Cumulative Cost (M CAD)")
    ax.set_title(f"Cumulative Cost Breakdown Over {months} Months — Powell River ABM", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig("outputs/powell_river_cost_breakdown.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: powell_river_cost_breakdown.png")


def plot_cost_per_night():
    """Reference chart: cost per night by care setting."""
    settings = ["Hospital\nAcute Bed", "Year-Round\nShelter Bed\n(Driftwood est.)",
                "Warming\nCentre Bed"]
    # Driftwood per-bed-night: $85,700/mo ÷ 30 days ÷ 40 beds ≈ $71/bed/night
    driftwood_per_bed_night = DRIFTWOOD_FIXED_MONTHLY / 30 / DRIFTWOOD_CAPACITY
    warming_per_bed_night   = WARMING_CENTRE_FIXED_MONTHLY / 30 / WARMING_CENTRE_CAPACITY
    costs   = [COST_HOSPITAL_BED_DAY, driftwood_per_bed_night, warming_per_bed_night]
    colours = ["#e74c3c", "#3498db", "#27ae60"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(settings, costs, color=colours, width=0.45)
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"${cost:,.0f}/night", ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ratio_s = COST_HOSPITAL_BED_DAY / driftwood_per_bed_night
    ratio_w = COST_HOSPITAL_BED_DAY / warming_per_bed_night
    ax.text(0.97, 0.95,
            f"Hospital is ~{ratio_s:.0f}× more costly than shelter\n"
            f"Hospital is ~{ratio_w:.0f}× more costly than warming centre",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#fffbcc', alpha=0.9))

    ax.set_ylim(0, max(costs) * 1.3)
    ax.set_ylabel("CAD per bed-night")
    ax.set_title("Cost per Bed-Night by Care Setting — Powell River", fontsize=11)
    ax.grid(axis='y', alpha=0.25)

    plt.tight_layout()
    plt.savefig("outputs/powell_river_cost_per_night.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: powell_river_cost_per_night.png")


def plot_demographics(scenarios: dict):
    """Gender, age, readmission frequency — Baseline only."""
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
    print(f"  Start             : April 2026")
    print(f"  Duration          : {SIM_MONTHS} months")
    print(f"  Initial population: {INITIAL_POPULATION}")
    print(f"  Admission prob    : {ADMISSION_PROB:.0%}  ← SWAP with Z59/(NFA+Z59)")
    print(f"  LOS mean          : {LOS_MEAN_DAYS} days  ← SWAP with real data")
    print("=" * 60 + "\n")

    scenarios = run_all_scenarios(months=SIM_MONTHS, seed=RNG_SEED)

    print_summary(scenarios)
    plot_cost_per_night()
    plot_main(scenarios, months=SIM_MONTHS)
    plot_cost_breakdown(scenarios, months=SIM_MONTHS)
    plot_demographics(scenarios)