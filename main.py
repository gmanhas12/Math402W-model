"""
Agent-Based Model: Homeless Population & Emergency Department Utilization
=========================================================================
Extended model featuring:
  - Open population  : births (new homeless entries), deaths, spontaneous housing exits
  - Agent lifecycle  : HOMELESS → ADMITTED → POST_DISCHARGE → (re-admit or HOUSED/DECEASED)
  - Readmissions     : elevated re-admission probability in the post-discharge risk window
  - Seasonal effects : winter boosts admission probability and activates warming centres
  - Four scenarios   : Baseline | Shelter | Housing/Prevention | Combined
  - Cost ledger      : hospital beds vs shelter beds vs warming-centre beds
                       → net saving / cost of each intervention vs Baseline

Admission logic
---------------
  Arrivals  : n_arrivals  ~ Poisson(ARRIVAL_LAMBDA)          ED footfall per month
  Pool      : n_pool      ~ Poisson(ADMISSION_LAMBDA)        independent clinical draw
  Admissions: n_admitted  ~ Binomial(n_pool, p_eff)

  p_eff = ADMISSION_PROB  ← placeholder 10 %
  → swap with  Z59 / (NFA + Z59)  once real data arrives

Plug-in points are flagged with  # ← SWAP  throughout.

All distribution parameters live in the PARAMETERS block below; nowhere else.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


# =============================================================================
# PARAMETERS  ←  edit here; nowhere else
# =============================================================================

# ── Simulation length & calendar ─────────────────────────────────────────────
SIM_MONTHS          = 24          # total months to simulate
SIM_START_MONTH     = 1           # calendar month of sim month 1 (1 = January)
RNG_SEED            = 42

# ── Arrival process (ED footfall) ────────────────────────────────────────────
ARRIVAL_LAMBDA      = 11          # Poisson mean: new homeless arriving at ED per month

# ── Admission process (independent of arrivals) ──────────────────────────────
ADMISSION_LAMBDA    = 10          # Poisson mean: pool size n for Binomial draw
ADMISSION_PROB      = 0.10        # Binomial p  ← SWAP with Z59/(NFA+Z59) when ready

# ── Length of stay ────────────────────────────────────────────────────────────
LOS_MEAN_DAYS       = 15.4        # Normal mean: inpatient length of stay (days)
LOS_SD_DAYS         = 2.0         # Normal SD

# ── Starting population ───────────────────────────────────────────────────────
INITIAL_POPULATION  = 80          # Homeless individuals in system at t = 0

# ── Open-population dynamics ──────────────────────────────────────────────────
# New entries into homelessness each month (new births to the homeless pool)
NEW_HOMELESS_LAMBDA         = 8       # Poisson mean: new homeless individuals per month
# Monthly probability of death for a homeless individual (all states)
MONTHLY_MORTALITY_RATE      = 0.005   # ≈ 6 % annual; elevated vs general population  ← SWAP
# Monthly probability of spontaneously exiting homelessness (housed / left area)
MONTHLY_SPONTANEOUS_EXIT    = 0.015   # ≈ 18 % annual                                  ← SWAP

# ── Readmissions ──────────────────────────────────────────────────────────────
READMISSION_PROB            = 0.22    # probability of readmission per eligible month   ← SWAP
READMISSION_RISK_MONTHS     = 2       # months post-discharge during which risk applies

# ── Seasonal parameters ───────────────────────────────────────────────────────
WINTER_MONTHS               = {11, 12, 1, 2, 3}
SEASONAL_ADMISSION_BOOST    = 0.05    # extra admission probability in winter

# ── Demographics (adjust proportions once real data arrives) ──────────────────
GENDER_PROBS = {"Male": 0.70, "Female": 0.26, "Other": 0.04}
AGE_GROUPS   = ["18-29", "30-44", "45-59", "60+"]
AGE_PROBS    = [0.20,    0.35,    0.30,    0.15]

# ── Cost parameters ───────────────────────────────────────────────────────────
# Hospital
COST_HOSPITAL_BED_DAY           = 1_500    # CAD; acute-care bed-day in BC
# Shelter (year-round intervention)
COST_SHELTER_BED_NIGHT          = 75       # CAD; community shelter bed
SHELTER_CAPACITY_BEDS           = 30       # beds available in shelter intervention
SHELTER_FIXED_MONTHLY           = 15_000   # CAD; fixed monthly operating overhead
# Warming centres (winter-only)
COST_WARMING_CENTRE_BED_NIGHT   = 45       # CAD; overnight warming centre bed
WARMING_CENTRE_CAPACITY_BEDS    = 20       # beds available in warming centre intervention
WARMING_CENTRE_FIXED_MONTHLY    = 8_000    # CAD; fixed monthly overhead (winter only)

# ── Intervention: Shelter ─────────────────────────────────────────────────────
SHELTER_ARRIVAL_REDUCTION       = 0.25     # fraction of ED arrivals diverted
SHELTER_ADMISSION_REDUCTION     = 0.05     # absolute reduction in admission probability p
SHELTER_READMISSION_REDUCTION   = 0.08     # absolute reduction in readmission probability

# ── Intervention: Housing / Preventative care ─────────────────────────────────
HOUSING_LOS_REDUCTION_DAYS      = 3.0      # days removed from mean LOS
HOUSING_EXIT_BOOST              = 0.02     # extra monthly exit probability (case management)


# =============================================================================
# AGENT STATE MACHINE
# =============================================================================

class State(Enum):
    HOMELESS        = auto()   # in community, eligible for ED arrival & admission
    ADMITTED        = auto()   # currently occupying a hospital bed
    POST_DISCHARGE  = auto()   # recently discharged; at elevated readmission risk
    HOUSED          = auto()   # successfully housed; removed from homeless pool
    DECEASED        = auto()   # dead; removed from simulation


@dataclass
class Agent:
    agent_id:       int
    entry_month:    int                    # month they joined the simulation
    gender:         str   = "Male"
    age_group:      str   = "30-44"

    state:                  State         = State.HOMELESS
    admission_count:        int           = 0    # total admissions (incl. readmissions)
    current_admission_month: Optional[int] = None
    los_days:               float         = 0.0
    remaining_days:         float         = 0.0
    discharge_month:        Optional[int] = None
    months_post_discharge:  int           = 0    # counter for readmission risk window

    # Accumulators for the cost ledger
    total_hospital_bed_days: float        = 0.0

    # ── State transitions ──────────────────────────────────────────────────────

    def admit(self, month: int, los_days: float):
        """Transition → ADMITTED."""
        self.state                  = State.ADMITTED
        self.admission_count       += 1
        self.current_admission_month = month
        self.los_days               = los_days
        self.remaining_days         = los_days
        self.months_post_discharge  = 0    # reset risk window on readmission

    def discharge(self, month: int):
        """Transition → POST_DISCHARGE."""
        self.state                  = State.POST_DISCHARGE
        self.discharge_month        = month
        self.months_post_discharge  = 0

    def house(self):
        """Transition → HOUSED."""
        self.state = State.HOUSED

    def die(self):
        """Transition → DECEASED."""
        self.state = State.DECEASED

    # ── Per-month update ───────────────────────────────────────────────────────

    def update(self, current_month: int, days_in_month: int = 30) -> int:
        """
        Advance agent one month.
        Returns bed-days consumed this month (0 if not admitted).
        Discharge is flagged but NOT resolved here; caller checks state after.
        """
        bed_days = 0
        if self.state is State.ADMITTED:
            days_used                 = min(self.remaining_days, days_in_month)
            bed_days                  = days_used
            self.total_hospital_bed_days += days_used
            self.remaining_days      -= days_in_month
            if self.remaining_days <= 0:
                self.discharge(current_month)
        elif self.state is State.POST_DISCHARGE:
            self.months_post_discharge += 1
            if self.months_post_discharge > READMISSION_RISK_MONTHS:
                # Risk window expired; revert to general homeless pool
                self.state = State.HOMELESS
        return bed_days

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        return self.state not in (State.HOUSED, State.DECEASED)

    @property
    def in_bed(self) -> bool:
        return self.state is State.ADMITTED

    @property
    def eligible_new_admission(self) -> bool:
        """Can receive a first-time (or post-window) admission."""
        return self.state is State.HOMELESS

    @property
    def eligible_readmission(self) -> bool:
        """Within the post-discharge readmission risk window."""
        return self.state is State.POST_DISCHARGE


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def run_simulation(
    months:               int   = SIM_MONTHS,
    initial_agents:       int   = INITIAL_POPULATION,
    arrival_lambda:       float = ARRIVAL_LAMBDA,
    admission_lambda:     float = ADMISSION_LAMBDA,
    admission_prob:       float = ADMISSION_PROB,   # ← SWAP with Z59/(NFA+Z59)
    readmission_prob:     float = READMISSION_PROB,
    los_mean:             float = LOS_MEAN_DAYS,
    los_sd:               float = LOS_SD_DAYS,
    seasonal:             bool  = True,
    shelter_intervention: bool  = False,
    housing_intervention: bool  = False,
    seed:                 int   = RNG_SEED,
    start_month:          int   = SIM_START_MONTH,
) -> dict:
    """
    Simulate one scenario.

    Returns a dict of monthly time-series and aggregate counters.
    """
    rng          = np.random.default_rng(seed)
    gender_keys  = list(GENDER_PROBS.keys())
    gender_vals  = list(GENDER_PROBS.values())

    # Effective parameters modified by interventions
    eff_los_mean       = max(1.0, los_mean  - (HOUSING_LOS_REDUCTION_DAYS if housing_intervention else 0))
    eff_exit_base      = MONTHLY_SPONTANEOUS_EXIT + (HOUSING_EXIT_BOOST   if housing_intervention else 0)
    eff_adm_base       = admission_prob            - (SHELTER_ADMISSION_REDUCTION  if shelter_intervention else 0)
    eff_readm_base     = readmission_prob          - (SHELTER_READMISSION_REDUCTION if shelter_intervention else 0)
    eff_arr_lambda     = arrival_lambda * (1 - SHELTER_ARRIVAL_REDUCTION   if shelter_intervention else 1)

    # ── Initialise population ─────────────────────────────────────────────────
    agents        = []
    next_id       = 0

    def _new_agent(entry_month: int) -> Agent:
        nonlocal next_id
        ag = Agent(
            agent_id   = next_id,
            entry_month= entry_month,
            gender     = rng.choice(gender_keys, p=gender_vals),
            age_group  = rng.choice(AGE_GROUPS,  p=AGE_PROBS),
        )
        next_id += 1
        return ag

    for _ in range(initial_agents):
        agents.append(_new_agent(0))

    # ── Monthly accumulators ──────────────────────────────────────────────────
    monthly_arrivals      = []    # ED footfall (new arrivals)
    monthly_new_homeless  = []    # open-population new entries
    monthly_admissions    = []    # first-time admissions
    monthly_readmissions  = []    # readmissions
    monthly_discharges    = []    # discharges
    monthly_deaths        = []    # deaths
    monthly_exits         = []    # spontaneous housing exits
    monthly_occupancy     = []    # hospital beds occupied EOM
    monthly_pop_homeless  = []    # homeless individuals in community EOM
    monthly_bed_days      = []    # hospital bed-days consumed
    monthly_cost_hospital = []    # hospital bed costs (CAD)
    monthly_cost_shelter  = []    # shelter + warming-centre costs (CAD)
    monthly_cost_total    = []    # combined cost (CAD)

    for sim_month in range(1, months + 1):
        cal_month   = ((start_month - 1 + sim_month - 1) % 12) + 1   # 1–12
        is_winter   = cal_month in WINTER_MONTHS

        # ── Step 1: Update each agent; collect bed-days & discharges ──────────
        bed_days_this_month = 0.0
        discharges_this     = 0

        for ag in agents:
            if not ag.is_active:
                continue
            bd = ag.update(sim_month)
            bed_days_this_month += bd
            if ag.state is State.POST_DISCHARGE and ag.discharge_month == sim_month:
                discharges_this += 1

        # ── Step 2: Mortality ─────────────────────────────────────────────────
        deaths_this = 0
        for ag in agents:
            if ag.is_active and rng.random() < MONTHLY_MORTALITY_RATE:
                ag.die()
                deaths_this += 1

        # ── Step 3: Spontaneous housing exits ─────────────────────────────────
        exits_this = 0
        for ag in agents:
            if ag.state is State.HOMELESS and rng.random() < eff_exit_base:
                ag.house()
                exits_this += 1

        # ── Step 4: New homeless entries (open-population births) ─────────────
        n_new = rng.poisson(NEW_HOMELESS_LAMBDA)
        for _ in range(n_new):
            agents.append(_new_agent(sim_month))

        # ── Step 5: ED arrivals from existing homeless pool ───────────────────
        n_arrivals = rng.poisson(eff_arr_lambda)
        # (Arrivals represent ED footfall; they don't change state by themselves)

        # ── Step 6: Effective admission probability (seasonal, interventions) ──
        p_eff = max(0.0, eff_adm_base)
        if seasonal and is_winter:
            p_eff = min(1.0, p_eff + SEASONAL_ADMISSION_BOOST)

        # ── Step 7: First-time admissions (Poisson pool → Binomial) ──────────
        eligible_new   = [ag for ag in agents if ag.eligible_new_admission]
        n_pool         = rng.poisson(admission_lambda)
        n_admit_draw   = rng.binomial(n_pool, p_eff)
        n_admit        = min(n_admit_draw, len(eligible_new))
        if n_admit > 0:
            chosen = rng.choice(len(eligible_new), size=n_admit, replace=False)
            for idx in chosen:
                los = max(1.0, rng.normal(eff_los_mean, los_sd))
                eligible_new[idx].admit(sim_month, los)

        # ── Step 8: Readmissions ──────────────────────────────────────────────
        eligible_readm = [ag for ag in agents if ag.eligible_readmission]
        n_readmit      = 0
        p_readm        = max(0.0, eff_readm_base)
        if seasonal and is_winter:
            p_readm = min(1.0, p_readm + SEASONAL_ADMISSION_BOOST)
        for ag in eligible_readm:
            if rng.random() < p_readm:
                los = max(1.0, rng.normal(eff_los_mean, los_sd))
                ag.admit(sim_month, los)
                n_readmit += 1

        # ── Step 9: Shelter / warming-centre cost ─────────────────────────────
        shelter_cost_this = 0.0
        if shelter_intervention:
            # Year-round shelter
            shelter_beds_used   = min(
                len([ag for ag in agents if ag.state is State.HOMELESS]),
                SHELTER_CAPACITY_BEDS,
            )
            shelter_cost_this  += shelter_beds_used * COST_SHELTER_BED_NIGHT * 30
            shelter_cost_this  += SHELTER_FIXED_MONTHLY
            # Warming centre in winter
            if is_winter:
                wc_beds_used    = min(
                    len([ag for ag in agents if ag.state is State.HOMELESS]),
                    WARMING_CENTRE_CAPACITY_BEDS,
                )
                shelter_cost_this  += wc_beds_used * COST_WARMING_CENTRE_BED_NIGHT * 30
                shelter_cost_this  += WARMING_CENTRE_FIXED_MONTHLY

        # ── Step 10: Collect monthly tallies ──────────────────────────────────
        occupancy        = sum(1 for ag in agents if ag.in_bed)
        pop_homeless     = sum(1 for ag in agents if ag.state is State.HOMELESS)
        hospital_cost    = bed_days_this_month * COST_HOSPITAL_BED_DAY
        total_cost       = hospital_cost + shelter_cost_this

        monthly_arrivals.append(n_arrivals)
        monthly_new_homeless.append(n_new)
        monthly_admissions.append(n_admit)
        monthly_readmissions.append(n_readmit)
        monthly_discharges.append(discharges_this)
        monthly_deaths.append(deaths_this)
        monthly_exits.append(exits_this)
        monthly_occupancy.append(occupancy)
        monthly_pop_homeless.append(pop_homeless)
        monthly_bed_days.append(bed_days_this_month)
        monthly_cost_hospital.append(hospital_cost)
        monthly_cost_shelter.append(shelter_cost_this)
        monthly_cost_total.append(total_cost)

    return {
        # Time-series
        "monthly_arrivals":       monthly_arrivals,
        "monthly_new_homeless":   monthly_new_homeless,
        "monthly_admissions":     monthly_admissions,
        "monthly_readmissions":   monthly_readmissions,
        "monthly_discharges":     monthly_discharges,
        "monthly_deaths":         monthly_deaths,
        "monthly_exits":          monthly_exits,
        "monthly_occupancy":      monthly_occupancy,
        "monthly_pop_homeless":   monthly_pop_homeless,
        "monthly_bed_days":       monthly_bed_days,
        "monthly_cost_hospital":  monthly_cost_hospital,
        "monthly_cost_shelter":   monthly_cost_shelter,
        "monthly_cost_total":     monthly_cost_total,
        # Agents (for demographics)
        "agents":                 agents,
    }


# =============================================================================
# SCENARIO RUNNER
# =============================================================================

def run_all_scenarios(months: int = SIM_MONTHS, seed: int = RNG_SEED) -> dict:
    """Run all four policy scenarios with shared parameters."""
    return {
        "Baseline":           run_simulation(months=months, seed=seed,
                                             shelter_intervention=False,
                                             housing_intervention=False),
        "Shelter":            run_simulation(months=months, seed=seed,
                                             shelter_intervention=True,
                                             housing_intervention=False),
        "Housing/Prevention": run_simulation(months=months, seed=seed,
                                             shelter_intervention=False,
                                             housing_intervention=True),
        "Combined":           run_simulation(months=months, seed=seed,
                                             shelter_intervention=True,
                                             housing_intervention=True),
    }


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary(scenarios: dict):
    baseline_cost = sum(scenarios["Baseline"]["monthly_cost_total"])
    print()
    print("=" * 100)
    print(f"{'Scenario':<22} {'Arrivals':>9} {'Admits':>7} {'Re-admits':>10} "
          f"{'Bed-Days':>10} {'Deaths':>7} {'Exits':>6} "
          f"{'Hosp Cost (M)':>14} {'Shelt Cost (M)':>15} {'Total Cost (M)':>15} {'vs Baseline':>13}")
    print("=" * 100)
    for name, res in scenarios.items():
        arrivals   = sum(res["monthly_arrivals"])
        admits     = sum(res["monthly_admissions"])
        readmits   = sum(res["monthly_readmissions"])
        bed_days   = sum(res["monthly_bed_days"])
        deaths     = sum(res["monthly_deaths"])
        exits      = sum(res["monthly_exits"])
        h_cost     = sum(res["monthly_cost_hospital"])
        s_cost     = sum(res["monthly_cost_shelter"])
        t_cost     = sum(res["monthly_cost_total"])
        delta      = baseline_cost - t_cost
        delta_str  = f"-${delta/1e6:.3f}M" if delta >= 0 else f"+${abs(delta)/1e6:.3f}M"
        print(
            f"{name:<22} {arrivals:>9} {admits:>7} {readmits:>10} "
            f"{bed_days:>10.0f} {deaths:>7} {exits:>6} "
            f"{h_cost/1e6:>14.3f} {s_cost/1e6:>15.3f} {t_cost/1e6:>15.3f} {delta_str:>13}"
        )
    print("=" * 100)
    print()


# =============================================================================
# PLOTTING
# =============================================================================

COLORS = {
    "Baseline":           "#e74c3c",
    "Shelter":            "#3498db",
    "Housing/Prevention": "#2ecc71",
    "Combined":           "#9b59b6",
}

def plot_scenarios(scenarios: dict, months: int = SIM_MONTHS):
    """6-panel figure: flow metrics + cumulative cost breakdown."""
    month_axis = np.arange(1, months + 1)

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    ax_arr  = fig.add_subplot(gs[0, 0])   # arrivals
    ax_adm  = fig.add_subplot(gs[0, 1])   # admissions + readmissions
    ax_occ  = fig.add_subplot(gs[1, 0])   # hospital occupancy
    ax_pop  = fig.add_subplot(gs[1, 1])   # homeless community population
    ax_cost = fig.add_subplot(gs[2, :])   # cumulative total cost
    ax_cmp  = fig.add_subplot(gs[3, :])   # cost component comparison (bars, last month cumulative)

    for name, res in scenarios.items():
        c   = COLORS[name]
        adm = np.array(res["monthly_admissions"])
        rdm = np.array(res["monthly_readmissions"])

        ax_arr.plot(month_axis, res["monthly_arrivals"],      color=c, marker='o', ms=4, label=name)
        ax_adm.plot(month_axis, adm + rdm,                    color=c, marker='s', ms=4, label=name)
        ax_occ.plot(month_axis, res["monthly_occupancy"],     color=c, marker='d', ms=4, label=name)
        ax_pop.plot(month_axis, res["monthly_pop_homeless"],  color=c, marker='^', ms=4, label=name)
        ax_cost.plot(month_axis,
                     np.cumsum(res["monthly_cost_total"]) / 1e6,
                     color=c, lw=2.5, label=name)

    for ax, title, ylabel in [
        (ax_arr, "Monthly ED Arrivals (existing homeless pool)",   "# Arrivals"),
        (ax_adm, "Monthly Admissions + Readmissions",             "# Admitted"),
        (ax_occ, "Hospital Occupancy — End of Month",             "# Beds Occupied"),
        (ax_pop, "Homeless Community Population — End of Month",  "# Individuals"),
    ]:
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Simulation Month")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    ax_cost.set_title("Cumulative Total Cost (Hospital + Shelter/Warming Centre)", fontsize=10)
    ax_cost.set_xlabel("Simulation Month")
    ax_cost.set_ylabel("Cumulative Cost (M CAD)")
    ax_cost.legend()
    ax_cost.grid(alpha=0.3)

    # ── Stacked-bar: cumulative cost breakdown per scenario ───────────────────
    names     = list(scenarios.keys())
    hosp_cum  = [sum(s["monthly_cost_hospital"]) / 1e6 for s in scenarios.values()]
    shelt_cum = [sum(s["monthly_cost_shelter"])  / 1e6 for s in scenarios.values()]
    x         = np.arange(len(names))
    w         = 0.45

    bars_h = ax_cmp.bar(x, hosp_cum, w, label="Hospital bed costs", color="#e74c3c", alpha=0.85)
    bars_s = ax_cmp.bar(x, shelt_cum, w, bottom=hosp_cum,
                        label="Shelter / Warming-centre costs", color="#3498db", alpha=0.85)

    # Annotate total
    for i, (h, s) in enumerate(zip(hosp_cum, shelt_cum)):
        ax_cmp.text(i, h + s + 0.02, f"${h+s:.2f}M", ha='center', va='bottom', fontsize=8)

    ax_cmp.set_title(f"Cumulative Cost Breakdown Over {months} Months", fontsize=10)
    ax_cmp.set_xticks(x)
    ax_cmp.set_xticklabels(names)
    ax_cmp.set_ylabel("Cumulative Cost (M CAD)")
    ax_cmp.legend()
    ax_cmp.grid(axis='y', alpha=0.3)

    fig.suptitle(
        "Homeless ED Utilization ABM — Scenario Comparison\n"
        "(open population · readmissions · seasonal effects · cost ledger)",
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/scenario_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_population_flows(scenarios: dict, months: int = SIM_MONTHS):
    """Population-flow panel: new entries, deaths, exits, readmissions."""
    month_axis = np.arange(1, months + 1)
    fig, axes  = plt.subplots(2, 2, figsize=(14, 8))
    axes       = axes.flatten()

    series = [
        ("monthly_new_homeless",  "New Homeless Entries / Month",    "# Individuals"),
        ("monthly_deaths",        "Deaths / Month",                  "# Deaths"),
        ("monthly_exits",         "Spontaneous Housing Exits / Month","# Exits"),
        ("monthly_readmissions",  "Readmissions / Month",            "# Readmissions"),
    ]

    for ax, (key, title, ylabel) in zip(axes, series):
        for name, res in scenarios.items():
            ax.plot(month_axis, res[key], color=COLORS[name], marker='o', ms=3, label=name)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Simulation Month")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.suptitle("Open-Population Dynamics — All Scenarios", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/population_flows.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_demographics(scenarios: dict):
    """Gender & age breakdown of all agents ever admitted (Baseline only)."""
    admitted = [ag for ag in scenarios["Baseline"]["agents"] if ag.admission_count > 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Gender
    gc = Counter(ag.gender for ag in admitted)
    axes[0].bar(gc.keys(), gc.values(), color=["#3498db", "#e74c3c", "#95a5a6"])
    axes[0].set_title("Admitted Agents by Gender (Baseline)")
    axes[0].set_ylabel("Count")

    # Age group
    ac    = Counter(ag.age_group for ag in admitted)
    order = ["18-29", "30-44", "45-59", "60+"]
    axes[1].bar(order, [ac.get(g, 0) for g in order], color="#2ecc71")
    axes[1].set_title("Admitted Agents by Age Group (Baseline)")
    axes[1].set_ylabel("Count")

    # Readmission count distribution
    rc = Counter(ag.admission_count for ag in admitted)
    max_adm = max(rc.keys(), default=1)
    axes[2].bar(range(1, max_adm + 1),
                [rc.get(i, 0) for i in range(1, max_adm + 1)],
                color="#9b59b6")
    axes[2].set_title("Admission Frequency per Agent (Baseline)")
    axes[2].set_xlabel("Total Admissions per Individual")
    axes[2].set_ylabel("Count")
    axes[2].set_xticks(range(1, max_adm + 1))

    fig.suptitle("Demographics of Admitted Homeless Individuals — Baseline", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/demographics.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_cost_per_day_comparison():
    """
    Simple reference bar chart: cost per night/day for each care setting.
    Helps justify the intervention investment vs hospital bed costs.
    """
    settings = [
        "Hospital\nAcute Bed",
        "Community\nShelter Bed",
        "Warming\nCentre Bed",
    ]
    costs = [
        COST_HOSPITAL_BED_DAY,
        COST_SHELTER_BED_NIGHT,
        COST_WARMING_CENTRE_BED_NIGHT,
    ]
    colours = ["#e74c3c", "#3498db", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars    = ax.bar(settings, costs, color=colours, width=0.5)

    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 10,
                f"${cost:,}/night", ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel("Cost (CAD per night/bed-day)")
    ax.set_title("Cost per Bed-Night by Care Setting\n(BC ballpark figures)", fontsize=11)
    ax.set_ylim(0, max(costs) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(COST_HOSPITAL_BED_DAY, color='grey', ls='--', lw=1, alpha=0.5)

    ratio_shelter = COST_HOSPITAL_BED_DAY / COST_SHELTER_BED_NIGHT
    ratio_wc      = COST_HOSPITAL_BED_DAY / COST_WARMING_CENTRE_BED_NIGHT
    ax.text(0.98, 0.97,
            f"Hospital is {ratio_shelter:.0f}× more costly than shelter\n"
            f"Hospital is {ratio_wc:.0f}× more costly than warming centre",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/cost_per_day_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"Running ABM — {SIM_MONTHS} months, seed={RNG_SEED}")
    print(f"  Admission probability (placeholder): {ADMISSION_PROB:.0%}  ← SWAP with Z59/(NFA+Z59)")
    print(f"  Readmission probability            : {READMISSION_PROB:.0%}")
    print(f"  Monthly mortality rate             : {MONTHLY_MORTALITY_RATE:.1%}")
    print()

    scenarios = run_all_scenarios(months=SIM_MONTHS, seed=RNG_SEED)

    print_summary(scenarios)
    plot_cost_per_day_comparison()
    plot_scenarios(scenarios, months=SIM_MONTHS)
    plot_population_flows(scenarios, months=SIM_MONTHS)
    plot_demographics(scenarios)
