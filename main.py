"""
Agent-Based Model: Homeless Population & Emergency Department Utilization


Extensions over baseline model:
  - Agent demographics: age, gender
  - Seasonal admission boost (winter afflictions)
  - Shelter intervention: reduces arrivals & admission probability
  - Housing intervention: reduces length-of-stay
  - Cost tracking: per-day bed cost, savings from interventions
  - Scenario comparison: baseline vs shelter vs housing vs combined

Data plug-in: replace the PARAMETERS section values once real data arrives.
All distribution parameters are clearly labelled and isolated.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# PARAMETERS  <-- swap these out when real data arrives
# =============================================================================

# Arrival & admission
ARRIVAL_LAMBDA      = 3        # Poisson mean: new homeless individuals arriving per month
ADMISSION_PROB      = 0.10     # Binomial p: probability an arrival is admitted
LOS_MEAN_DAYS       = 15.4     # Normal mean: length of stay (days)
LOS_SD_DAYS         = 2.0      # Normal SD: length of stay (days)
INITIAL_POPULATION  = 80       # Starting homeless population (not yet admitted)

# Seasonal parameters (winter = months 11,12,1,2,3)
WINTER_MONTHS       = {11, 12, 1, 2, 3}
SEASONAL_ADMISSION_BOOST = 0.05   # Additional admission probability in winter months

# Demographics (proportions, used for sampling)
# Source: adjust once real data available
GENDER_PROBS        = {"Male": 0.70, "Female": 0.26, "Other": 0.04}
AGE_GROUPS          = ["18-29", "30-44", "45-59", "60+"]
AGE_PROBS           = [0.20,    0.35,    0.30,    0.15]   # proportions

# Cost parameters
COST_PER_BED_DAY    = 1500     # CAD; typical acute care cost per bed-day in BC
SHELTER_COST_MONTHLY = 50000   # CAD; monthly cost to run an additional shelter

# Intervention parameters
# Shelter intervention: reduces monthly arrivals reaching ED
SHELTER_ARRIVAL_REDUCTION   = 0.25   # fraction of arrivals diverted by shelter
SHELTER_ADMISSION_REDUCTION = 0.05   # additional reduction in admission probability
# Housing/preventative care: reduces length of stay
HOUSING_LOS_REDUCTION_DAYS  = 3.0   # days shaved off average LOS


# =============================================================================
# AGENT
# =============================================================================

@dataclass
class Agent:
    agent_id: int
    arrival_month: int
    gender: str = "Male"
    age_group: str = "30-44"

    admitted: bool = False
    admission_month: Optional[int] = None
    length_of_stay_days: float = 0.0
    remaining_days: float = 0.0
    discharged: bool = False
    discharge_month: Optional[int] = None
    total_bed_days: float = 0.0   # accumulates for cost calculation

    def admit(self, month: int, los_days: float):
        self.admitted = True
        self.admission_month = month
        self.length_of_stay_days = los_days
        self.remaining_days = los_days

    def update_one_month(self, current_month: int, days_in_month: int = 30):
        if self.admitted and not self.discharged:
            days_this_month = min(self.remaining_days, days_in_month)
            self.total_bed_days += days_this_month
            self.remaining_days -= days_in_month
            if self.remaining_days <= 0:
                self.discharged = True
                self.discharge_month = current_month


# =============================================================================
# SIMULATION
# =============================================================================

def run_simulation(
    months: int = 24,
    initial_agents: int = INITIAL_POPULATION,
    arrival_lambda: float = ARRIVAL_LAMBDA,
    admission_prob: float = ADMISSION_PROB,
    los_mean: float = LOS_MEAN_DAYS,
    los_sd: float = LOS_SD_DAYS,
    seasonal: bool = True,
    shelter_intervention: bool = False,
    housing_intervention: bool = False,
    seed: int = 42,
    start_month: int = 1,          # calendar month of month-1 (1=Jan, used for seasons)
) -> dict:
    """
    Run one scenario of the ABM.

    Parameters
    ----------
    shelter_intervention : bool
        If True, applies SHELTER_ARRIVAL_REDUCTION and SHELTER_ADMISSION_REDUCTION.
    housing_intervention : bool
        If True, reduces LOS by HOUSING_LOS_REDUCTION_DAYS.
    seasonal : bool
        If True, applies SEASONAL_ADMISSION_BOOST in winter months.
    start_month : int
        The calendar month (1-12) corresponding to simulation month 1.
    """
    rng = np.random.default_rng(seed)

    agents = []
    next_agent_id = 0

    gender_keys   = list(GENDER_PROBS.keys())
    gender_vals   = list(GENDER_PROBS.values())

    # Effective LOS
    eff_los_mean = los_mean - (HOUSING_LOS_REDUCTION_DAYS if housing_intervention else 0)
    eff_los_mean = max(1.0, eff_los_mean)

    # Initial population
    for _ in range(initial_agents):
        g = rng.choice(gender_keys, p=gender_vals)
        a = rng.choice(AGE_GROUPS, p=AGE_PROBS)
        agents.append(Agent(next_agent_id, arrival_month=0, gender=g, age_group=a))
        next_agent_id += 1

    monthly_arrivals    = []
    monthly_admissions  = []
    monthly_discharges  = []
    monthly_occupancy   = []
    monthly_bed_days    = []
    monthly_cost        = []

    for sim_month in range(1, months + 1):
        cal_month = ((start_month - 1 + sim_month - 1) % 12) + 1  # 1-12

        # ---- 1. Update admitted agents ----
        discharges_this_month = 0
        bed_days_this_month   = 0.0
        for agent in agents:
            prev_remaining = agent.remaining_days
            was_in = agent.admitted and not agent.discharged
            agent.update_one_month(sim_month)
            if was_in:
                bed_days_this_month += min(prev_remaining, 30)
            if was_in and agent.discharged:
                discharges_this_month += 1

        # ---- 2. Effective arrival rate (shelter diverts some) ----
        eff_lambda = arrival_lambda
        if shelter_intervention:
            eff_lambda = arrival_lambda * (1 - SHELTER_ARRIVAL_REDUCTION)

        n_arrivals = rng.poisson(eff_lambda)

        new_agents = []
        for _ in range(n_arrivals):
            g = rng.choice(gender_keys, p=gender_vals)
            a = rng.choice(AGE_GROUPS, p=AGE_PROBS)
            new_agents.append(Agent(next_agent_id, arrival_month=sim_month, gender=g, age_group=a))
            next_agent_id += 1
        agents.extend(new_agents)

        # ---- 3. Admissions ----
        eff_adm_prob = admission_prob
        if seasonal and cal_month in WINTER_MONTHS:
            eff_adm_prob = min(1.0, eff_adm_prob + SEASONAL_ADMISSION_BOOST)
        if shelter_intervention:
            eff_adm_prob = max(0.0, eff_adm_prob - SHELTER_ADMISSION_REDUCTION)

        n_admissions = rng.binomial(n_arrivals, eff_adm_prob)
        if n_arrivals > 0 and n_admissions > 0:
            admitted_indices = rng.choice(n_arrivals, size=n_admissions, replace=False)
            for idx in admitted_indices:
                los_days = max(1.0, rng.normal(eff_los_mean, los_sd))
                new_agents[idx].admit(sim_month, los_days)

        # ---- 4. Occupancy & cost ----
        occupancy = sum(1 for a in agents if a.admitted and not a.discharged)

        monthly_cost_value = bed_days_this_month * COST_PER_BED_DAY
        if shelter_intervention:
            monthly_cost_value += SHELTER_COST_MONTHLY

        monthly_arrivals.append(n_arrivals)
        monthly_admissions.append(n_admissions)
        monthly_discharges.append(discharges_this_month)
        monthly_occupancy.append(occupancy)
        monthly_bed_days.append(bed_days_this_month)
        monthly_cost.append(monthly_cost_value)

    return {
        "agents":              agents,
        "monthly_arrivals":    monthly_arrivals,
        "monthly_admissions":  monthly_admissions,
        "monthly_discharges":  monthly_discharges,
        "monthly_occupancy":   monthly_occupancy,
        "monthly_bed_days":    monthly_bed_days,
        "monthly_cost":        monthly_cost,
    }


# =============================================================================
# SCENARIO COMPARISON
# =============================================================================

def run_all_scenarios(months=24, seed=42):
    scenarios = {
        "Baseline":           run_simulation(months=months, seed=seed,
                                             shelter_intervention=False, housing_intervention=False),
        "Shelter":            run_simulation(months=months, seed=seed,
                                             shelter_intervention=True,  housing_intervention=False),
        "Housing/Prevention": run_simulation(months=months, seed=seed,
                                             shelter_intervention=False, housing_intervention=True),
        "Combined":           run_simulation(months=months, seed=seed,
                                             shelter_intervention=True,  housing_intervention=True),
    }
    return scenarios


# =============================================================================
# PLOTTING
# =============================================================================

COLORS = {
    "Baseline":           "#e74c3c",
    "Shelter":            "#3498db",
    "Housing/Prevention": "#2ecc71",
    "Combined":           "#9b59b6",
}

def plot_scenarios(scenarios: dict, months: int = 24):
    month_axis = np.arange(1, months + 1)
    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])

    for name, res in scenarios.items():
        c = COLORS[name]
        ax1.plot(month_axis, res["monthly_arrivals"],   color=c, marker='o', ms=4, label=name)
        ax2.plot(month_axis, res["monthly_admissions"], color=c, marker='s', ms=4, label=name)
        ax3.plot(month_axis, res["monthly_occupancy"],  color=c, marker='d', ms=4, label=name)
        ax4.plot(month_axis, res["monthly_discharges"], color=c, marker='^', ms=4, label=name)
        cumcost = np.cumsum(res["monthly_cost"]) / 1e6
        ax5.plot(month_axis, cumcost, color=c, lw=2.5, label=name)

    for ax, title, ylabel in [
        (ax1, "Monthly ED Arrivals",      "# Arrivals"),
        (ax2, "Monthly Admissions",       "# Admissions"),
        (ax3, "Hospital Occupancy (EOM)", "# Beds Occupied"),
        (ax4, "Monthly Discharges",       "# Discharges"),
    ]:
        ax.set_title(title); ax.set_xlabel("Month"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax5.set_title("Cumulative Cost Over Time (CAD Millions)")
    ax5.set_xlabel("Month"); ax5.set_ylabel("Cumulative Cost (M CAD)")
    ax5.legend(); ax5.grid(alpha=0.3)

    fig.suptitle("Homeless ED Utilization ABM — Scenario Comparison", fontsize=14, fontweight='bold')
    plt.savefig("outputs/scenario_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_demographics(scenarios: dict):
    """Show age & gender breakdown of admitted agents for baseline."""
    baseline_agents = [a for a in scenarios["Baseline"]["agents"] if a.admitted]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Gender
    from collections import Counter
    gender_counts = Counter(a.gender for a in baseline_agents)
    axes[0].bar(gender_counts.keys(), gender_counts.values(),
                color=["#3498db", "#e74c3c", "#95a5a6"])
    axes[0].set_title("Admitted Agents by Gender (Baseline)")
    axes[0].set_ylabel("Count")

    # Age group
    age_counts = Counter(a.age_group for a in baseline_agents)
    age_order  = ["18-29", "30-44", "45-59", "60+"]
    axes[1].bar(age_order, [age_counts.get(g, 0) for g in age_order],
                color="#2ecc71")
    axes[1].set_title("Admitted Agents by Age Group (Baseline)")
    axes[1].set_ylabel("Count")

    fig.suptitle("Demographics of Admitted Homeless Individuals", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/demographics.png", dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# SUMMARY TABLE
# =============================================================================

def print_summary(scenarios: dict):
    print("\n" + "="*70)
    print(f"{'Scenario':<22} {'Arrivals':>9} {'Admissions':>11} {'Bed-Days':>10} {'Total Cost (M CAD)':>19} {'vs Baseline':>12}")
    print("="*70)
    baseline_cost = sum(scenarios["Baseline"]["monthly_cost"])
    for name, res in scenarios.items():
        arrivals    = sum(res["monthly_arrivals"])
        admissions  = sum(res["monthly_admissions"])
        bed_days    = sum(res["monthly_bed_days"])
        total_cost  = sum(res["monthly_cost"])
        saving      = baseline_cost - total_cost
        saving_str  = f"-${saving/1e6:.3f}M" if saving > 0 else f"+${abs(saving)/1e6:.3f}M"
        print(f"{name:<22} {arrivals:>9} {admissions:>11} {bed_days:>10.0f} {total_cost/1e6:>19.3f} {saving_str:>12}")
    print("="*70 + "\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    MONTHS = 24

    print("Running all scenarios...")
    scenarios = run_all_scenarios(months=MONTHS, seed=42)

    print_summary(scenarios)
    plot_scenarios(scenarios, months=MONTHS)
    plot_demographics(scenarios)

    print("Plots saved to outputs/")
