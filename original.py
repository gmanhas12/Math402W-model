import numpy as np
import matplotlib.pyplot as plt

#the orignal code
class Agent:
    def __init__(self, agent_id, arrival_month):
        self.agent_id = agent_id
        self.arrival_month = arrival_month
        self.admitted = False
        self.admission_month = None
        self.length_of_stay_days = None
        self.remaining_days = None
        self.discharged = False
        self.discharge_month = None

    def admit(self, month, los_days):
        self.admitted = True
        self.admission_month = month
        self.length_of_stay_days = los_days
        self.remaining_days = los_days

    def update_one_month(self, current_month, days_in_month=30):
        """
        Reduce remaining hospital stay by one month.
        If remaining_days <= 0 after update, agent is discharged.
        """
        if self.admitted and not self.discharged:
            self.remaining_days -= days_in_month
            if self.remaining_days <= 0:
                self.discharged = True
                self.discharge_month = current_month


def run_simulation(
    months=24,
    initial_agents=80,
    arrival_lambda=3,
    admission_prob=0.1,
    los_mean=15.4,
    los_sd=2.0,
    seed=42
):
    rng = np.random.default_rng(seed)

    agents = []
    next_agent_id = 0

    # initial 80 agents exist in the population, but are not yet in hospital
    for _ in range(initial_agents):
        agents.append(Agent(next_agent_id, arrival_month=0))
        next_agent_id += 1

    monthly_arrivals = []
    monthly_admissions = []
    monthly_discharges = []
    monthly_occupancy = []

    for month in range(1, months + 1):
        # 1. Update currently admitted agents and discharge if their stay ends
        discharges_this_month = 0
        for agent in agents:
            was_in_hospital = agent.admitted and not agent.discharged
            agent.update_one_month(month)
            if was_in_hospital and agent.discharged and agent.discharge_month == month:
                discharges_this_month += 1

        # 2. Generate new arrivals this month
        n_arrivals = rng.poisson(arrival_lambda)
        new_agents = []
        for _ in range(n_arrivals):
            new_agent = Agent(next_agent_id, arrival_month=month)
            new_agents.append(new_agent)
            next_agent_id += 1

        agents.extend(new_agents)

        # 3. Among new arrivals, determine who gets admitted
        n_admissions = rng.binomial(n_arrivals, admission_prob)

        admitted_indices = []
        if n_arrivals > 0 and n_admissions > 0:
            admitted_indices = rng.choice(n_arrivals, size=n_admissions, replace=False)

        for idx in admitted_indices:
            los_days = rng.normal(los_mean, los_sd)

            # prevent negative or unrealistically tiny stays
            los_days = max(1, los_days)

            new_agents[idx].admit(month, los_days)

        # 4. Count hospital occupancy at end of this month
        occupancy = sum(1 for agent in agents if agent.admitted and not agent.discharged)

        monthly_arrivals.append(n_arrivals)
        monthly_admissions.append(n_admissions)
        monthly_discharges.append(discharges_this_month)
        monthly_occupancy.append(occupancy)

    return {
        "agents": agents,
        "monthly_arrivals": monthly_arrivals,
        "monthly_admissions": monthly_admissions,
        "monthly_discharges": monthly_discharges,
        "monthly_occupancy": monthly_occupancy
    }


def plot_results(results):
    months = np.arange(1, len(results["monthly_arrivals"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(months, results["monthly_arrivals"], marker='o', label="Arrivals")
    plt.plot(months, results["monthly_admissions"], marker='s', label="Admissions")
    plt.plot(months, results["monthly_discharges"], marker='^', label="Discharges")
    plt.plot(months, results["monthly_occupancy"], marker='d', label="Hospital occupancy")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.title("Agent-Based Hospital Flow Simulation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Example run
results = run_simulation(months=24, initial_agents=80, seed=42)
plot_results(results)

# Print a short summary
print("Total arrivals:", sum(results["monthly_arrivals"]))
print("Total admissions:", sum(results["monthly_admissions"]))
print("Total discharges:", sum(results["monthly_discharges"]))
print("Final occupancy:", results["monthly_occupancy"][-1])