import numpy as np
import csv
import matplotlib.pyplot as plt

# call the functions from g_simul.py
from g_simul import rhs, Jac, RK2step

# Set parameters: time step, system parameters.
h = 0.001
nStep = 40000
p_base = np.zeros(4)
p_base[0] = 0.1
p_base[1] = 1.0
p_base[2] = 1.0
p_base[3] = 1.6

def simulate(u0, p):
    """Integrate one trajectory and classify its observed behavior."""
    u = np.array(u0, dtype=float)
    trajectory = [u.copy()]
    outcome = "not_converged"

    # Compute the equilibria for this parameter set
    D = (p[0] + p[2])**2 + 4*p[0]*p[2]*p[3]

    if p[0] == 0 or p[2] == 0 or D <= 0:
        eq = None
    else:
        eq = np.array([
            -(p[0]+p[2])/(2*p[0]) + np.sqrt(D)/(2*p[0]),
            -(p[0]+p[2])/(2*p[2]) + np.sqrt(D)/(2*p[2])
        ])

    for step in range(nStep):
        u = RK2step(u, h, p)
        trajectory.append(u.copy())

        # Stop if the state becomes extreme
        if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 1e4:
            outcome = "divergent"
            break

        if u[1] < -100:
            outcome = "glucose_below_minus_100"
            break

        # Check whether the trajectory approaches eq0
        if eq is not None and step % 100 == 0:
            if np.linalg.norm(u - eq) < 1e-3:
                outcome = "approaches_eq0"
                break

    return outcome, np.array(trajectory)

trajectories = {
    "initial_condition": [],
    "vary_p3": [],
    "vary_p0": []
}

results = []

# Experiment 1: vary initial v, keeping u0 fixed
for v0 in [-3.0, -2.5, -2.0, -1.8, -1.5, -1.0, 0.0, 0.5]:
    p = p_base.copy()
    initial = [-8.0, v0]
    outcome, traj = simulate(initial, p)

    trajectories["initial_condition"].append((v0, outcome, traj))

    results.append([
        "initial_condition", *p, *initial,
        outcome, len(traj) - 1,
        traj[-1, 0], traj[-1, 1]
    ])

# Experiment 2: vary p3, keeping the initial condition fixed
for p3 in [-1.0, 0.0, 0.5, 1.0, 1.6, 2.0, 3.0, 5.0]:
    p = p_base.copy()
    p[3] = p3
    initial = [-8.0, -1.8]
    outcome, traj = simulate(initial, p)

    trajectories["vary_p3"].append((p3, outcome, traj))

    results.append([
        "vary_p3", *p, *initial,
        outcome, len(traj) - 1,
        traj[-1, 0], traj[-1, 1]
    ])

# Experiment 3: vary p0
for p0 in [0.05, 0.1, 0.2, 0.5, 1.0]:
    p = p_base.copy()
    p[0] = p0
    initial = [-8.0, -1.8]
    outcome, traj = simulate(initial, p)

    trajectories["vary_p0"].append((p0, outcome, traj))

    results.append([
        "vary_p0", *p, *initial,
        outcome, len(traj) - 1,
        traj[-1, 0], traj[-1, 1]
    ])

# Save all experimental results
header = [
    "experiment", "p0", "p1", "p2", "p3",
    "u0", "v0", "outcome", "steps",
    "final_u", "final_v"
]

with open("glucose_experiments.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(results)

for row in results:
    print(row)

print("Results saved to glucose_experiments.csv")


# Plot the trajectories in the phase plane
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

experiment_titles = [
    ("initial_condition", "Varying initial v"),
    ("vary_p3", "Varying p3"),
    ("vary_p0", "Varying p0")
]

for ax, (experiment, title) in zip(axes, experiment_titles):

    for value, outcome, traj in trajectories[experiment]:
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            label=f"{value}: {outcome}"
        )

    ax.set_title(title)
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.grid(True)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.show()


# Plot final v against the parameter or initial condition
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (experiment, title) in zip(axes, experiment_titles):

    values = []
    final_v = []

    for value, outcome, traj in trajectories[experiment]:
        values.append(value)
        final_v.append(traj[-1, 1])

    ax.plot(values, final_v, "o-")
    ax.set_title(title)
    ax.set_xlabel(
        "Initial v" if experiment == "initial_condition"
        else experiment.replace("vary_", "")
    )
    ax.set_ylabel("Final v")
    ax.grid(True)

plt.tight_layout()
plt.show()
