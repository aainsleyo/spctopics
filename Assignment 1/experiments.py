# libraries
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from g_simul import rhs, Jac, RK2step

# parameters and settings
BASE = np.array([0.1, 1.0, 1.0, 1.6])
H, MAX_TIME, TOL = 0.001, 40.0, 1e-3
COLORS = {"near_stable_eq": "darkgreen", "cutoff": "firebrick",
          "unresolved": "darkorange", "numerical_failure": "navy"}


# Compute the equilibria of the system given parameters p = [a, c, b, G].
def equilibria(p):
    a, c, b, G = p
    D = (a + b)**2 + 4*a*b*G
    if D < -1e-12:
        return []
    root = np.sqrt(max(D, 0.0)) if abs(D) > 1e-12 else 0.0
    signs = [1] if root == 0 else [1, -1]
    return [np.array([(-a-b+s*root)/(2*a), (-a-b+s*root)/(2*b)])
            for s in signs]


# Simulate the system from an initial condition until it reaches a stable equilibrium,
# hits a cutoff, or the maximum time is reached, as recommended
def simulate(initial, p, h=H, max_time=MAX_TIME):
    stable = [eq for eq in equilibria(p)
              if np.max(np.linalg.eigvals(Jac(eq, p)).real) < -1e-8]
    state = np.array(initial, dtype=float)
    path = [state.copy()]
    outcome = "unresolved"
    steps = int(round(max_time / h))
    for step in range(1, steps + 1):
        state = RK2step(state, h, p)
        if not np.all(np.isfinite(state)):
            outcome = "numerical_failure"
            break
        path.append(state.copy())
        if state[1] < -100 or np.max(np.abs(state)) > 1e4:
            outcome = "cutoff"
            break
        if step % 100 == 0 and any(np.linalg.norm(state-eq) < TOL for eq in stable):
            outcome = "near_stable_eq"
            break
    return outcome, np.asarray(path), step*h

# Main function to run various experiments and collect results.
def main():
    a, c, b, _ = BASE
    fold_G = -(a+b)**2 / (4*a*b)
    print(f"G_SN = {fold_G:.6f}")
    # Include the fold, the upper-branch stability change, and the base case.
    for G in [fold_G-0.1, fold_G, -3.0, -2.0, -1.5, BASE[3]]:
        p = BASE.copy()
        p[3] = G
        if not equilibria(p):
            print(f"G={G:.3f}: no real equilibria")
        for eq in equilibria(p):
            J = Jac(eq, p)
            print("Jacobian:\n", J)
            print(f"G={G:.3f}, eq={eq.round(4)}, trace={np.trace(J):.4f}, "
                  f"det={np.linalg.det(J):.4f}, eigenvalues={np.linalg.eigvals(J)}")

    groups, rows = {}, []

    def run(group, initial, G=BASE[3], h=H, max_time=MAX_TIME, a_value=BASE[0]):
        p = BASE.copy()
        p[3] = G
        p[0] = a_value
        outcome, path, time = simulate(initial, p, h, max_time)
        rows.append(dict(experiment=group, a=p[0], b=p[2], c=p[1], G=G,
                         u0=initial[0], v0=initial[1], h=h, max_time=max_time,
                         outcome=outcome, time=time,
                         final_u=path[-1, 0], final_v=path[-1, 1],
                         equilibrium_count=len(equilibria(p))))
        groups.setdefault(group, []).append((p, path, outcome))
        print(f"{group}: G={G:g}, initial={np.round(initial, 4)}, {outcome}, t={time:g}")

    for v0 in np.r_[-3, -2.5, -2, -1.8, np.linspace(-1.5, -1, 11), 0, 0.5]:
        run("initial", [-8, v0])
    upper = equilibria(BASE)[0]
    for offset in [[0, 0], [-0.05, 0], [0.05, 0], [0, -0.05], [0, 0.05]]:
        run("local", upper + offset)
    for G in [-3.5, -3.1, fold_G, -3, -2.9, -2.5, -2, -1.5]:
        run("fold", [-8, -1.8], G)
    for G in np.arange(3, 5.01, 0.25):
        run("input", [-8, -1.8], G)

    for a_value in [0.05, 0.1, 0.2, 0.5, 1.0]:
        run("decay", [-8, -1.8], a_value=a_value)

    # Retry any unresolved ordinary runs with a longer horizon from the same start.
    #for row in rows.copy():
    #    if row["outcome"] == "unresolved":
    #        run("longer", [row["u0"], row["v0"]], row["G"],
    #            max_time=200, a_value=row["a"])

    # Comparing the same step sizes at the same longer horizon near apparent boundaries.
    for v0 in [-1.40, -1.35]:
        for h in [H, H/4]:
            run("check", [-8, v0], h=h, max_time=200)

    for G in [4.5, 4.75]:
        for h in [H, H/4]:
            run("check_G", [-8, -1.8], G, h=h, max_time=200)

    # Save the results to a CSV file.
    with open("glucose_experiments.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Plot the phase planes for the different experiment groups.
    fig, axes = plt.subplots(3, 2, figsize=(11, 11))
    titles = {"initial": "Vary initial v; G=1.6, u0=-8",
              "local": "Upper equilibrium and +/-0.05 perturbations",
              "fold": "Across the fold; initial=(-8,-1.8)",
              "input": "Refine G; initial=(-8,-1.8)",
              "decay": "Vary a; G=1.6, initial=(-8,-1.8)"}
    for ax, (group, title) in zip(axes.flat, titles.items()):
        
        for p, path, outcome in groups[group]:
            color = COLORS[outcome]
            ax.plot(*path.T, color=color, alpha=0.7, lw=1)
            ax.plot(*path[0], "o", color=color, ms=3)
            # Show direction over a visible segment near the start.
            k = min(50, len(path)-1)
            if k:
                ax.annotate("", xy=path[k], xytext=path[0],
                            arrowprops=dict(arrowstyle="->", color=color))
        for p in {tuple(p) for p, _, _ in groups[group]}:
            for eq, marker in zip(equilibria(p), ["*", "x"]):
                ax.plot(*eq, marker, color="black", ms=7)
        if group == "local":
            ax.set_xlim(upper[0]-0.08, upper[0]+0.08)
            ax.set_ylim(upper[1]-0.08, upper[1]+0.08)
        else:
            ax.set_xlim(-16, 6)
            ax.set_ylim(-3.5, 7.5)  # Includes the recovery overshoot.
        ax.axhline(-1, color="gray", ls=":", lw=0.7)
        ax.set(title=title, xlabel="u", ylabel="v")
        ax.grid(alpha=0.2)
        
    axes[2, 1].axis("off")
    
    for name, color in COLORS.items():
        axes[2, 1].plot([], [], color=color, label=name.replace("_", " "))
    axes[2, 1].legend(loc="center", frameon=False)
    
    fig.suptitle("Phase planes; circles: starts, *: plus branch, x: minus branch")
    fig.subplots_adjust(hspace=0.55, wspace=0.32, top=0.92, bottom=0.06)
    fig.savefig("phase_planes.png", dpi=180)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    names = [name for name in COLORS if any(
        row["outcome"] == name and row["experiment"] in ["initial", "fold", "input", "decay"]
        for row in rows)]
    for ax, group, xkey in zip(axes.flat, ["initial", "fold", "input", "decay"], ["v0", "G", "G", "a"]):
        for row in rows:
            if row["experiment"] == group:
                ax.scatter(row[xkey], names.index(row["outcome"]), color=COLORS[row["outcome"]])
        ax.set(title=titles[group], xlabel=xkey, yticks=range(len(names)), yticklabels=[name.replace("_", " ") for name in names])
        ax.set_ylim(-0.3, len(names)-0.7)
        if group == "fold":
            ax.axvline(fold_G, color="black", ls=":", label="G_SN")
            ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.suptitle("Observed outcomes among tested values")
    fig.tight_layout()
    fig.savefig("outcomes.png", dpi=180)
    # hiding full captions in the settings
    def values(group, key):
        return ", ".join(f"{value:.6g}" for value in sorted(
            {row[key] for row in rows if row["experiment"] == group}))
    plt.show()


if __name__ == "__main__":
    main()
