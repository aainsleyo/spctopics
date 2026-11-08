# Code is a compilation of handwritten, reference from https://phaseportrait.github.io/, imports from class, and help from ChatGPT.
# Addison A. Olstad

# Libraries and Imports
import numpy as np
import matplotlib.pyplot as plt
import phaseportrait
from g_simul import Jac, RK2step

# Parameters
b = c = 1.0
h, nStep = 0.001, 20000

# values for a and G to plot bifurcation diagrams and phase portraits
# (a, list of G values for phase portraits)
cases = {0.1: [-2.0, -1.7355, 1.6],
         0.5: [-1.0, 0.0, 1.6]}

# Initial conditions for trajectories in phase portraits
ICS = [[-8, -1.8], [-8, -1.0], [-4, -2.0], [-4, 0.0], [0, -2.0], [0, 1.0], [3, 0.0]]

# ---------------- Equilibria ----------------
def equilibria(a, G):
    """Return [eq_plus, eq_minus] (each [u*, v*]), or None if no real equilibria."""
    D = (a + c)**2 + 4*a*c*G
    if D < 0:
        return None
    return [np.array([(-(a + c) + s*np.sqrt(D)) / (2*a),
                      (-(a + c) + s*np.sqrt(D)) / (2*c)]) for s in (+1, -1)]

# ---------------- Trajectories ----------------
def trajectory(u0, p):
    """Compute trajectory starting from u0 with parameters p = [a, b, c, G]."""
    u, traj = np.array(u0, float), [np.array(u0, float)]
    for _ in range(nStep):
        u = RK2step(u, h, p) # Uses RK2step from g_simul.py
        if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 1e3 or u[1] < -100:
            break
        traj.append(u.copy())
    return np.array(traj)


# ---------------- Bifurcation diagrams ----------------
fig, axes = plt.subplots(1, len(cases), figsize=(13, 5))

# Plot equilibrium branches and saddle-node bifurcation points
for ax, a in zip(axes, cases):
    G_SN = -(a + c)**2 / (4*a*c)
    G = np.linspace(G_SN, 5, 500)
    root = np.sqrt(np.maximum((a + c)**2 + 4*a*c*G, 0))

    ax.plot(G, (-(a + c) + root) / (2*c), label="Upper branch")
    ax.plot(G, (-(a + c) - root) / (2*c), label="Lower branch")
    ax.axvline(G_SN, ls="--", color="k", label="Saddle-node")

    hopf = (1 - 2*a - a**2) / (a + 1) > 0
    if hopf:
        G_H, v_H = -(a + 2) / (a + 1)**2, -a*(a + 2) / (a + 1)
        ax.plot(G_H, v_H, "ro", ms=8, label="Candidate Hopf point")
        print(f"a = {a}: Candidate Hopf at G = {G_H:.4f}")
    else:
        print(f"a = {a}: No Hopf point")

    ax.set(title=f"a = {a} ({'Hopf condition satisfied' if hopf else 'No Hopf point'})",
           xlabel="G", ylabel="Equilibrium v*")
    ax.grid(True)
    ax.legend()

# Plotting the bifurcation diagrams
plt.suptitle("Equilibrium diagrams for different values of a")
plt.tight_layout()
plt.savefig("equilibrium_diagrams.pdf")
plt.show()


# ---------------- Phase portraits ----------------
for a, G_list in cases.items():
    for G in G_list:
        p = np.array([a, b, c, G])

        def vector_field(u, v):
            f = -v - u*(v + 1) + G
            return -a*u + b*f + c*v, f

        # Plot phase portrait with trajectories and equilibria
        fig, ax = phaseportrait.PhasePortrait2D(
            vector_field, [[-12, 8], [-4, 2]],
            Title=f"Phase Portrait: a={a}, G={G}",
            xlabel="u", ylabel="v", MeshDim=25).plot()

        # Plot trajectories from initial conditions
        for ic in ICS:
            tr = trajectory(ic, p)
            ax.plot(tr[:, 0], tr[:, 1], lw=1.2)
            ax.plot(*ic, "ko", ms=3)

        # Plot equilibria and their eigenvalues
        for i, eq in enumerate(equilibria(a, G) or []):
            ax.plot(*eq, "r*" if i == 0 else "bs", ms=12)
            print(f"a={a}, G={G}, equilibrium {i}: {eq}, "
                  f"eigenvalues={np.linalg.eigvals(Jac(eq, p))}")

        ax.set_xlim(-12, 8)
        ax.set_ylim(-4, 2)
        fig.tight_layout()
        fig.savefig(f"phase_portrait_a_{a}_G_{G}.pdf")
        plt.show()