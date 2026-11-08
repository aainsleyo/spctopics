# Libraries
import numpy as np
from scipy.linalg import lu_factor, lu_solve

# Parameters
mu = 1.0
n = 50
delta = 0.001
t_f = 1.0

# Spatial grid
h = 2 / n
x = np.linspace(-1, 1, n + 1)

# Initial condition (interior points only)
u = np.sin(np.pi * (x[1:-1] + 1) / 2)

# Build matrix M
m = n - 1
I = np.eye(m)

D2 = (
    np.diag(-2 * np.ones(m))
    + np.diag(np.ones(m - 1), 1)
    + np.diag(np.ones(m - 1), -1)
) / h**2

M = I - delta * D2

# LU factorization (only needs to be done once)
LU = lu_factor(M)

# Time stepping
for t in np.arange(0, t_f, delta):

    # Calculate F
    F = u + delta * mu * np.sin(u)

    # Solve M*u_new = F
    u = lu_solve(LU, F)

# Add boundary conditions
u = np.concatenate(([0], u, [0]))

print(u)