"""Example of adding custom constraints to an mpc problem."""
import numpy as np
import mpctools as mpc

# Define linear model.
A = np.array([[1, 0.05], [0, 0.95]])
B = np.array([[0], [1]])

Nx = 2
Nu = 1
f = mpc.getCasadiFunc(lambda x,u: A.dot(x) + B.dot(u), [Nx, Nu], ["x", "u"], "f")

# Bounds on u.
umax = 1
lb = dict(u=[-umax])
ub = dict(u=[umax])

# Define cost function.
Q = np.eye(Nx)
R = np.eye(Nu)
def lfunc(x,u):
    """Quadratic stage cost."""
    return mpc.mtimes(x.T, Q, x) + mpc.mtimes(u.T, R, u)
l = mpc.getCasadiFunc(lfunc, [Nx, Nu], ["x", "u"], "l")

# Initial condition and sizes.
x0 = np.array([5, 0])
Nt = 75
N = {"x" : Nx, "u" : Nu, "t" : Nt}

# Create solver.
solver = mpc.nmpc(f, l, N, x0, lb, ub, verbosity=0)

# Add some input blocking constraints. Note that you can achieve the same
# effect using time-varying rate-of-change constraints on u, but we want to
# illustrate custom constraints.
blocksizes = [10, 5]*5
if sum(blocksizes) > Nt:
    raise ValueError("Total block length is too long!")
newcon = []
t = 0
u = solver.varsym["u"] # Get CasADi symbolic variables.
for b in blocksizes:
    u0 = u[t]
    for i in range(b - 1):
        t += 1
        newcon.append(u0 - u[t])
    t += 1
solver.addconstraints(newcon)
solver.solve()
print(solver.stats["status"])

# Plot things.
[x, u] = [solver.vardict[k] for k in ["x", "u"]]
fig = mpc.plots.mpcplot(x, u, np.arange(Nt + 1), xsp=np.zeros_like(x))
uax = fig.axes[0]
t = 0
for delta in blocksizes:
    t += delta
    uax.axvline(t, color="black", linewidth=0.5)
