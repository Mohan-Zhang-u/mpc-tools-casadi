# Example of steady-state target with an unreachable setpoint and
# soft constraints. Tradeoff is between violating output constraints and
# moving further from setpoint.
import numpy as np
import mpctools as mpc
import matplotlib.pyplot as plt

# Sizes and model.
Nx = 1
Ny = 1
Nu = 1
Ns = Ny

f = mpc.getCasadiFunc(lambda x,u: 0.5*x + u, [Nx, Nu], ["x", "u"],
                      funcname="f")
h = mpc.getCasadiFunc(lambda x: 0.5*x, [Nx], ["x"], funcname="h")

# Slacked output constraints and objective function.
ylb = -1
yub = 1
def outputconstraints(y, s):
    """Slacked output constraints."""
    terms = [
        ylb - y - s,
        y - yub - s,
    ]
    return mpc.vcat(terms)
e = mpc.getCasadiFunc(outputconstraints, [Ny, Ns], ["y", "s"], funcname="e")

def stagecost(y, u, ysp, usp, s, slackpen):
    """Quadratic penalty with linear slacks."""
    dy = y - ysp
    ypen = mpc.mtimes(dy.T, dy)
    du = u - usp
    upen = 100*mpc.mtimes(du.T, du)
    slack = slackpen*mpc.sum(s)
    return ypen + upen + slack
phi = mpc.getCasadiFunc(stagecost, [Ny, Nu, Ny, Nu, Ns, 1],
                        ["y", "u", "ysp", "usp", "s", "slackpen"],
                        funcname="phi")

# Build steady-state target finder.
N = dict(x=Nx, u=Nu, y=Ny, s=Ns, e=2*Ny)
extrapar = dict(slackpen=10, ysp=1, usp=2)
sstarg = mpc.sstarg(f=f, h=h, e=e, phi=phi, N=N, inferargs=True,
                    extrapar=extrapar, verbosity=0)

# Find optimal steady state as a function of slack penalty.
slackpens = np.linspace(0, 250, 251)
targ = dict(u=[], y=[], s=[])
for sp in slackpens:
    sstarg.par["slackpen"] = sp
    sstarg.solve()
    
    for (k, v) in targ.items():
        v.append(sstarg.vardict[k][0])

# Make a plot.
fields = ["y", "u", "s"]
targets = [extrapar["ysp"], extrapar["usp"], 0]
[fig, ax] = plt.subplots(nrows=3)
for (i, (k, sp)) in enumerate(zip(fields, targets)):
    ax[i].plot(slackpens, targ[k], color="green")
    ax[i].axhline(sp, color="k", linestyle="--")
    ax[i].set_ylabel(k, rotation=0)
ax[-1].set_xlabel("Slack Penalty Coefficient")
