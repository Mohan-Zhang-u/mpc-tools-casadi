"""On/off control of a tank with softened constraints."""
import numpy as np
import mpctools as mpc
import matplotlib.pyplot as plt

# Choose parameters.
udiscrete = False # Must be True or False.
hmax = 2
hsp = 1
hdb = 0.25 # Deadband on tank height.
qmax = 1

# Define model.
A = 0.85
B = 0.5

Nx = 1
Nu = 1
Ns = 2*Nx # One for each bound.
Nt = 25

N = dict(x=Nx, u=Nu, t=Nt, s=Ns, sf=Ns, e=Ns, ef=Ns)

# Casadi functions for model, stage cost, and constraints.
f = mpc.getCasadiFunc(lambda x, u: A*x + B*u, [Nx, Nu], ["x", "u"],
                      funcname="f")
l = mpc.getCasadiFunc(lambda s, Du: 1000*sum(s) + mpc.mtimes(Du.T, Du),
                      [Ns, Nu], ["s", "Du"], funcname="l")
Vf = mpc.getCasadiFunc(lambda sf: 10000*sum(sf), [Ns], ["sf"], funcname="Vf")

def constraints(x, s):
    """Constraints on height."""
    terms = [
        x - (hsp + hdb) - s[0],
        (hsp - hdb) - x - s[1],
    ]
    return mpc.vcat(terms)

e = mpc.getCasadiFunc(constraints, [Nx, Ns], ["x", "s"], funcname="e")
ef = mpc.getCasadiFunc(constraints, [Nx, Ns], ["x", "sf"], funcname="ef")

# Specify bounds.
lb = {}
lb["u"] = np.zeros((Nt, Nu))
lb["x"] = np.zeros((Nt + 1, Nx))

ub = {}
ub["u"] = qmax*np.ones((Nt, Nu))
ub["x"] = hmax*np.ones((Nt + 1, Nx))

# Build controller and solve.
if udiscrete:
    solver = 'gurobi'
else:
    solver = 'ipopt'

x0 = np.array([0]) # Start with empty tank.
controller = mpc.nmpc(f=f, l=l, Pf=Vf, e=e, ef=ef, N=N, lb=lb, ub=ub,
                      uprev=np.zeros(Nu), x0=x0,
                      udiscrete=np.array([udiscrete]), solver=solver,
                      isQP=True, inferargs=True, verbosity=0)
controller.solve()
print(controller.stats["status"])


s = controller.vardict["s"]
Du = controller.vardict["Du"]
sf = controller.vardict["sf"]
obj = sum(l(s[t,:], Du[t,:]) for t in range(Nt)) + Vf(sf[0,:])

# Make a plot.
[fig, ax] = plt.subplots(nrows=2, figsize=(4, 6))

x = controller.vardict["x"]
u = controller.vardict["u"]
t = np.arange(Nt + 1)

ax[0].plot(t, x[:,0], marker="o", color="black")
for y in [hsp + hdb, hsp - hdb]:
    ax[0].axhline(y, color="blue", linestyle="--")
ax[0].set_ylabel("$h$", rotation=0)

ax[1].plot(t, np.concatenate((u[:,0], u[-1:,0])), marker="o", color="black",
           drawstyle="steps-post")
for y in [qmax, 0]:
    ax[1].axhline(y, color="blue", linestyle="--")
ax[1].set_xlabel('Time');
ax[1].set_ylabel("$q$", rotation=0)
