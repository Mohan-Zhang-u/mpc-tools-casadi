import mpctools as mpc
from mpctools import util
import mpctools.plots as mpcplots
import numpy as np
import matplotlib.pyplot as plt

# Build model.
k00 = -2
k11 = -.5
k10 = 1

def F(x,u):
    """Model function"""
    return [k00*x[0], k11*x[1] + k10*x[0]]
def L(x,u):
    """Stage cost"""
    return mpc.mtimes(x.T, x) + mpc.mtimes(u.T, u)
Nx = 2
Nu = 1
f = mpc.getCasadiFunc(F, [Nx,Nu], ["x","u"], "f")
l = mpc.getCasadiFunc(L, [Nx,Nu], ["x","u"], "l")
x0 = [1, 1]
Delta = .5

# Pick horizon and number of collocation points.
Nt = 10
Nc = 5
verbosity = 3

# Solve "optimization" to get feasible x points.
N = dict(x=Nx, u=Nu, c=Nc, t=Nt)
sol = mpc.callSolver(mpc.nmpc(f, l, N, x0, Delta=Delta, verbosity=verbosity))
x = sol["x"]
u = sol["u"]
z = sol["xc"]

# Plot some stuff.
colloc = util.smushColloc(None, x, None, z, Delta=Delta, asdict=True)
tx = colloc["tp"]
x = colloc["xp"]
tz = colloc["tc"]
z = colloc["xc"]
tfine = np.linspace(0,Nt*Delta,250)
xfine = np.zeros((2,len(tfine)))

# Analytical solution.
xfine[0,:] = x0[0]*np.exp(k00*tfine)

xfine[1,:] = xfine[0,:]/x0[0]*k10/(k00 - k11)
xfine[1,:] += (x0[1] - xfine[1,0])*np.exp(k11*tfine) 

# Plots.
f = plt.figure()
for i in range(2):
    ax = f.add_subplot(2,1,1+i)
    ax.plot(tfine, xfine[i,:], "-k", label="Analytical")
    ax.plot(tx, x[:,i],'o', label="State Points", markerfacecolor="k",
            markeredgecolor="k")
    ax.plot(tz, z[:,i], "o", label="Collocation Points",
            markerfacecolor="none", markeredgecolor="r")
    ax.set_ylabel("$x_{%d}$" % i)    
    ax.legend(loc="upper right")
ax.set_xlabel("Time")
mpcplots.showandsave(f,"collocationexample.pdf")
