import mpctools as mpc
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

random.seed(927)
MEASURE_PREY = False # Decide whether we know the total number of prey.

# Sizes.
Nx = 3
Nu = 1
Ny = 1 + MEASURE_PREY
Delta = 0.1

# Pick coefficients.
a = 0.5 # Kill rate for prey.
b = 1 # Birth rate for prey.
c = 0.5 # Birth rate for predators.
d = 1 # Death rate for predators.
N0prey = 50 # Scale factor for prey.
N0pred = 25 # Scale factor for predators.
def ode(x, u):
    """Predator/prey dynamics."""
    [Nprey, Ntag, Npred] = x[:]
    Nprey /= N0prey
    Ntag /= N0prey
    Npred /= N0pred
    [Rtag] = u[:]
    dxdt = [
        N0prey*(-a*Npred*Nprey + b*Nprey - Rtag*Nprey),
        N0prey*(-a*Npred*Ntag + Rtag*Nprey),
        N0pred*(c*Npred*(Nprey + Ntag) - d*Npred),
    ]
    return np.array(dxdt)
def measurement(x):
    """Returns fraction of tagged animals."""
    Nprey = x[0]
    Ntag = x[1]
    y = [Ntag/(Nprey + Ntag)]
    if MEASURE_PREY:
        y.append(Nprey + Ntag)
    return np.array(y)

# Convert to Casadi functions and simulator.    
model = mpc.DiscreteSimulator(ode, Delta, [Nx,Nu], ["x","u"])
f = mpc.getCasadiFunc(ode, [Nx,Nu], ["x","u"], "f", rk4=True, Delta=Delta)
h = mpc.getCasadiFunc(measurement, [Nx], ["x"], "h")
x0 = np.array([N0prey, 0, N0pred])
x0bar = np.array([1.5*N0prey, 0, 0.6*N0pred]) # Initial estimate.

# Simulate dynamics.
Nsim = 250
x = np.NaN*np.ones((Nsim + 1, Nx))
x[0,:] = x0
u = np.zeros((Nsim, Nu))
t = Delta*np.arange(Nsim)
u[:,0] = 0.1*(1 + np.sin(2*np.pi*t/10))
y = np.NaN*np.ones((Nsim + 1, Ny))
noise = 0.05*random.randn(Nsim + 1, Nx) # Multiplicative noise term.
for t in range(Nsim + 1):
    # Round x and take measurement.
    x[t,:] = np.maximum(np.round(x[t,:]*(1 + noise[t,:])), 0)
    y[t] = measurement(x[t,:])    
    
    # Simulate step.
    if t < Nsim:
        x[t + 1,:] = model.sim(x[t,:], u[t,:])

# Define a plotting function.
def doplot(t, x, y, xhat, yhat):
    """Makes a plot of actual and estimated states and outputs."""
    labels = ["Untagged Prey", "Tagged Prey", "Predators", "Tag Fraction"]
    estimated = [True, True, True, False]
    if MEASURE_PREY:
        labels.append("Total Prey")
        estimated.append(False)
    data = np.concatenate((x, y), axis=1)
    datahat = np.concatenate((xhat, yhat), axis=1)
    [fig, ax] = plt.subplots(nrows=len(labels))
    for (i, (label, est)) in enumerate(zip(labels, estimated)):
        ax[i].plot(t, data[:,i], color="green", label="Actual")
        ax[i].plot(t, datahat[:,i], color="red", label="Estimated")
        label = "\n".join([label, "(Estimated)" if est else "(Measured)"])
        ax[i].set_ylabel(label)
        if i == 0:
            ax[i].legend(loc="lower center", bbox_to_anchor=(0.5,1.01), ncol=2)
    ax[i].set_xlabel("Time")
    return fig

# Now try MHE.
def stagecost(w, v):
    """Stage cost for measurement and state noise."""
    return 0.1*mpc.mtimes(w.T, w) + 10*mpc.mtimes(v.T, v)
def prior(dx):
    """Prior weight function."""
    return 10*mpc.mtimes(dx.T, dx)
l = mpc.getCasadiFunc(stagecost, [Nx,Ny], ["w","v"])
lx = mpc.getCasadiFunc(prior, [Nx], ["dx"])

Nt = 25 # Window size for MHE.
N = dict(x=Nx, u=Nu, y=Ny, w=Nx, v=Ny, t=Nt)
guess = dict(x=np.tile(x0, (Nt + 1, 1)))
lb = dict(x=0*np.ones((Nt + 1, Nx))) # Lower bounds are ones.
mhe = mpc.nmhe(f, h, u[:Nt,:], y[:Nt + 1,:], l, N, lx, x0bar, lb=lb,
               guess=guess, wAdditive=True, verbosity=0)

xhat = np.NaN*np.ones((Nsim + 1, Nx))
for t in range(Nsim + 1):
    # Solve current MHE problem.
    mhe.solve()
    status = mhe.stats["status"]
    if t % 25 == 0 or status != "Solve_Succeeded":
        print("Step %d of %d: %s" % (t, Nsim - Nt, status))
    
    # Make a plot for the first step.
    if t == 0:
        firstfig = doplot(Delta*np.arange(Nt + 1), x[:Nt + 1,:], y[:Nt + 1,:],
                          mhe.vardict["x"], y[:Nt + 1,:] - mhe.vardict["v"])
    
    # If at end, save the remaining trajectory. Otherwise, cycle.
    if t + Nt >= Nsim:
        xhat[t:,:] = mhe.vardict["x"]
        break
    else:
        xhat[t,:] = mhe.vardict["x"][0,:]
        mhe.newmeasurement(y[t + Nt,:], u[t + Nt,:], mhe.vardict["x"][1,:])
        mhe.saveguess()
yhat = np.array([measurement(xhat[i,:]) for i in range(Nsim + 1)])

# Make a plot.
t = np.arange(Nsim + 1)*Delta
fig = doplot(t, x, y, xhat, yhat)
#mpc.plots.showandsave(firstfig, "predatorprey_first.pdf")
mpc.plots.showandsave(fig, "predatorprey.pdf")
