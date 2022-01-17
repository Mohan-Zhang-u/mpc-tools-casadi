# Car driving on an icy hill.
import numpy as np
import mpctools as mpc
import matplotlib.pyplot as plt

# Define sizes and hill functions.
Nx = 2
Nu = 1
Nc = 2
Nt = 60
Delta = 0.25
umax = 0.5

def hill(x):
    """Normal distribution as hill."""
    return np.exp(-x**2/2)
dhill = mpc.util.getScalarDerivative(hill, vectorize=False)

# Define ODE and get LQR.
def ode(x, u):
    """
    Model dx/dt = f(x,u)
    
    x is [position, velocity]
    u is [external thrust]
    """
    dh = dhill(x[0])
    sintheta = dh/np.sqrt(1 + dh**2)
    costheta = 1/np.sqrt(1 + dh**2)
    dxdt = [x[1], (u[0] - sintheta)*costheta]
    return np.array(dxdt)
model = mpc.DiscreteSimulator(ode, Delta, [Nx, Nu], ["x", "u"])
odecasadi = mpc.getCasadiFunc(ode, [Nx, Nu], ["x", "u"], "f")

# Get LQR.
xss = np.zeros((Nx,))
uss = np.zeros((Nu,))
fcasadi = mpc.getCasadiFunc(ode, [Nx, Nu], ["x", "u"], funcname="f")
ss = mpc.util.getLinearizedModel(fcasadi, [xss,uss], ["A","B"], Delta=Delta)
ss["Q"] = np.eye(Nx)
ss["R"] = np.eye(Nu)
[ss["K"], ss["Pi"]] = mpc.util.dlqr(ss["A"], ss["B"], ss["Q"], ss["R"])

# Build controller.
def l(x, u):
    """Quadratic stage cost x'Qx + u'Ru."""
    return mpc.mtimes(x.T, ss["Q"], x) + mpc.mtimes(u.T, ss["R"], u)
lcasadi = mpc.getCasadiFunc(l, [Nx, Nu], ["x", "u"], "l")

def Vf(x):
    """Quadratic terminal penalty."""
    return mpc.mtimes(x.T, ss["Pi"], x)
Vfcasadi = mpc.getCasadiFunc(Vf, [Nx], ["x"], "Vf")

lb = dict(u=-umax*np.ones((Nt, Nu)))
ub = dict(u=umax*np.ones((Nt, Nu)))

N = {"t" : Nt, "x" : Nx, "u" : Nu, "c" : Nc}

x0s = [np.array([-x, 0]) for x in np.linspace(0, 2, 21)]
controller = mpc.nmpc(f=odecasadi, l=lcasadi, Pf=Vfcasadi, N=N, lb=lb, ub=ub,
                      Delta=Delta, verbosity=0)

# Find open-loop solution for each initial condition.
xs = []
us = []
Vs = []
for (i, x0) in enumerate(x0s):
    controller.saveguess(default=True) # Reset guess to default.
    controller.fixvar("x", 0, x0)
    controller.solve()
    print("Step %d of %d: %s" % (i + 1, len(x0s), controller.stats["status"]))
    xs.append(controller.vardict["x"].copy())
    us.append(controller.vardict["u"].copy())
    Vs.append(controller.obj)

# Plot trajectories.
[fig, ax] = plt.subplots()
for x in xs:
    ax.plot(x[:,0], x[:,1], color="black")
    ax.plot(x[0,0], x[0,1], marker="o", markeredgecolor="black",
            markerfacecolor="black")
ax.set_xlabel("Position")
ax.set_ylabel("Velocity")
fig.tight_layout()
mpc.plots.showandsave(fig, "icyhill.pdf")
