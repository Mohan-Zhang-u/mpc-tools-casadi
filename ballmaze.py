import numpy as np
import mpctools as mpc
import matplotlib.pyplot as plt
import itertools

# Rolling ball game example. Linear model but nonlinear constraints.

# Some global options and parameters.
movingHorizon = False
terminalConstraint = True
terminalWeight = False
transientCost = True

Nx = 4
Nu = 2
Nt = 25
Nsim = 75

# Some bounds.
umax = 1
xmax = 2
cushion = .1

# Four states: x1, x2, v1, v2
# Two controls: a1, a2
Acont = np.array([
    [0,0,1,0],
    [0,0,0,1],
    [0,0,0,0],
    [0,0,0,0],
])
Bcont = np.array([
    [0,0],
    [0,0],
    [1,0],
    [0,1],
])

# Discretize.
Delta = .05
(A,B) = mpc.util.c2d(Acont, Bcont, Delta)

# Models.
f = mpc.getCasadiFunc(lambda x,u: mpc.mtimes(A,x) + mpc.mtimes(B,u),
                      [Nx,Nu],["x","u"], "f")

# Specify holes as (center, radius).
r = .25
m = 3                      
centers = np.linspace(0,xmax,m+1)
centers = list(.5*(centers[1:] + centers[:-1]))
holes = [(p,r) for p in itertools.product(centers,centers)]


Ne = len(holes)

def nlcon(x):
    # [x1, x2] = x[0:2] # Doesn't work in Casadi 3.0
    x1 = x[0]
    x2 = x[1]
    resid = [r**2 - (x1 - p1)**2 - (x2 - p2)**2 for ((p1,p2),r) in holes]
    return np.array(resid)
e = mpc.getCasadiFunc(nlcon, [Nx], ["x"], "e")

if not movingHorizon:
    Nt = Nsim
    Nsim = 1

x0 = np.array([xmax,xmax,0,0])
lb = {
    "u": -umax*np.ones((Nt,Nu)),
    "x": np.tile([-cushion,-cushion,-np.inf,-np.inf], (Nt+1,1)),
}
ub = {
    "u" : umax*np.ones((Nt,Nu)),
    "x" : np.tile([xmax+cushion,xmax+cushion,np.inf,np.inf], (Nt+1,1)),
}
guess = {"x" : np.tile(x0, (Nt+1,1))}
if transientCost:
    def lfunc(x):
        return mpc.mtimes(x[0:2].T,x[0:2])
else:
    def lfunc(x):
        return 0
l = mpc.getCasadiFunc(lfunc, [Nx], ["x"], "l")

rmin = .001
def terminalconstraint(x):
    return np.array([x[0]**2 +  x[1]**2 - rmin**2])
if terminalConstraint:
    ef = mpc.getCasadiFunc(terminalconstraint, [Nx], ["x"], "ef")
else:
    ef = None

funcargs = {"f" : ["x","u"], "e" : ["x"], "l" : ["x"], "ef" : ["x"]}
    
if terminalWeight:
    Pf = mpc.getCasadiFunc(lambda x: mpc.mtimes(x[0:2].T,x[0:2]), [Nx], ["x"])
else:
    Pf = None

# Build controller and adjust some ipopt options.
N = {"x":Nx, "u":Nu, "e":Ne, "t":Nt}
controller = mpc.nmpc(f, l, N, x0, lb, ub, funcargs=funcargs, e=e, Pf=Pf,
                      ef=ef, verbosity=(0 if movingHorizon else 5),
                      casaditype="SX")
controller.initialize(solveroptions=dict(max_iter=5000))

# Now ready for simulation.
x = np.zeros((Nsim+1,Nx))
x[0,:] = x0
u = np.zeros((Nsim,Nu))
for t in range(Nsim):
    controller.fixvar("x",0,x[t,:])
    controller.solve()
    print("%5d: %20s" % (t,controller.stats["status"]))
    
    x[t+1,:] = np.squeeze(controller.var["x",1])
    u[t,:] = np.squeeze(controller.var["u",0])
    
if not movingHorizon:
    sol = mpc.util.casadiStruct2numpyDict(controller.var)
    x = sol["x"]
    u = sol["u"]

def plotsol(x,holes,xmax,cushion=1):
    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    for (p,r) in holes:
        circ = plt.Circle(p,r,edgecolor="red",facecolor=(1,0,0,.5))
        ax.add_artist(circ)
    ax.plot(x[:,0],x[:,1],'-ok')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_xlim((-cushion,xmax+cushion))
    ax.set_ylim((-cushion,xmax+cushion))
    return f
    
fig = plotsol(x,holes,xmax,cushion)
mpc.plots.showandsave(fig, "ballmaze.pdf")
