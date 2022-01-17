# Example from "A Lyapunov Function for Economic Optimizing Model Predictive
# Control" by Diehl, Amrit, and Rawlings (IEEE Trans. Auto. Cont., 56(3)), 2011
import numpy as np
import mpctools as mpc
import matplotlib.pyplot as plt
from matplotlib import cm, gridspec
import time

# Sizes.
Nx = 2
Nu = 1
Nt = 30
Nsim = 9
Nc = 4
Nrho = Nx + Nu

# Parameters.
cAf = 1
cBf = 0
Vr = 10
kr = 1.2
Qmax = 20
cAs = .5
cBs = .5
Qs = 12
Delta = .5

xs = np.array([cAs,cBs])
us = np.array([Qs])

# Models.
def cstrmodel(cA,cB,Q):
    dxdt = [
        Q/Vr*(cAf - cA) - kr*cA,
        Q/Vr*(cBf - cB) + kr*cA,
    ]
    return np.array(dxdt)

def stagecost(cA,cB,Q,rho):
    cost = (
        -2*Q*cB + 0.5*Q # Economics.
        + rho[0]*(cA - cAs)**2
        + rho[1]*(cB - cBs)**2
        + rho[2]*(Q - Qs)**2 # Deviation penalties.
    )
    return cost

# Some options based on whether or not we use collocation.
times = np.arange(0,Nsim+1)

# Convert to (x,u) notation.
def ode(x,u):
    # We would like to write
    #
    # [cA,cB] = x[:Nx]
    # [Q] = u[:Nu]
    #
    # but it doesn't work in Casadi 3.0. So,
    cA = x[0]
    cB = x[1]
    Q = u[0]    
    return cstrmodel(cA,cB,Q)

def lfunc(x,u,rho):
    # We would like to write
    #
    # [cA,cB] = x[:Nx]
    # [Q] = u[:Nu]
    #
    # but it doesn't work in Casadi 3.0. So,
    cA = x[0]
    cB = x[1]
    Q = u[0]
    return stagecost(cA,cB,Q,rho)

def Pffunc(x):
    dx = x[:Nx] - xs
    return 1e6*Nt*mpc.mtimes(dx.T,dx)

# Get Casadi functions.
f = mpc.getCasadiFunc(ode, [Nx,Nu], ["x","u"], "f")
largs = ["x","u","rho"]
l = mpc.getCasadiFunc(lfunc, [Nx,Nu,Nrho], largs, "L")
Pf = mpc.getCasadiFunc(Pffunc, [Nx], ["x"], "Pf")
model = mpc.DiscreteSimulator(ode, Delta, [Nx,Nu], ["x","u"])

# Build Lagrangian and reduced lagrangian.
def lagrangian(cA,cB,Q,lam,rho):
    return stagecost(cA,cB,Q,rho) + lam.dot(cstrmodel(cA,cB,Q))

# Now check strong duality.
equations = ["(41)","(40)"]
rhos = [np.array([.505,.505,.505]), np.zeros((Nx+Nu,))]
lam = np.array([-10,-20])
colors = ["blue","red"]
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(1,1,1)
eps = 5e-2
Qvals = np.linspace(Qs-1,Qs+1,101)
for (eq,rho,color) in zip(equations,rhos,colors):
    Lvals = lagrangian(cAs,cBs,Qvals,lam,rho)
    ax.plot(Qvals,Lvals,color=color,label="Objective " + eq)
mpc.plots.zoomaxis(ax,yscale=1.1)
ax.set_xlabel("$Q$ (L/min)")
ax.set_ylabel("Lagrangian (Slice in $Q$ with $c = c_{ss}$)")
ax.legend(loc="upper center")
fig.tight_layout(pad=.5)
mpc.plots.showandsave(fig,"duality.pdf")

# Build controller.
buildtime = -time.time()
contargs = dict(
    f=f,
    l=l,
    N={"x":Nx, "u":Nu, "t":Nt, "c":Nc},
    Delta=Delta,
    ub={"u" : Qmax*np.ones((Nt,Nu))},
    lb={"x" : np.zeros((Nt+1,Nx)), "u" : np.zeros((Nt,Nu))},
    guess={
        "x" : np.tile(xs,(Nt+1,1)),
        "u" : np.tile(us,(Nt,1)),
        "xc" : np.tile(xs.reshape((1,Nx,1)),(Nt,1,Nc)),
    },
    x0=xs,
    extrapar={"rho" : np.zeros((Nrho,))},
    funcargs={"l" : largs},
    Pf=Pf,
    verbosity=0,
    discretel=False,
)
controller = mpc.nmpc(**contargs)
buildtime += time.time()
print("Building controller took %.4g s." % buildtime)

# Pick different initial conditions and get open-loop profiles.
x0vals = [np.array([(.7*np.cos(t) + 1)*cAs, (.7*np.sin(t)+1)*cBs])
    for t in np.linspace(0,2*np.pi,10)[:-1]]
XCL = [] # "Closed-loop x trajectory."
XCLC = [] # "Closed-loop x collocation points."
LVALS = []
LROTVALS = []
XF = []
for x0 in x0vals:
    print("x0 = [%10.5g, %10.5g]" % (x0[0],x0[1]))
    # Preallocate.
    xcl = {}
    xclc = {}
    lvals = {}
    xf = {}
    lrotvals = {}
    for eq in equations:
        xcl[eq] = np.zeros((Nsim+1,Nx))
        xcl[eq][0,:] = x0
        xclc[eq] = np.zeros((Nsim*(Nc+1)+1,Nx))
        xclc[eq][0,:] = x0
        xf[eq] = np.zeros((Nsim,Nx))
        lvals[eq] = np.zeros((Nsim,))
        lrotvals[eq] = np.zeros((Nsim,))
        
    # Simulate.
    for t in range(Nsim):
        for (eq,rho) in zip(equations,rhos):
            controller.fixvar("x",0,xcl[eq][t,:])
            controller.par["rho"] = rho

            # Solve with just terminal penalty.
            controller.solve()
            print("    %s %3d: %s" % (eq,t,controller.stats["status"]))
            if controller.stats["status"] != "Solve_Succeeded":
                mpc.keyboard()
                break

            # Now grab results.
            lamalt = np.array([-10,-20])
            xcl[eq][t+1,:Nx] = model.sim(controller.var["x",0,:Nx],
                controller.var["u",0])
            lvals[eq][t] = (controller.obj  - Nt*Delta*lfunc(xs,us,rho)
                - Pffunc(controller.var["x",-1]))
            lrotvals[eq][t] = lvals[eq][t] - lam.dot(xcl[eq][t,:Nx] - xs)
            tmin = t*(Nc+1) + 1
            tmax = tmin + Nc
            xclc[eq][tmin:tmax] = controller.var["xc",0].T
            xclc[eq][tmax,:] = np.squeeze(controller.var["x",1])
            xf[eq][t,:] = np.squeeze(controller.var["x",-1])
        controller.saveguess()    
    XCL.append(xcl)
    XCLC.append(xclc)
    XF.append(xf)
    LVALS.append(lvals)
    LROTVALS.append(lrotvals)

# Plots.
colors = [cm.Set1(c) for c in np.linspace(0,1,len(x0vals))]
for eq in equations:
    gs = gridspec.GridSpec(2,2)
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(gs[:,0])
    costax = fig.add_subplot(gs[0,1])
    rotax = fig.add_subplot(gs[1,1])
    for (lvals,lrotvals,xcl,xclc,c) in zip(LVALS,LROTVALS,XCL,XCLC,colors):
        # Plot closed-loop stage cost.
        t = np.arange(Nsim)
        costax.plot(t, lvals[eq], color=c)
        rotax.plot(t, lrotvals[eq], color=c)
        # Plot phase trajectory with collocation points.
        ax.plot(xcl[eq][:,0],xcl[eq][:,1],"o",color=c,markerfacecolor=c,
                markeredgecolor=c,markersize=6)
        ax.plot(xclc[eq][:,0],xclc[eq][:,1],"-o",markersize=4,color=c,
                markerfacecolor="none",markeredgecolor=c)
        ax.plot(xcl[eq][0,0],xcl[eq][0,1],"o",markeredgecolor=c,
                markerfacecolor=c,markersize=8)
    # Clean up.
    ax.set_xlabel("$c_A$ (mol/L)")
    ax.set_ylabel("$c_B$ (mol/L)")
    ax.axis("equal")
    mpc.plots.zoomaxis(costax,yscale=1.1)
    mpc.plots.zoomaxis(rotax,yscale=1.1)
    rotax.set_xlabel("Time")
    rotax.set_ylabel("Rotated Cost")
    costax.set_ylabel("Economic Cost")
    costax.set_title("Objective %s" % (eq,))
    fig.tight_layout(pad=.5)
    mpc.plots.showandsave(fig,"econmpc%s.pdf" % (eq,))
