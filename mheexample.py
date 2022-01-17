# Example for nonlinear MPC with Casadi.

# Imports.
import numpy as np
from numpy import random
from scipy import linalg
import mpctools as mpc
import matplotlib.pyplot as plt

random.seed(927) # Seed random number generator.

verb = 2
doPlots = True

# Problem parameters.
Delta = .1
Nt = 50
t = np.arange(Nt+1)*Delta

Nx = 3
Nu = 2
Ny = 2
Nw = Nx
Nv = Ny
Nc = 4

Acont = np.array([[-1,1,0],[0,-2,2],[0,0,-.5]])
Bcont = np.array([[1,0],[0,1],[1,1]])
Ccont = np.array([[1,1,0],[0,1,1]])
Gcont = Acont.dot(linalg.inv(linalg.expm(Acont*Delta) - np.eye(Nx)))

def fcontinuous(x,u,w):
    return mpc.mtimes(Acont,x) + mpc.mtimes(Bcont,u) + mpc.mtimes(Gcont,w)
fcontinuous = mpc.getCasadiFunc(fcontinuous,[Nx,Nu,Nw],["x","u","w"],"f")

(A, B) = mpc.util.c2d(Acont, Bcont, Delta)
C = Ccont

def Fdiscrete(x,u,w):
    return mpc.mtimes(A, x) + mpc.mtimes(B, u) + w
Fdiscrete = mpc.getCasadiFunc(Fdiscrete,[Nx,Nu,Nw],["x","u","w"],"F")

def H(x): return mpc.mtimes(C, x)
H = mpc.getCasadiFunc(H,[Nx],["x"],"H")

# Noise covariances.
Q = .01*np.diag([.1,.25,.05])
Qhalf = linalg.cholesky(Q,lower=True)
Qinv = linalg.inv(Q)

R = np.diag([.5,.25])
Rhalf = linalg.cholesky(R,lower=True)
Rinv = linalg.inv(R)

# First simulate the noisy system.
x0 = np.array([1,2,3])
x0hat = np.array([0,0,0])

omega = 2*np.pi/(Nt*Delta)
u = np.vstack((np.sin(omega*t),np.cos(omega*t))).T # Use periodic input.
u = u[:-1,:] # Get rid of final u (it isn't used).
w = Qhalf.dot(random.randn(Nw,Nt)).T
v = Rhalf.dot(random.randn(Nv,Nt+1)).T

x = np.zeros((Nt+1,Nx))
x[0,:] = x0
y = np.zeros((Nt+1,Ny))

for k in range(Nt+1):
    thisy = H(x[k,:])
    y[k,:] = np.squeeze(thisy) + v[k,:]
    if k < Nt:
        xnext = Fdiscrete(x[k,:],u[k,:], w[k,:])
        x[k+1,:] = np.squeeze(xnext)
    
# Plot simulation.
if doPlots:
    f = plt.figure()
    
    # State
    ax = f.add_subplot(2,1,1)
    for i in range(Nx):
        ax.plot(t,x[:,i],label="$x_{%d}$" % (i,))
    ax.set_ylabel("$x$")
    ax.legend()
    
    # Measurement
    ax = f.add_subplot(2,1,2)
    for i in range(Ny):
        ax.plot(t,y[:,i],label="$y_{%d}$" % (i,))
    ax.set_ylabel("$y$")
    ax.set_xlabel("Time")
    ax.legend()
    f.tight_layout(pad=.5)

# Now we're ready to try some state estimation. First define stage cost and
# prior.
def l(w,v):
    return mpc.mtimes(w.T,Qinv,w) + mpc.mtimes(v.T,Rinv,v)
l = mpc.getCasadiFunc(l,[Nw,Nv],["w","v"],"l")
def lx(x):
    return 10*mpc.mtimes(x.T,x)
lx = mpc.getCasadiFunc(lx,[Nx],["x"],"lx")

N = {"t" : Nt, "x" : Nx, "y" : Ny, "u" : Nu, "c" : Nc}

out = mpc.callSolver(mpc.nmhe(fcontinuous, H, u, y, l, N, lx, x0hat,
                              verbosity=5, Delta=Delta))

west = out["w"]
vest = out["v"]
xest = out["x"]
xerr = xest - x

# Now we need to smush together all of the collocation points and actual
# points.
[T,X,Tc,Xc] = mpc.util.smushColloc(out["t"],out["x"],out["tc"],out["xc"])

# Plot estimation.
if doPlots:
    colors = ["red","blue","green"]    
    
    f = plt.figure()
    
    # State
    ax = f.add_subplot(2,1,1)
    for i in range(Nx):
        c = colors[i % len(colors)]
        ax.plot(T,X[:,i],label=r"$\hat{x}_{%d}$" % (i,),color=c)
        ax.plot(t,xest[:,i],"o",markeredgecolor=c,markerfacecolor=c,
                markersize=3.5)
        ax.plot(Tc,Xc[:,i],"o",markeredgecolor=c,markerfacecolor="none",
                markersize=3.5)
    ax.set_ylabel("$x$")
    ax.legend()
    
    # Measurement
    ax = f.add_subplot(2,1,2)
    for i in range(Nx):
        ax.plot(t,xerr[:,i],label=r"$\hat{x}_{%d} - x_{%d}$" % (i,i))
    ax.set_ylabel("Error")
    ax.set_xlabel("Time")
    ax.legend()
    f.tight_layout(pad=.5)
    mpc.plots.showandsave(f,"mpcexample.pdf")
