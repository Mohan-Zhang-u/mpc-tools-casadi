import mpctools as mpc
import casadi
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import linalg
from numpy import random

random.seed(927) # Seed random number generator.

doPlots = True
fullInformation = False # True for full information estimation, False for MHE.

# Problem parameters.
Nt = 10 # Horizon length
Delta = 0.25 # Time step
Nsim = 80 # Length of the simulation
tplot = np.arange(Nsim+1)*Delta

Nx = 3
Nu = 1
Ny = 1
Nw = Nx
Nv = Ny

sigma_v = 0.25 # Standard deviation of the measurements
sigma_w = 0.001 # Standard deviation for the process noise
sigma_p = 0.5 # Standard deviation for prior

# Make covariance matrices.
P = np.diag((sigma_p*np.ones((Nx,)))**2) # Covariance for prior.
Q = np.diag((sigma_w*np.ones((Nw,)))**2)
R = np.diag((sigma_v*np.ones((Nv,)))**2)

x_0 = np.array([1.0,0.0,4.0])
x0 = np.array([0.5,0.05,0.0])

# Parameters of the system
k1 = 0.5
k_1 = 0.05
k2 = 0.2
k_2 = 0.01
RT = 32.84

# Continuous-time models.
def ode(x, u, w=[0,0,0]): # We define the model with u, but there isn't one.
    """ODE for reactor species evolution."""
    [cA, cB, cC] = x[:]
    rate1 = k1*cA - k_1*cB*cC    
    rate2 = k2*cB**2 - k_2*cC
    dxdt = [
        -rate1 + w[0],
        rate1 - 2*rate2 + w[1],
        rate1 + rate2 + w[2],
    ]
    return np.array(dxdt)    

def measurement(x):
    """Pressure measurement."""
    return RT*(x[0] + x[1] + x[2])

ode_casadi = mpc.getCasadiFunc(ode,[Nx,Nu,Nw],["x","u","w"],"F")

# Make a simulator.
model = mpc.DiscreteSimulator(ode, Delta, [Nx,Nu,Nw], ["x","u","w"])    

# Convert continuous-time f to explicit discrete-time F with RK4.
F = mpc.getCasadiFunc(ode,[Nx,Nu,Nw],["x","u","w"],"F",rk4=True,Delta=Delta,M=2)
H = mpc.getCasadiFunc(measurement,[Nx],["x"],"H")

# Define stage costs.
def lfunc(w,v):
    return sigma_w**-2*mpc.mtimes(w.T, w) + sigma_v**-2*mpc.mtimes(v.T, v) 
l = mpc.getCasadiFunc(lfunc,[Nw,Nv],["w","v"],"l")
def lxfunc(x, x0bar, Pinv):
    dx = x - x0bar
    return mpc.mtimes(dx.T, Pinv, dx)
lx = mpc.getCasadiFunc(lxfunc, [Nx, Nx, (Nx, Nx)], ["x", "x0bar", "Pinv"], "lx")

# First simulate everything.
w = sigma_w*random.randn(Nsim,Nw)
v = sigma_v*random.randn(Nsim,Nv)

usim = np.zeros((Nsim,Nu)) # This is just a dummy input.
xsim = np.zeros((Nsim+1,Nx))
xsim[0,:] = x0
yclean = np.zeros((Nsim, Ny))
ysim = np.zeros((Nsim, Ny))

for t in range(Nsim):
    yclean[t,:] = measurement(xsim[t]) # Get zero-noise measurement.
    ysim[t,:] = yclean[t,:] + v[t,:] # Add noise to measurement.    
    xsim[t+1,:] = model.sim(xsim[t,:],usim[t,:],w[t,:])

# Now do estimation.
xhat_ = np.zeros((Nsim+1,Nx))
xhat = np.zeros((Nsim,Nx))
yhat = np.zeros((Nsim,Ny))
vhat = np.zeros((Nsim,Nv))
what = np.zeros((Nsim,Nw))
x0bar = x_0
xhat[0,:] = x0bar
guess = {}
totaltime = -time.time()
for t in range(Nsim):
    # Define sizes of everything.    
    N = {"x":Nx, "y":Ny, "u":Nu}
    if fullInformation:
        N["t"] = t
        tmin = 0
    else:
        N["t"] = min(t,Nt)
        tmin = max(0,t - Nt)
    tmax = t+1        
    lb = {"x":np.zeros((N["t"] + 1,Nx))}  

    # Build and call solver. If using full information or before the horizon
    # fills up, need to make a new solver. Otherwise, can reuse the old one.
    buildtime = -time.time()
    if fullInformation or t < Nt:
        solver = mpc.nmhe(f=F, h=H, u=usim[tmin:tmax-1,:],
                          y=ysim[tmin:tmax,:], l=l, N=N, lx=lx,
                          x0bar=x0bar, verbosity=0, guess=guess,
                          lb=lb, extrapar=dict(Pinv=linalg.inv(P)),
                          inferargs=True)
    else:
        solver.par["Pinv"] = linalg.inv(P)
        solver.par["x0bar"] = x0bar
        solver.par["y"] = list(ysim[tmin:tmax,:])
        solver.par["u"] = list(usim[tmin:tmax - 1,:])
    buildtime += time.time()
    solvetime = -time.time()
    sol = mpc.callSolver(solver)
    solvetime += time.time()
    print(("%3d (%5.3g s build, %5.3g s solve): %s"
           % (t, buildtime, solvetime, sol["status"])))
    if sol["status"] != "Solve_Succeeded":
        break
    xhat[t,:] = sol["x"][-1,...] # This is xhat( t  | t )
    yhat[t,:] = measurement(xhat[t,:])    
    vhat[t,:] = sol["v"][-1,...]
    if t > 0:
        what[t-1,:] = sol["w"][-1,...]
    
    # Apply model function to get xhat(t+1 | t )
    xhat_[t+1,:] = np.squeeze(F(xhat[t,:], usim[t,:], np.zeros((Nw,))))
    
    # Save stuff to use as a guess. Cycle the guess.
    guess = {}
    for k in set(["x","w","v"]).intersection(sol.keys()):
        guess[k] = sol[k].copy()
    
    # Update guess and prior if not using full information estimation.    
    if not fullInformation and t + 1 > Nt:
        for k in guess.keys():
            guess[k] = guess[k][1:,...] # Get rid of oldest measurement.
            
        # Do EKF to update prior covariance, but don't take EKF state.
        [P, x0bar, _, _] = mpc.ekf(F,H,x=sol["x"][0,...],
            u=usim[tmin,:],w=sol["w"][0,...],y=ysim[tmin,:],P=P,Q=Q,R=R)
    
     # Add final guess state for new time point.
    for k in guess.keys():
        guess[k] = np.concatenate((guess[k],guess[k][-1:,...]))

totaltime += time.time()
print("Simulation took %.5g s." % totaltime)

# Plots.
if doPlots:
    [fig, ax] = plt.subplots(nrows=2)
    xax = ax[0]
    yax = ax[1]

    # Plot states.    
    colors = ["red","blue","green"]
    species = ["A", "B", "C"]    
    for (i, (c, s)) in enumerate(zip(colors, species)):
        xax.plot(tplot, xsim[:,i], color=c, label="$c_%s$" % s)
        xax.plot(tplot[:-1], xhat[:,i], marker="o", color=c, markersize=3, 
             markeredgecolor=c, linestyle="", label=r"$\hat{c}_%s$" % s)
    mpc.plots.zoomaxis(xax, xscale=1.05, yscale=1.05)
    xax.set_ylabel("Concentration")
    xax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    
    # Plot measurements.
    yax.plot(tplot[:-1], yclean[:,0], color="black", label="$P$")
    yax.plot(tplot[:-1], yhat[:,0], marker="o", markersize=3, linestyle="",
             markeredgecolor="black", markerfacecolor="black",
             label=r"$\hat{P}$")
    yax.plot(tplot[:-1], ysim[:,0], color="gray", label="$P + v$")
    yax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    yax.set_ylabel("Pressure")
    yax.set_xlabel("Time")

    # Tweak layout and save.
    fig.subplots_adjust(left=0.1, right=0.8)
    mpc.plots.showandsave(fig,"nmheexample.pdf")
