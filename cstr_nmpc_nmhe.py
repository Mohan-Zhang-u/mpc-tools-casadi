# Example 1.11 from Rawlings and Mayne with linear and nonlinear control.
import mpctools as mpc
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time

# Decide whether to use casadi SX objects or MX. Generally, SX is going to be
# faster, and that looks like the case for this problem.
useCasadiSX = True
useMeasuredState = False
useSstargObjective = True

# Define some parameters and then the CSTR model.
Nx = 3
Nu = 2
Nd = 1
Ny = Nx
Nid = Ny # Number of integrating disturbances.
Nw = Nx + Nid # Noise on augmented state.
Nv = Ny # Noise on outputs.
Delta = 1
eps = 1e-6 # Use this as a small number.

T0 = 350
c0 = 1
r = .219
k0 = 7.2e10
E = 8750
U = 54.94
rho = 1000
Cp = .239
dH = -5e4

def cstrmodel(c,T,h,Tc,F,F0):
    # ODE for CSTR.
    rate = k0*c*np.exp(-E/T)
        
    dxdt = np.array([
        F0*(c0 - c)/(np.pi*r**2*h) - rate,
        F0*(T0 - T)/(np.pi*r**2*h)
            - dH/(rho*Cp)*rate
            + 2*U/(r*rho*Cp)*(Tc - T),    
        (F0 - F)/(np.pi*r**2)
    ])
    return dxdt

# Steady-state values.
cs = .878
Ts = 324.5
hs = .659
Fs = .1
Tcs = 300
F0s = .1

def ode(x,u,d):
    # Grab the states, controls, and disturbance. We would like to write   
    [c, T, h] = x[0:Nx]
    [Tc, F] = u[0:Nu]
    [F0] = d[0:Nd]
    return cstrmodel(c,T,h,Tc,F,F0)

# Turn into casadi function and simulator.
ode_casadi = mpc.getCasadiFunc(ode,[Nx,Nu,Nd],["x","u","d"],"ode")
ode_rk4_casadi = mpc.getCasadiFunc(ode,[Nx,Nu,Nd],["x","u","d"],
                                   "ode_rk4",rk4=True,Delta=Delta,M=1)
ode_sstarg_casadi = mpc.getCasadiFunc(ode,[Nx+Nid,Nu,Nd],["xhat","u","d"],"ode")

cstr = mpc.DiscreteSimulator(ode, Delta, [Nx,Nu,Nd], ["x","u","d"])

# Update the steady-state values a few times to make sure they don't move.
for i in range(10):
    [cs,Ts,hs] = cstr.sim([cs,Ts,hs],[Tcs,Fs],[F0s]).tolist()
xs = np.array([cs,Ts,hs])
xaugs = np.concatenate((xs,np.zeros((Nid,))))
us = np.array([Tcs,Fs])
ds = np.array([F0s])
ps = np.concatenate((ds,xs,us))

# Define augmented model for state estimation. We put output disturbances on
# c and h, and an input disturbance on F. Although h is an integrator, we can
# put an output disturbance on h because of the input disturbance on F.

# We need to define two of these because Ipopt isn't smart enough to throw out
# the 0 = 0 equality constraints. ode_disturbance only gives dx/dt for the
# actual states, and ode_augmented appends the zeros so that dx/dt is given for
# all of the states.    
def ode_disturbance(x,u,d=ds):
    # Grab states, estimated disturbances, controls, and actual disturbance.
    [c, T, h] = x[0:Nx]
    dhat = x[Nx:Nx+Nid]
    [Tc, F] = u[0:Nu]
    [F0] = d[0:Nd]
    
    dxdt = cstrmodel(c,T,h,Tc,F+dhat[2],F0)
    return dxdt
def ode_augmented(x,u,d=ds):
    # Need to add extra zeros for derivative of disturbance states.
    dxdt = np.concatenate((ode_disturbance(x,u,d),np.zeros((Nid,))))
    return dxdt
cstraug = mpc.DiscreteSimulator(ode_augmented, Delta,
                                [Nx+Nid,Nu,Nd], ["xaug","u","d"])
def measurement(x,d=ds):
    [c, T, h] = x[0:Nx]
    dhat = x[Nx:Nx+Nid]
    return np.array([c + dhat[0], T, h + dhat[1]])
ys = measurement(xaugs)

# Turn into casadi functions.
ode_disturbance_casadi = mpc.getCasadiFunc(ode_disturbance,
    [Nx+Nid,Nu,Nd],["xaug","u","d"],"ode_disturbance")
ode_augmented_casadi = mpc.getCasadiFunc(ode_augmented,
    [Nx+Nid,Nu,Nd],["xaug","u","d"],"ode_augmented")
ode_augmented_rk4_casadi = mpc.getCasadiFunc(ode_augmented,
    [Nx+Nid,Nu,Nd],["xaug","u","d"],"ode_augmented_rk4",
    rk4=True,Delta=Delta,M=2)
measurement_casadi = mpc.getCasadiFunc(measurement,
    [Nx+Nid,Nd],["xaug","d"],"measurement")

# Weighting matrices for controller.
Q = .5*np.diag(xs**-2)
R = 2*np.diag(us**-2)

# Now get a linearization at this steady state and calculate Riccati cost-to-go.
ss = mpc.util.getLinearizedModel(ode_casadi, [xs,us,ds], ["A","B","Bp"], Delta)
A = ss["A"]
B = ss["B"]
Bp = ss["Bp"]
C = np.eye(Nx)

[K, Pi] = mpc.util.dlqr(A,B,Q,R)

def stagecost(x,u,xsp,usp,Deltau):
    # Return deviation variables.
    dx = x[:Nx] - xsp[:Nx]
    du = u - usp
    
    # Calculate stage cost.
    return (mpc.mtimes(dx.T,Q,dx) + 0.1*mpc.mtimes(du.T,R,du)
        + mpc.mtimes(Deltau.T,R,Deltau))

largs = ["x","u","x_sp","u_sp","Du"]
l = mpc.getCasadiFunc(stagecost,
    [Nx+Nid,Nu,Nx+Nid,Nu,Nu], largs, funcname="l")

def costtogo(x,xsp):
    # Deviation variables.
    dx = x[:Nx] - xsp[:Nx]
    
    # Calculate cost to go.
    return mpc.mtimes(dx.T,Pi,dx)
Pf = mpc.getCasadiFunc(costtogo, [Nx+Nid,Nx+Nid], ["x","s_xp"],
                       funcname="Pf")

# Build augmented estimator matrices.
Qw = eps*np.eye(Nx + Nid)
Qw[-1,-1] = 1
Rv = eps*np.diag(xs**2)
Qwinv = linalg.inv(Qw)
Rvinv = linalg.inv(Rv)

# Define stage costs for estimator.
def lest(w,v):
    return mpc.mtimes(w.T,Qwinv,w) + mpc.mtimes(v.T,Rvinv,v) 
lest = mpc.getCasadiFunc(lest, [Nw,Nv], ["w","v"], "l")

# Don't use a prior.
lxest = None
x0bar = None

# Check if the augmented system is detectable. (Rawlings and Mayne, Lemma 1.8)
Aaug = mpc.util.getLinearizedModel(ode_augmented_casadi,[xaugs, us, ds],
                               ["A","B","Bp"], Delta)["A"]
Caug = mpc.util.getLinearizedModel(measurement_casadi,[xaugs, ds],
                               ["C","Cp"])["C"]
Oaug = np.vstack((np.eye(Nx,Nx+Nid) - Aaug[:Nx,:], Caug))
svds = linalg.svdvals(Oaug)
rank = sum(svds > 1e-8)
if rank < Nx + Nid:
    print("***Warning: augmented system is not detectable!")

# Now simulate things.
Nsim = 51
t = np.arange(Nsim)*Delta
starttime = time.clock()
x = np.zeros((Nsim,Nx))
x[0,:] = xs # Start at steady state.    

u = np.zeros((Nsim,Nu))
u[0,:] = us # Start at steady state.

usp = np.zeros((Nsim,Nu))
xaugsp = np.zeros((Nsim,Nx+Nid))
y = np.zeros((Nsim,Ny))
err = y.copy()
v = y.copy()

# xhatm is xhat( k | k-1 ). xhat is xhat( k | k ).
xhatm = np.zeros((Nsim,Nx+Nid))
xhatm[0,:] = xaugs # Start with estimate at steaty state.
xhat = np.zeros((Nsim,Nx+Nid))
xhat[0,:] = xaugs

# Pick disturbance, setpoint, and initial condition.
d = np.zeros((Nsim,Nd))
d[:,0] = (t >= 10)*(t <= 30)*.1*F0s
d += ds

ysp = np.tile(xs, (Nsim,1)) # Specify setpoint.
contVars = [0,2] # Concentration and height.

# Make NMPC solver.
Nt = 5
ubounds = np.array([.05*Tcs, .5*Fs])
Dubounds = .05*ubounds
lb = {"u" : us - ubounds, "Du" : -Dubounds}
ub = {"u" : us + ubounds, "Du" : Dubounds}

N = {"x" : Nx+Nid, "u" : Nu, "p" : Nd, "t" : Nt}
p = ds
sp = {"x" : xaugs, "u" : us}
guess = sp.copy()
x0 = xs
xaug0 = xaugs
nmpcargs = {
    "f" : ode_augmented_rk4_casadi,
    "l" : l,
    "funcargs" : {"l" : largs},
    "N" : N,
    "x0" : xaug0,
    "uprev" : us,
    "lb" : lb,
    "ub" : ub,
    "guess" : guess,
    "Pf" : Pf,
    "sp" : sp,
    "p" : p,
    "verbosity" : 0,
    "timelimit" : 60,
    "casaditype" : "SX" if useCasadiSX else "MX",
}
controller = mpc.nmpc(**nmpcargs)

# Make NMHE solver.
Nmhe = 5
nmheargs = {
    "f" : ode_augmented_rk4_casadi,
    "wAdditive" : True,
    "h" : measurement_casadi,
    "u" : us,
    "y" : ys,
    "l" : lest,
    "N" : {"x" : Nx + Nid, "u" : Nu, "y" : Ny, "p" : Nd, "t" : Nmhe},
    "lx" : lxest,
    "x0bar" : x0bar,
    "p" : ds,
    "verbosity" : 0,
    "guess" : {"x" : xaugs, "y" : ys, "u" : us},
    "timelimit" : 5,
    "casaditype" : "SX" if useCasadiSX else "MX",                       
}
estimator = mpc.nmhe(**nmheargs)

# Declare ydata and udata. Note that it would make the most sense to declare
# these using collection.deques since we're always popping the left element or
# appending a new element, but for these sizes, we can just use a list without
# any noticable slowdown.
ydata = [ys]*Nmhe
udata = [us]*(Nmhe-1)

# Make steady-state target selector.
def sstargobj(y,y_sp,u,u_sp,Q,R):
    dy = y - y_sp
    du = u - u_sp
    return mpc.mtimes(dy.T,Q,dy) + mpc.mtimes(du.T,R,du)
if useSstargObjective:
    phiargs = ["y","y_sp","u","u_sp","Q","R"]
    phi = mpc.getCasadiFunc(sstargobj, [Ny,Ny,Nu,Nu,(Ny,Ny),(Nu,Nu)],
                            phiargs)
else:
    phiargs = None
    phi = None

# Weighting matrices.
Rss = np.zeros((Nu,Nu))
Qss = np.zeros((Ny,Ny))
Qss[contVars,contVars] = 1 # Only care about controlled variables.

sstargargs = {
    "f" : ode_disturbance_casadi,
    "h" : measurement_casadi,
    "lb" : {"u" : us - ubounds},
    "ub" : {"u" : us + ubounds},
    "guess" : {
        "u" : us,
        "x" : np.concatenate((xs, np.zeros((Nid,)))),
        "y" : xs,
    },
    "p" : ds, # Parameters for system.
    "N" : {"x" : Nx + Nid, "u" : Nu, "y" : Ny, "p" : Nd, "f" : Nx},
    "phi" : phi,
    "funcargs" : {"phi" : phiargs},
    "extrapar" : {"R" : Rss, "Q" : Qss, "y_sp" : ys, "u_sp" : us},
    "verbosity" : 0,
    "discretef" : False,
    "casaditype" : "SX" if useCasadiSX else "MX",
}
targetfinder = mpc.sstarg(**sstargargs)

for n in range(1,Nsim):
    # Simulate with nonilnear model.
    try:
        x[n,:] = cstr.sim(x[n-1,:], u[n-1,:], d[n-1,:])
    except:
        print("***Error during simulation!")
        break
    udata.append(u[n-1,:]) # Store latest control move.
    
    # Advance state estimate. Use disturbance as modeled.
    xhatm[n,:] = cstraug.sim(xhat[n-1,:], u[n-1,:], ds)
    print("")    
    
    # Take plant measurement.
    y[n,:] = measurement(np.concatenate(
        (x[n,:],np.zeros((Nid,))))) + v[n,:]
    ydata.append(y[n,:])
    
    # Update state estimate with measurement.
    err[n,:] = y[n,:] - measurement(xhatm[n,:])
    
    print("(%3d) " % (n,), end="") 
    
    # Handle disturbance.
    if useMeasuredState:
        # Just directly measure state and get the correct disturbance.
        x0hat = x[n,:]
        xhat[n,:Nx] = x0hat # Estimate is exact.
        controller.par["p"] = [d[n,:]]*Nt
        targetfinder.par["p",0] = d[n,:]
    else:
        # Do Nonlinear MHE.
        estimator.par["y"] = ydata
        estimator.par["u"] = udata
        estimator.solve()
        estsol = mpc.util.casadiStruct2numpyDict(estimator.var)
        
        print("Estimator: %s, " % (estimator.stats["status"],), end="")
        xhat[n,:] = estsol["x"][-1,:] # Update our guess.
        
        estimator.saveguess()        
        
        dguess = ds
        targetfinder.par["p",0] = ds                
        
    # Use nonlinear steady-state target selector.
    x0hat = xhat[n,:]
     
    # Pick setpoint for augmented state and fix the augmented states.            
    xtarget = np.concatenate((ysp[n,:],xhat[n,Nx:]))
    targetfinder.guess["x",0] = xtarget
    targetfinder.fixvar("x",0,xhat[n,Nx:],range(Nx,Nx+Nid))
    if useSstargObjective:
        targetfinder.par["y_sp"] = ysp[n,:]
    else:
        targetfinder.fixvar("y",0,ysp[n,contVars],contVars)
        
    
    uguess = u[n-1,:]            
    targetfinder.guess["u",0] = uguess
    targetfinder.solve()
    
    xaugsp[n,:] = np.squeeze(targetfinder.var["x",0,:])
    usp[n,:] = np.squeeze(targetfinder.var["u",0,:])

    print("Target: %s, " % (targetfinder.stats["status"],), end="") 
    if targetfinder.stats["status"] != "Solve_Succeeded":
        print("*** Optimization failed. Enter debug mode. ***")        
        mpc.keyboard()
        break

    # Now use nonlinear MPC controller.
    controller.par["x_sp"] = [xaugsp[n,:]]*(Nt + 1)
    controller.par["u_sp"] = [usp[n,:]]*Nt
    controller.par["u_prev"] = [u[n-1,:]]
    controller.fixvar("x",0,x0hat)            
    controller.solve()
    print("Controller: %s, " % (controller.stats["status"],), end="") 
    
    controller.saveguess()
    u[n,:] = np.squeeze(controller.var["u",0])
    
    # Get rid of oldest y and u data.
    ydata.pop(0)
    udata.pop(0)    
    
endtime = time.clock()
print("\n\nNonlinear Took %.5g s." % (endtime - starttime,))

# *****
# Plots
# *****

# Define plotting function.
def cstrplot(x,u,ysp=None,contVars=[],title=None):
    u = np.concatenate((u,u[-1:,:]))
    t = np.arange(0,x.shape[0])*Delta
    ylabelsx = ["$c$ (mol/L)", "$T$ (K)", "$h$ (m)"]
    ylabelsu = ["$T_c$ (K)", "$F$ (kL/min)"]
    
    gs = gridspec.GridSpec(Nx*Nu,2)    
    
    fig = plt.figure(figsize=(10,6))
    for i in range(Nx):
        ax = fig.add_subplot(gs[i*Nu:(i+1)*Nu,0])
        ax.plot(t,x[:,i],'-ok')
        if i in contVars:
            ax.step(t,ysp[:,i],'-r',where="post")
        ax.set_ylabel(ylabelsx[i])
        mpc.plots.zoomaxis(ax,yscale=1.1)
    ax.set_xlabel("Time (min)")
    for i in range(Nu):
        ax = fig.add_subplot(gs[i*Nx:(i+1)*Nx,1])
        ax.step(t,u[:,i],'-k',where="post")
        ax.set_ylabel(ylabelsu[i])
        mpc.plots.zoomaxis(ax,yscale=1.25)
    ax.set_xlabel("Time (min)")
    fig.tight_layout(pad=.5)
    if title is not None:
        fig.canvas.set_window_title(title)
    return fig


estfig = cstrplot(xhatm[:,:Nx],u[:-1,:],ysp=None,contVars=[],title="Estimated")
mpc.plots.showandsave(estfig,"cstr_estimated.pdf")
actfig = cstrplot(x,u[:-1,:],ysp=ysp,contVars=contVars,title="Actual")
mpc.plots.showandsave(actfig,"cstr_actual.pdf")

