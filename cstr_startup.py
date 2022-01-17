# Linear and nonlinear control of startup of a CSTR.
import mpctools as mpc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Define some parameters and then the CSTR model.
Nx = 3
Nu = 2
Nd = 1
Ny = Nx
Delta = .25
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

def ode(x,u,d):
    # Grab the states, controls, and disturbance. We would like to write
    #    
    # [c, T, h] = x[0:Nx]
    # [Tc, F] = u[0:Nu]
    # [F0] = d[0:Nd]
    #    
    # but this doesn't work in Casadi 3.0. So, we're stuck with the following:
    c = x[0]
    T = x[1]
    h = x[2]
    Tc = u[0]
    F = u[1]
    F0 = d[0]

    # Now create the ODE.
    rate = k0*c*np.exp(-E/T)
        
    dxdt = np.array([
        F0*(c0 - c)/(np.pi*r**2*h) - rate,
        F0*(T0 - T)/(np.pi*r**2*h)
            - dH/(rho*Cp)*rate
            + 2*U/(r*rho*Cp)*(Tc - T),    
        (F0 - F)/(np.pi*r**2)
    ])
    return dxdt

# Turn into casadi function and simulator.
ode_casadi = mpc.getCasadiFunc(ode,[Nx,Nu,Nd],["x","u","d"],funcname="ode")
ode_rk4_casadi = mpc.getCasadiFunc(ode,[Nx,Nu,Nd],["x","u","d"],
                                   funcname="ode_rk4",rk4=True,Delta=Delta)
cstr = mpc.DiscreteSimulator(ode, Delta, [Nx,Nu,Nd], ["x","u","d"])

# Steady-state values.
cs = .878
Ts = 324.5
hs = .659
Fs = .1
Tcs = 300
F0s = .1

# Update the steady-state values a few times to make sure they don't move.
for i in range(10):
    [cs,Ts,hs] = cstr.sim([cs,Ts,hs],[Tcs,Fs],[F0s]).tolist()
xs = np.array([cs,Ts,hs])
us = np.array([Tcs,Fs])
ds = np.array([F0s])

# Now get a linearization at this steady state.
ss = mpc.util.getLinearizedModel(ode_casadi, [xs,us,ds], ["A","B","Bp"], Delta)
A = ss["A"]
B = ss["B"]
Bp = ss["Bp"]
C = np.eye(Nx)

# Weighting matrices for controller.
Q = .5*np.diag(xs**-2)
R = 2*np.diag(us**-2)

model_casadi = mpc.getCasadiFunc(ode,[Nx,Nu,Nd],["x","u","d"],funcname="cstr")

[K, Pi] = mpc.util.dlqr(A,B,Q,R)

# Define casadi functions.
Fnonlinear = ode_rk4_casadi

def measurement(x,d):
    return x
h = mpc.getCasadiFunc(measurement,[Nx,Nd],["x","d"],funcname="h")

def linmodel(x,u,d):
    Ax = mpc.mtimes(A,x-xs) + xs
    Bu = mpc.mtimes(B,u-us)
    Bpd = mpc.mtimes(Bp,d-ds)
    return Ax + Bu + Bpd
Flinear = mpc.getCasadiFunc(linmodel,[Nx,Nu,Nd],["x","u","d"],funcname="F")

def stagecost(x,u,xsp,usp,Q,R):
    # Return deviation variables.
    dx = x - xsp
    du = u - usp
    # Calculate stage cost.
    return mpc.mtimes(dx.T,Q,dx) + mpc.mtimes(du.T,R,du)
largs = ["x","u","x_sp","u_sp","Q","R"]
l = mpc.getCasadiFunc(stagecost,[Nx,Nu,Nx,Nu,(Nx,Nx),(Nu,Nu)],largs,
                      funcname="l")

def costtogo(x,xsp):
    # Deviation variables.
    dx = x - xsp
    
    # Calculate cost to go.
    return mpc.mtimes(dx.T,Pi,dx)
Pf = mpc.getCasadiFunc(costtogo,[Nx,Nx],["x","s_xp"],funcname="Pf")

# First see what happens if we try to start up the reactor under no control.
Nsim = 100
x0 = np.array([.05*cs,.75*Ts,.5*hs])
xcl = {}
ucl = {}
xcl["uncont"] = np.zeros((Nsim+1,Nx))
xcl["uncont"][0,:] = x0
ucl["uncont"] = np.tile(us,(Nsim,1))
for t in range(Nsim):
    xcl["uncont"][t+1,:] = cstr.sim(xcl["uncont"][t,:],ucl["uncont"][t,:],ds)

# Build a solver for the linear and nonlinear models.
Nt = 15
sp = {"x" : np.tile(xs, (Nt+1,1)), "u" : np.tile(us, (Nt,1))}
xguesslin = np.zeros((Nt+1,Nx))
xguesslin[0,:] = x0
for t in range(Nt):
    xguesslin[t+1,:] = A.dot(xguesslin[t,:] - xs) + xs
guesslin = {"x" : xguesslin, "u" : np.tile(us,(Nt,1))}
guessnonlin = sp.copy()

# Control bounds.
umax = np.array([.05*Tcs,.15*Fs])
dumax = .2*umax # Maximum for rate-of-change.
bounds = dict(uub=[us + umax],ulb=[us - umax])
ub = {"u" : np.tile(us + umax, (Nt,1)), "Du" : np.tile(dumax, (Nt,1))}
lb = {"u" : np.tile(us - umax, (Nt,1)), "Du" : np.tile(-dumax, (Nt,1))}

N = {"x":Nx, "u":Nu, "p":Nd, "t":Nt, "y":Ny}
p = np.tile(ds, (Nt,1)) # Parameters for system.
nmpc_commonargs = {
    "N" : N,
    "x0" : x0,
    "lb" : lb,
    "ub" : ub,
    "p" : p,
    "verbosity" : 0,
    "Pf" : Pf,
    "l" : l,
    "sp" : sp,
    "uprev" : us,
    "funcargs" : {"l" : largs},
    "extrapar" : {"Q" : Q, "R" : R}, # In case we want to tune online.
}
solvers = {}
solvers["lmpc"] = mpc.nmpc(f=Flinear,guess=guesslin,**nmpc_commonargs)
solvers["nmpc"] = mpc.nmpc(f=Fnonlinear,guess=guessnonlin,**nmpc_commonargs)

# Also build steady-state target finders.
contVars = [0,2]
sstarg_commonargs = {
    "N" : N,
    "lb" : {"u" : np.tile(us - umax, (1,1))},
    "ub" : {"u" : np.tile(us + umax, (1,1))},
    "verbosity" : 0,
    "h" : h,
    "p" : np.array([ds]),
}
sstargs = {}
sstargs["lmpc"] = mpc.sstarg(f=Flinear,**sstarg_commonargs)
sstargs["nmpc"] = mpc.sstarg(f=Fnonlinear,**sstarg_commonargs)

# Now simulate the process under control.
for method in solvers:
    xcl[method] = np.zeros((Nsim+1,Nx))
    xcl[method][0,:] = x0
    thisx = x0    
    ucl[method] = np.zeros((Nsim,Nu))
    ysp = np.tile(xs,(Nsim+1,1))
    
    xsp = np.zeros((Nsim+1,Nx))
    usp = np.zeros((Nsim,Nu))    
    
    ysp[int(Nsim/3):int(2*Nsim/3),:] = xs*np.array([.85,.75,1.15])    
    for t in range(Nsim):
        # Figure out setpoints.
        if t == 0 or not np.all(ysp[t,:] == ysp[t-1,:]):        
            thisysp = ysp[t,:]            
            sstargs[method].fixvar("y",0,thisysp[contVars],contVars)
            sstargs[method].guess["u",0] = us
            sstargs[method].guess["x",0] = thisysp
            sstargs[method].guess["y",0] = thisysp
            sstargs[method].solve()            
            
            print("%10s %3d: %s" % ("sstarg",t,sstargs[method].stats["status"]))
            if sstargs[method].stats["status"] != "Solve_Succeeded":
                print("***Target finder failed!")
                break
            
            xsp[t,:] = np.squeeze(sstargs[method].var["x",0])
            usp[t,:] = np.squeeze(sstargs[method].var["u",0])
            
            solvers[method].par["x_sp"] = [xsp[t,:]]*(Nt + 1)
            solvers[method].par["u_sp"] = [usp[t,:]]*Nt
        
        # Fix initial condition and solve.
        solvers[method].fixvar("x",0,thisx)
        solvers[method].solve()
        print("%10s %3d: %s" % (method,t,solvers[method].stats["status"]))
        if solvers[method].stats["status"] != "Solve_Succeeded":
            print("***Solver failed!")         
            break
        else:
            solvers[method].saveguess()
            
        thisu = np.squeeze(solvers[method].var["u"][0])
        ucl[method][t,:] = thisu
        thisx = cstr.sim(thisx,thisu,ds)
        xcl[method][t+1,:] = thisx
        
        # Update previous u.
        solvers[method].par["u_prev",0] = ucl[method][t,:]

# Define plotting function.
def cstrplot(x,u,xsp=None,contVars=[],title=None,colors={},labels={},
             markers={},keys=None,bounds=None,ilegend=0):
    if keys is None:
        keys = x.keys()
    for k in keys:    
        u[k] = np.concatenate((u[k],u[k][-1:,:]))
    ylabelsx = ["$c$ (mol/L)", "$T$ (K)", "$h$ (m)"]
    ylabelsu = ["$T_c$ (K)", "$F$ (kL/min)"]
    
    gs = gridspec.GridSpec(Nx*Nu,2)    
    fig = plt.figure(figsize=(10,6),facecolor="none")
    leglines = []
    leglabels = []
    for i in range(Nx):
        ax = fig.add_subplot(gs[i*Nu:(i+1)*Nu,0])
        for k in keys:
            t = np.arange(0,x[k].shape[0])*Delta
            args = {"color":colors.get(k,"black"), "label":labels.get(k,k),
                    "marker":markers.get(k,"")}
            [line] = ax.plot(t,x[k][:,i],markeredgecolor="none",**args)
            if i == ilegend:
                leglines.append(line)
                leglabels.append(args["label"])
        if i in contVars and xsp is not None:
            ax.step(t,xsp[:,i],linestyle="--",color="black",where="post")
        ax.set_ylabel(ylabelsx[i])
        mpc.plots.zoomaxis(ax,yscale=1.1)
        mpc.plots.prettyaxesbox(ax)
        mpc.plots.prettyaxesbox(ax,
            facecolor="white",front=False)
    ax.set_xlabel("Time (min)")
    for i in range(Nu):
        ax = fig.add_subplot(gs[i*Nx:(i+1)*Nx,1])
        for k in keys:
            t = np.arange(0,u[k].shape[0])*Delta
            args = {"color":colors.get(k,"black"), "label":labels.get(k,k)}
            ax.step(t,u[k][:,i],where="post",**args)
        if bounds is not None:
            for b in set(["uub", "ulb"]).intersection(bounds):
                ax.plot(np.array([t[0],t[-1]]),np.ones((2,))*bounds[b][i],
                        '--k')
        ax.set_ylabel(ylabelsu[i])
        mpc.plots.zoomaxis(ax,yscale=1.25)
        mpc.plots.prettyaxesbox(ax)
        mpc.plots.prettyaxesbox(ax,
            facecolor="white",front=False)
    ax.set_xlabel("Time (min)")
    fig.legend(leglines,leglabels,loc="lower center",ncol=len(keys))
    fig.tight_layout(pad=.5,rect=(0,.075,1,1))
    if title is not None:
        fig.canvas.set_window_title(title)
    return fig

# Make plots.
keys = ["uncont", "lmpc", "nmpc"]
colors = {"lmpc":"blue", "nmpc":"green", "uncont":"red"}
labels = {"lmpc":"LMPC", "nmpc":"NMPC", "uncont":"Uncontrolled"}
markers = {"lmpc":"s", "nmpc":"o", "uncont":"^"}
plotbounds = dict([(k,bounds[k][0]) for k in ["ulb","uub"]])
fig = cstrplot(xcl, ucl, ysp, colors=colors, contVars=contVars, labels=labels,
               keys=keys, markers={}, bounds=plotbounds, ilegend=2)
mpc.plots.showandsave(fig,"cstr_startup.pdf",facecolor="none")
