# Linear and nonlinear control of a system.
import mpctools as mpc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Some constants.
Delta = .1
Nx = 2
Nu = 2

# Define model and get simulator.
def ode(x,u):
    return np.array([-x[0] - (1+u[0])*x[1], (1+u[0])*x[0] + x[1] + u[1]])

# Create a simulator.
model = mpc.DiscreteSimulator(ode, Delta, [Nx,Nu], ["x","u"])

# Then get nonlinear casadi functions and the linearization.
ode_casadi = mpc.getCasadiFunc(ode, [Nx,Nu], ["x","u"], funcname="f")
xss = np.zeros((Nx,)) # Define steady-state solution.
uss = np.zeros((Nu,))
lin = mpc.util.getLinearizedModel(ode_casadi,[xss,uss],["A","B"],Delta=Delta)

# Define stage cost and terminal weight.
def lfunc(x,u):
    return 100*mpc.mtimes(x.T,x) + mpc.mtimes(u.T,u)
l = mpc.getCasadiFunc(lfunc, [Nx,Nu], ["x","u"], funcname="l")

def Pffunc(x):
    return 1000*mpc.mtimes(x.T,x)
Pf = mpc.getCasadiFunc(Pffunc, [Nx], ["x"], funcname="Pf")

# Create linear discrete-time model for comparison.
A = lin["A"]
B = lin["B"]
def Ffunc(x,u):
    return mpc.mtimes(A,x) + mpc.mtimes(B,u)
F = mpc.getCasadiFunc(Ffunc, [Nx,Nu], ["x","u"], funcname="F")

# Make optimizers. Note that the linear and nonlinear solvers have some common
# arguments, so we collect those below.
x0 = np.array([2,2])
Nt = 25
commonargs = dict(
    verbosity=0,
    l=l,
    x0=x0,
    Pf=Pf,
    lb={"u" : -np.ones((Nt,Nu))},
    ub={"u" : np.ones((Nt,Nu))},
)
Nlin = {"t":Nt, "x":Nx, "u":Nu}
Nnonlin = Nlin.copy()
Nnonlin["c"] = 2 # Use collocation to discretize.

solvers = {}
solvers["LMPC"] = mpc.nmpc(f=F,N=Nlin,**commonargs)
solvers["NMPC"] = mpc.nmpc(f=ode_casadi,N=Nnonlin,Delta=Delta,**commonargs)

# Now simulate.
Nsim = 100
x = {}
u = {}
for method in solvers:
    x[method] = np.zeros((Nsim+1,Nx))
    x[method][0,:] = x0
    u[method] = np.zeros((Nsim,Nu))
    for t in range(Nsim):
        # Update initial condition and solve OCP.
        solvers[method].fixvar("x",0,x[method][t,:])
        solvers[method].solve()
        
        # Print status and make sure solver didn't fail.        
        print("%5s %d: %s" % (method,t,solvers[method].stats["status"]))
        if solvers[method].stats["status"] != "Solve_Succeeded":
            break
        else:
            solvers[method].saveguess()
        u[method][t,:] = np.squeeze(solvers[method].var["u",0])
        x[method][t+1,:] = model.sim(x[method][t,:],u[method][t,:])
        
        # We can stop early if we are already close to the origin.
        if np.sum(x[method][t+1,:]**2) < 1e-4:
            print("%s at origin after %d iterations." % (method,t))
            break

# Prepare some plots.
fig = plt.figure(figsize=(12,6))
gs = gridspec.GridSpec(max(Nx,Nu),4)

phaseax = fig.add_subplot(gs[:,2:])
phaseax.set_xlabel("$x_1$")
phaseax.set_ylabel("$x_2$",rotation=0)
phaseax.set_title("Phase Space")
phaseax.axvline(color="grey")
phaseax.axhline(color="grey")

timexax = []
for i in range(Nx):
    timexax.append(fig.add_subplot(gs[i,0]))
    timexax[-1].set_ylabel("$x_{%d}$" % (i+1,),rotation=0)
timexax[-1].set_xlabel("Time")
timexax[0].set_title("$x$ Time Series")
timeuax = []
for i in range(Nu):
    timeuax.append(fig.add_subplot(gs[i,1]))
    timeuax[-1].set_ylabel("$u_{%d}$" % (i + 1,),rotation=0)
timeuax[0].set_title("$u$ Time Series")
timeuax[-1].set_xlabel("Time")

colors = {"LMPC":"red", "NMPC":"green"}
tplot = Nsim*Delta*np.linspace(0,1,Nsim+1)

for method in x:
    # Plot in time.
    uplot = np.vstack((u[method],u[method][-1,:])) # Fix t/u size mismatch.
    for i in range(Nu):
        timeuax[i].step(tplot,uplot[:,i],color=colors.get(method,"k"))
    for i in range(Nx):
        timexax[i].plot(tplot,x[method][:,i],color=colors.get(method,"k"))
    
    # Plot in phase space.
    phaseax.plot(x[method][:,0],x[method][:,1],'-k',label=method,color=colors.get(method,"k"))

# Some beautification.
phaseax.legend(loc="upper right")
for ax in timexax + timeuax:
    mpc.plots.zoomaxis(ax,yscale=1.1)
fig.tight_layout(pad=.5)        
mpc.plots.showandsave(fig,"nmpc_exercise_full.pdf")        
