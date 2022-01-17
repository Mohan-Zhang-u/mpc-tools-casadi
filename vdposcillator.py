# Control of the Van der Pol oscillator.
import mpctools as mpc
import numpy as np

# Define model and get simulator.
Delta = .5
Nsim = 20
Nx = 2
Nu = 1
def ode(x,u):
    dxdt = [(1 - x[1]*x[1])*x[0] - x[1] + u, x[0]]
    return np.array(dxdt)

# Define some stuff.
us = np.array([0])
Dumax = np.array([.5])

# Create a simulator.
vdp = mpc.DiscreteSimulator(ode, Delta, [Nx,Nu], ["x","u"])

# Then get nonlinear casadi functions and a linearization.
ode_casadi = mpc.getCasadiFunc(ode, [Nx,Nu], ["x","u"], funcname="f")
lin = mpc.util.getLinearizedModel(ode_casadi, [[0,0],[0]],
                                  ["A","B"], Delta=Delta)

# Also discretize using RK4.
ode_rk4_casadi = mpc.getCasadiFunc(ode, [Nx,Nu], ["x","u"], funcname="F",
                                   rk4=True,Delta=Delta,M=1)

# Define stage cost and terminal weight.
def lfunc(x,u):
    return mpc.mtimes(x.T,x) + mpc.mtimes(u.T,u)
l = mpc.getCasadiFunc(lfunc, [Nx,Nu], ["x","u"], funcname="l")

def Pffunc(x):
    return 10*mpc.mtimes(x.T,x)
Pf = mpc.getCasadiFunc(Pffunc, [Nx], ["x"], funcname="Pf")

# Create linear discrete-time model for comparison.
def Ffunc(x,u):
    return mpc.mtimes(lin["A"], x) + mpc.mtimes(lin["B"], u)
F = mpc.getCasadiFunc(Ffunc, [Nx,Nu], ["x","u"], funcname="F")

# Make optimizers.
x0 = np.array([0,1])
Nt = 20
commonargs = dict(
    N={"x":Nx, "u":Nu, "t":Nt},
    verbosity=0,
    l=l,
    x0=x0,
    Pf=Pf,
    lb={"u" : -.75*np.ones((Nu,)), "Du" : -Dumax},
    ub={"u" : np.ones((Nu,)), "Du" : Dumax},
    uprev=us,
)
solvers = {}
solvers["lmpc"] = mpc.nmpc(f=F,**commonargs)
solvers["nmpc"] = mpc.nmpc(f=ode_rk4_casadi,**commonargs)

# Now simulate.
times = Delta*Nsim*np.linspace(0,1,Nsim+1)
x = {}
u = {}
Du = {}
for method in solvers:
    x[method] = np.zeros((Nsim+1,Nx))
    x[method][0,:] = x0
    u[method] = np.zeros((Nsim,Nu))
    Du[method] = np.zeros((Nsim,Nu))
    for t in range(Nsim):
        solvers[method].fixvar("x",0,x[method][t,:])
        solvers[method].solve()
        print("%5s %d: %s" % (method,t,solvers[method].stats["status"]))
        u[method][t,:] = solvers[method].var["u",0,:]
        Du[method][t,:] = solvers[method].var["Du",0,:]
        solvers[method].par["u_prev",0,:] = u[method][t,:]
        x[method][t+1,:] = vdp.sim(x[method][t,:],u[method][t,:])
    fig = mpc.plots.mpcplot(x[method],u[method],times,
                            np.zeros(x[method].shape),title=method)
    mpc.plots.showandsave(fig,"vdposcillator_%s.pdf" % (method,),facecolor="none")
