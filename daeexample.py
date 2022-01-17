# Comparison of ODE and DAE formulations for Van der Pol oscillator.

import mpctools as mpc
import numpy as np

# Define model
Delta = 0.5
Nsim = 20
Nx = 2
Nz = 1
Nu = 1
Nc = 3
Nt = 25

# Explicit z function
def zfunc(x):
    return (1-x[1]**2)*x[0]

# ODE for VPO with and without z
def odefunc(x, u, z=[]):
    if len(z) == 0:
        z = zfunc(x)
    return [z[0]-x[1]+u[0],x[0]]

# Algebraic function for implicit solution of z
def gfunc(x,z):
    return z[0]-zfunc(x)

# Get discrete simulator for ODE
vdp = mpc.DiscreteSimulator(odefunc,Delta,[Nx,Nu],["x","u"])
# Get casadi functions for ODE and DAE
fode = mpc.getCasadiFunc(odefunc,[Nx,Nu],["x","u"],funcname="fode")
fdae = mpc.getCasadiFunc(odefunc,[Nx,Nu,Nz],["x","u","z"],funcname="fdae")
gdae = mpc.getCasadiFunc(gfunc,[Nx,Nz],["x","z"],funcname="gdae")

# Define stage cost
Q = np.eye(Nx)
R = np.eye(Nu)
def stagecost(x,u):
    return mpc.util.mtimes(x.T,Q,x) + mpc.util.mtimes(u.T,R,u)
l = mpc.getCasadiFunc(stagecost, [Nx,Nu], ["x","u"], funcname="l")

# Define terminal cost
def termcost(x):
    return 10*mpc.util.mtimes(x.T,Q,x)
Pf = mpc.getCasadiFunc(termcost, [Nx], ["x"], funcname="Pf")

# Upper and lower input bound
lb = dict(u=[-0.75])
ub = dict(u=[1])

# Size of variables
N = {"x":Nx,"u":Nu,'c':Nc,"t":Nt}
# Initial state
x0 = np.array([0,1])

solvers = {}
# ODE arguements
nmpcargs = {
        "f" : fode,
        "l" : l,
        "Pf" : Pf,
        "N" : N,
        "x0" : x0,
        "lb" : lb,
        "ub" : ub,
        "Delta" : Delta,
        "verbosity" : 3
        }
solvers["ode"] = mpc.nmpc(**nmpcargs)

# DAE arguements
N["z"] = Nz
nmpcargs["f"] = fdae
nmpcargs["g"] = gdae
nmpcargs["N"] = N
solvers["dae"] = mpc.nmpc(**nmpcargs)

# Solve and plot for initial horizon
sol_ode = mpc.callSolver(solvers["ode"])
sol_dae = mpc.callSolver(solvers["dae"])

fig_ode = mpc.plots.mpcplot(sol_ode["x"],sol_ode["u"],sol_ode["t"],
                            np.zeros(sol_ode["x"].shape),title="ode")
fig_dae = mpc.plots.mpcplot(sol_dae["x"],sol_dae["u"],sol_dae["t"],
                            np.zeros(sol_ode["x"].shape),title="dae")