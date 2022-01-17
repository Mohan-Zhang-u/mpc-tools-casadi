# Compares various model forms for a linear problem.

# Imports.
import numpy as np
import mpctools as mpc

verb = 2

# Problem parameters.
Delta = .1
Nt = 50
t = np.arange(Nt+1)*Delta

Nx = 2
Nu = 2
Nd = 0

xsp = np.zeros((Nt+1,Nx))

# We're going to solve the same problem using multiple methods:
# starting with an exact discrete-time model, and nonlinear mpc starting from
# a continuous-time model. In theory, the results should be identical.

Acont = np.array([[-1,1],[0,-1]])
Bcont = np.eye(2)

(A,B) = mpc.util.c2d(Acont,Bcont,Delta)

# Bounds, initial condition, and stage costs.
ulb = np.array([-1,-1])
uub = np.array([1,1])
lb = {"u" : np.tile(ulb,(Nt,1))}
ub = {"u" : np.tile(uub,(Nt,1))}
bounds = {"uub" : [uub], "ulb" : [ulb]}
x0 = [10,10]

Q = np.eye(Nx)
R = np.eye(Nx)

def lfunc(x,u):
    return mpc.mtimes(x.T,Q,x) + mpc.mtimes(u.T,R,u)
l = mpc.getCasadiFunc(lfunc,[Nx,Nu],["x","u"],"l")

def Pffunc(x):
    return mpc.mtimes(x.T,Q,x)
Pf = mpc.getCasadiFunc(Pffunc,[Nx],["x"],"Pf")

# Discrete-time example
def Fdiscrete(x,u):
    return mpc.mtimes(A,x) + mpc.mtimes(B,u)
F = mpc.getCasadiFunc(Fdiscrete,[Nx,Nu],["x","u"],"F")

print("\n=Exact Discretization=")
N = {"x":Nx, "u":Nu, "t":Nt}
opt_dnmpc = mpc.callSolver(mpc.nmpc(F, l, N, x0, lb, ub, Pf=Pf,
                                    verbosity=verb))
fig_dnmpc = mpc.plots.mpcplot(opt_dnmpc["x"],opt_dnmpc["u"],t,xsp,
                              xinds=[0,1])
fig_dnmpc.canvas.set_window_title("Discrete-time NMPC")
mpc.plots.showandsave(fig_dnmpc,"mpcmodelcomparison_discretized.pdf")

# Continuous time interfaces in nmpc.
def fcontinuous(x,u):
    return mpc.mtimes(Acont,x) + mpc.mtimes(Bcont,u)
f = mpc.getCasadiFunc(fcontinuous,[Nx,Nu],["x","u"],"f")

Mrk4 = 5
Mcolloc = 5

F_rk4 = mpc.getCasadiFunc(fcontinuous, [Nx,Nu], ["x","u"], "F_rk4", rk4=True,
                          Delta=Delta, M=Mrk4)    

print("\n=RK4 Discretization=")
opt_crk4nmpc = mpc.callSolver(mpc.nmpc(F_rk4, l, N, x0, lb, ub, Pf=Pf,
                                       verbosity=verb))
fig_crk4nmpc = mpc.plots.mpcplot(opt_crk4nmpc["x"],opt_crk4nmpc["u"],t,
                                 xsp,xinds=[0,1])
fig_crk4nmpc.canvas.set_window_title("Continuous-time NMPC (RK4)")
mpc.plots.showandsave(fig_crk4nmpc,"mpcmodelcomparison_rk4.pdf")

print("\n=Collocation Discretization=")
Ncolloc = N.copy()
Ncolloc["c"] = Mcolloc
opt_ccollocnmpc = mpc.callSolver(mpc.nmpc(f, l, Ncolloc, x0, lb, ub, Pf=Pf,
                                          verbosity=verb, Delta=Delta))
fig_ccollocnmpc = mpc.plots.mpcplot(opt_ccollocnmpc["x"],
                                    opt_ccollocnmpc["u"],t,xsp,xinds=[0,1])
fig_ccollocnmpc.canvas.set_window_title("Continuous-time NMPC (Collocation)")
mpc.plots.showandsave(fig_ccollocnmpc,"mpcmodelcomparison_collocation.pdf")

# Discrete-time but with Casadi's integrators. This is slow, but it may be
# necessary if your ODE is difficult to discretize.
print("\n=Casadi Integrator Discretization=")
F_integrator = mpc.tools.getCasadiIntegrator(fcontinuous,Delta,[Nx,Nu],
                                             ["x","u"],"int_f")
opt_integrator = mpc.callSolver(mpc.nmpc(F_integrator, l, N, x0, lb, ub, Pf=Pf,
                                         verbosity=verb, casaditype="MX"))
fig_integrator = mpc.plots.mpcplot(opt_integrator["x"],
                                   opt_integrator["u"],t,xsp,xinds=[0,1])
fig_integrator.canvas.set_window_title("NMPC with Casadi Integrators")
mpc.plots.showandsave(fig_integrator,"mpcmodelcomparison_casadiintegrator.pdf")
