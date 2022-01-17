# Periodic MPC for single-variable system.

# Imports.
import numpy as np
import scipy.signal as spsignal
import mpctools as mpc
import mpctools.plots as mpcplots

# Define optimal periodic solution.
T = 1
def xpfunc(t):
    return spsignal.sawtooth(2*np.pi/T*t + np.pi/2,.5)

# Define continuous time model.
Acont = np.array([[-1]])
Bcont = np.array([[10]])
n = Acont.shape[0] # Number of states.
m = Bcont.shape[1] # Number of control elements

# Discretize.
dt = .01
Nt = 250
t = np.arange(Nt + 1)*dt
(Adisc, Bdisc) = mpc.util.c2d(Acont,Bcont,dt)
def F(x, u):
    return mpc.mtimes(Adisc, x) + mpc.mtimes(Bdisc, u)
Fcasadi = mpc.getCasadiFunc(F, [n,m], ["x","u"], funcname="F")

# Bounds on u.
umax = 1
lb = {"u" : -umax*np.ones((Nt, m))}
ub = {"u" : umax*np.ones((Nt, m))}

# Define Q and R matrices and periodic setpoint.
R = np.eye(m)
Q = np.eye(n)
sp = {"x" : xpfunc(t)[:,np.newaxis], "u" : np.zeros((Nt, m))}
def l(x, u, xsp, usp):
    """Stage cost with setpoints."""
    dx = x - xsp
    du = u - usp
    return mpc.mtimes(dx.T, Q, dx) + mpc.mtimes(du.T, R, du)
lcasadi = mpc.getCasadiFunc(l, [n,m,n,m], ["x","u","x_sp","u_sp"],
                            funcname= "l")

# Initial condition.
x0 = np.array([-2])
N = {"x" : n, "u" : m, "t" : Nt}
funcargs = {"f" : ["x","u"], "l" : ["x","u","x_sp","u_sp"]}

# Solve linear MPC problem.
solution = mpc.callSolver(mpc.nmpc(Fcasadi, lcasadi, N, x0, lb, ub, sp=sp,
                                   funcargs=funcargs, verbosity=3))
x = solution["x"]
u = solution["u"]

# Plot things.
fig = mpcplots.mpcplot(x, u, t, sp["x"])
mpcplots.showandsave(fig,"periodicmpcexample.pdf")
