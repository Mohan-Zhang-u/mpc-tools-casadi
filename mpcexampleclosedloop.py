# MPC for a multivariable system.
import numpy as np
import mpctools as mpc
import mpctools.plots as mpcplots

# Define continuous time model.
Acont = np.array([[0,1],[0,-1]])
Bcont = np.array([[0],[10]])
n = Acont.shape[0] # Number of states.
m = Bcont.shape[1] # Number of control elements

# Discretize.
dt = .025
Nt = 20
(A, B) = mpc.util.c2d(Acont,Bcont,dt)
def ffunc(x,u):
    """Linear discrete-time model."""
    return mpc.mtimes(A, x) + mpc.mtimes(B, u)
f = mpc.getCasadiFunc(ffunc, [n, m], ["x", "u"], "f")

# Bounds on u.
umax = 1
lb = dict(u=[-umax])
ub = dict(u=[umax])

# Define Q and R matrices.
Q = np.diag([1,0])
R = np.eye(m)
def lfunc(x,u):
    """Quadratic stage cost."""
    return mpc.mtimes(x.T, Q, x) + mpc.mtimes(u.T, R, u)
l = mpc.getCasadiFunc(lfunc, [n,m], ["x","u"], "l")

# Initial condition and sizes.
x0 = np.array([10,0])
N = {"x" : n, "u" : m, "t" : Nt}

# Now simulate.
solver = mpc.nmpc(f, l, N, x0, lb, ub, verbosity=0, isQP=True)
nsim = 100
t = np.arange(nsim+1)*dt
xcl = np.zeros((n,nsim+1))
xcl[:,0] = x0
ucl = np.zeros((m,nsim))
for k in range(nsim):
    solver.fixvar("x", 0, x0)
    sol = mpc.callSolver(solver)
    print("Iteration %d Status: %s" % (k, sol["status"]))
    xcl[:,k] = sol["x"][0,:]
    ucl[:,k] = sol["u"][0,:]
    x0 = ffunc(x0, ucl[:,k]) # Update x0.
xcl[:,nsim] = x0 # Store final state.

# Plot things. Since time is along the second dimension, we must specify
# timefirst = False.
fig = mpc.plots.mpcplot(xcl,ucl,t,np.zeros(xcl.shape),xinds=[0],
                        timefirst=False)
mpcplots.showandsave(fig, "mpcexampleclosedloop.pdf")
