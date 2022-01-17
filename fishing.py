# Lotka-Volterra Fishing Problem
import numpy as np
import mpctools as mpc

# System parameters.
Nt = 40
Nx = 2
Nu = 1
Delta = 0.25
c0 = 0.4
c1 = 0.2

def ode(x, u):
    """ODE right-hand side."""
    dxdt = [
        x[0] - x[0]*x[1] - c0*x[0]*u,
        -x[1] + x[0]*x[1] - c1*x[1]*u,
    ]
    return np.array(dxdt)
f = mpc.getCasadiFunc(ode, [Nx, Nu], ["x", "u"], rk4=True, Delta=Delta, M=4)

# Initial condition, bounds, etc.
x0 = np.array([0.5, 0.7])
x = np.zeros((Nt + 1, Nx))
u = np.zeros((Nt, Nu))
x[0,:] = x0
for t in range(Nt):
    x[t + 1,:] = np.squeeze(f(x[t,:], u[t,:]))
guess = dict(x=x, u=u)
lb = dict(x=np.array([0,0]), u=np.array([0]))
ub = dict(x=np.array([2,2]), u=np.array([1]))
udiscrete = np.array([True])

# Stage cost.
def stagecost(x, u):
    """Quadratic stage cost."""
    return (x[0] - 1)**2 + (x[1] - 1)**2 + 0.1*u[0]
l = mpc.getCasadiFunc(stagecost, [Nx, Nu], ["x", "u"])

# Create controller.
N = dict(x=Nx, u=Nu, t=Nt)
cont = mpc.nmpc(f, l, N, x0, lb, ub, guess, udiscrete=udiscrete)
cont.solve()

# Plot solution.
t = Delta*np.arange(Nt + 1)
fig = mpc.plots.mpcplot(cont.vardict["x"], cont.vardict["u"], t,
                        xnames=["Prey Fish", "Predator Fish"],
                        unames=["Fishing Rate"])
mpc.plots.showandsave(fig, "fishing.pdf")
