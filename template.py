# Template for nonlinear MPC using RK4 on an ODE model.
import mpctools as mpc
import numpy as np

# Define model and parameters.
Delta = .5
Nt = 20
Nx = 2
Nu = 1
def ode(x,u):
    """Continuous-time ODE model."""
    dxdt = [
        (1 - x[1]*x[1])*x[0] - x[1] + u,
        x[0],
    ]
    return np.array(dxdt)

# Create a simulator. This allows us to simulate a nonlinear plant.
vdp = mpc.DiscreteSimulator(ode, Delta, [Nx,Nu], ["x","u"])

# Then get casadi function for rk4 discretization.
ode_rk4_casadi = mpc.getCasadiFunc(ode, [Nx,Nu], ["x","u"], funcname="F",
    rk4=True, Delta=Delta, M=1)

# Define stage cost and terminal weight.
Q = np.eye(Nx)
R = np.eye(Nu)
def lfunc(x,u):
    """Standard quadratic stage cost."""
    return mpc.mtimes(x.T, Q, x) + mpc.mtimes(u.T, R, u)
l = mpc.getCasadiFunc(lfunc, [Nx,Nu], ["x","u"], funcname="l")

P = 10*Q # Terminal penalty.
def Pffunc(x):
    """Quadratic terminal penalty."""
    return 10*mpc.mtimes(x.T, P, x)
Pf = mpc.getCasadiFunc(Pffunc, [Nx], ["x"], funcname="Pf")

# Bounds on u. Here, they are all [-1, 1]
lb = {"u" : -np.ones((Nu,))}
ub = {"u" : np.ones((Nu,))}

# Make optimizers.
x0 = np.array([0,1])
N = {"x":Nx, "u":Nu, "t":Nt}
solver = mpc.nmpc(f=ode_rk4_casadi, N=N, l=l, x0=x0, Pf=Pf, lb=lb, ub=ub,
                  verbosity=0)

# Now simulate.
Nsim = 20
times = Delta*Nsim*np.linspace(0,1,Nsim+1)
x = np.zeros((Nsim+1,Nx))
x[0,:] = x0
u = np.zeros((Nsim,Nu))
for t in range(Nsim):
    # Fix initial state.
    solver.fixvar("x", 0, x[t,:])  
    
    # Solve nlp.
    solver.solve()   
    
    # Print stats.
    print("%d: %s" % (t,solver.stats["status"]))
    u[t,:] = solver.var["u",0,:]    
    
    # Simulate.
    x[t+1,:] = vdp.sim(x[t,:],u[t,:])
    
# Plots.
fig = mpc.plots.mpcplot(x,u,times)
