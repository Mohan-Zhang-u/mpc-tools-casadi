import numpy as np
import mpctools as mpc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # analysis:ignore

# Aircraft navigation example.
useCollocation = False
Nx = 5
Nu = 3
Nc = 4 # Number of collocation points.
Nt = 12
Nsim = 50
Delta = 5

# Aircraft parameters.
g = 9.8
K = 1
m = 1000

# Pick wind speed function.
def wind(x,y,z):
    v = [
        -5,
        5,
        0,
    ]
    return np.array(v)

# 5 States
# - x,y,z : 3D spatial coordinates of plane.
# -  V    : Speed of plane
# - psi   : Heading angle (in xy plane)
# 3 Inputs
# - gamma : Path angle (between velocity and xy plane)
# -  phi  : Bank angle (roughly rotation about roll axis)
# -   T   : Engine thrust
def ode(x,u):
    """Continuous-time ODE model."""
    [x, y, z, V, psi] = x[:]
    [gam, phi, T] = u[:]
    [wx, wy, wz] = wind(x,u,z)[:]
    
    dxdt = [
        V*np.cos(psi)*np.cos(gam) + wx,
        V*np.sin(psi)*np.cos(gam) + wy,
        V*np.sin(gam) + wz,
        -K/m*V**2 - g*np.sin(gam) + T/m,
        g/V*np.tan(phi),    
    ]
    return np.array(dxdt)
if useCollocation:
    f = mpc.getCasadiFunc(ode, [Nx,Nu], ["x","u"])
else:
    f = mpc.getCasadiFunc(ode, [Nx,Nu], ["x","u"], rk4=True, Delta=Delta, M=1)
plane = mpc.DiscreteSimulator(ode, Delta, [Nx,Nu], ["x","u"])

x0 = np.array([1000,1000,500,50,0])
u0 = np.array([0,0,50])

xlb = np.array([-np.inf,-np.inf,0,15,-np.pi])
xub = np.array([np.inf,np.inf,np.inf,100,np.pi])

ulb = np.array([-np.pi/4,-np.pi/8,0])
uub = np.array([np.pi/4,np.pi/8,1000])

lb = {
    "u": np.tile(ulb,(Nt,1)),
    "x": np.tile(xlb,(Nt+1,1)),
}
ub = {
    "u" : np.tile(uub,(Nt,1)),
    "x" : np.tile(xub,(Nt+1,1)),
}
guess = {"x" : np.tile(x0[np.newaxis,:], (Nt+1,1)),
         "xc" : np.tile(x0[np.newaxis,:,np.newaxis],(Nt,1,Nc))}
def lfunc(x, u, xsp=None, usp=None):
    if xsp is None:
        xsp = np.zeros(x.shape)
    dx = x[0:3] - xsp[0:3]
    return mpc.mtimes(dx.T, dx)
ltarg = mpc.getCasadiFunc(lfunc, [Nx,Nu], ["x","u"])
l = mpc.getCasadiFunc(lfunc, [Nx,Nu,Nx,Nu], ["x","u","x_sp","u_sp"])

# First simulate to get a guess.
x = np.zeros((Nt+1,Nx))
x[0,:] = x0
u = np.zeros((Nt,Nu))
for t in range(Nt):
    u[t,:] = u0
    x[t+1,:] = plane.sim(x[t,:], u[t,:])
guess["x"] = x
guess["u"] = u

# Now build controller.
N = {"x":Nx, "u":Nu, "t":Nt}
if useCollocation:
    N["c"] = Nc
kwargs = dict(f=f, N=N, lb=lb, ub=ub, guess=guess, Delta=Delta, verbosity=0)
targetfinder = mpc.nmpc(l=ltarg, periodic=True, **kwargs)

# Find periodic trajectory and set as setpoint.
targetfinder.solve()
xtarg = targetfinder.vardict["x"]
utarg = targetfinder.vardict["u"]

sp = dict(x=xtarg, u=utarg)
controller = mpc.nmpc(l=l, x0=x0, sp=sp, **kwargs)

# Simulate closed-loop.
x = np.nan*np.ones((Nsim+1,Nx))
x[0,:] = x0
u = np.zeros((Nsim,Nu))
traj = []
for t in range(Nsim):
    # Solve OCP
    controller.fixvar("x",0,x[t,:])
    controller.solve()
    print("%5d: %20s" % (t,controller.stats["status"]))
    controller.saveguess()
    
    # Save predicted (x,y,z) trajectory.
    traj.append(mpc.util.listcatfirstdim(controller.var["x"]))    
    
    # Now save control input.
    u[t,:] = np.squeeze(controller.var["u",0])
    x[t+1,:] = plane.sim(x[t,:], u[t,:])

    # Cycle setpoint.
    for i in range(t, t + Nt):
        i = i % Nt
        controller.par["u_sp",i] = utarg[i,:]
        controller.par["x_sp",i] = xtarg[i,:]
    controller.par["x_sp",-1] = xtarg[t % Nt,:]   

# Plot closed loop trajectory with open-loop trajectories.
t = Delta*np.arange(Nsim+1)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="3d")
for p in traj:
    ax.plot(p[:,0], p[:,1], p[:,2], color="red", linewidth=.5, alpha=.5)
ax.plot(x[:,0], x[:,1], x[:,2], color="blue", linewidth=2)
ax.plot(xtarg[:,0], xtarg[:,1], xtarg[:,2], color="green", linewidth=4,
        alpha=0.5)
ax.set_xlabel("$x$ (m)")
ax.set_ylabel("$y$ (m)")
ax.set_zlabel("$z$ (m)")
fig.tight_layout(pad=.5)
mpc.plots.showandsave(fig, "airplane.pdf")
