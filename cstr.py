# Example 1.11 from Rawlings and Mayne.
import mpctools as mpc
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import gridspec

#<<ENDCHUNK>>

# Define some parameters and then the CSTR model.
Nx = 3
Nu = 2
Nd = 1
Ny = Nx
Delta = 1
eps = 1e-5 # Use this as a small number.

T0 = 350
c0 = 1
r = .219
k0 = 7.2e10
E = 8750
U = 54.94
rho = 1000
Cp = .239
dH = -5e4

#<<ENDCHUNK>>

def ode(x,u,d):
    # Grab the states, controls, and disturbance.
    [c, T, h] = x[0:Nx]
    [Tc, F] = u[0:Nu]
    [F0] = d[0:Nd]

    # Now create the ODE.
    rate = k0*c*np.exp(-E/T)
        
    dxdt = [
        F0*(c0 - c)/(np.pi*r**2*h) - rate,
        F0*(T0 - T)/(np.pi*r**2*h)
            - dH/(rho*Cp)*rate
            + 2*U/(r*rho*Cp)*(Tc - T),    
        (F0 - F)/(np.pi*r**2)
    ]
    return np.array(dxdt)

# Turn into casadi function and simulator.
ode_casadi = mpc.getCasadiFunc(ode,
    [Nx,Nu,Nd],["x","u","d"],funcname="ode")
cstr = mpc.DiscreteSimulator(ode, Delta,
    [Nx,Nu,Nd], ["x","u","d"])

# We don't need to take any derivatives by hand
# because Casadi can do that.

#<<ENDCHUNK>>

# Steady-state values.
cs = .878
Ts = 324.5
hs = .659
Fs = .1
Tcs = 300
F0s = .1

# Update the steady-state values a few times to make
# sure they don't move.
for i in range(10):
    [cs,Ts,hs] = cstr.sim([cs,Ts,hs],[Tcs,Fs],
        [F0s]).tolist()
xs = np.array([cs,Ts,hs])
us = np.array([Tcs,Fs])
ds = np.array([F0s])

#<<ENDCHUNK>>

# Now get a linearization at this steady state.
ss = mpc.util.getLinearizedModel(ode_casadi, 
    [xs,us,ds], ["A","B","Bp"], Delta)
A = ss["A"]
B = ss["B"]
Bp = ss["Bp"]
C = np.eye(Nx)

#<<ENDCHUNK>>

# Weighting matrices for controller.
Q = np.diag(xs**-2)
R = np.diag(us**-2)

[K, Pi] = mpc.util.dlqr(A,B,Q,R)

#<<ENDCHUNK>>

# Define disturbance model.
useGoodDisturbanceModel = True

# Bad disturbance model with offset.
if useGoodDisturbanceModel:
    Nid = Ny # Number of integrating disturbances.
else:
    Nid = Nu 

Bd = np.zeros((Nx,Nid))  
Cd = np.zeros((Ny,Nid))

if useGoodDisturbanceModel:
    Cd[0,0] = 1
    Cd[2,1] = 1
    Bd[:,2] = B[:,1] # or Bp[:,0]
else:
    Cd[0,0] = 1
    Cd[2,1] = 1

# Augmented system. Trailing .A casts to array type.    
Aaug = np.bmat([[A,Bd],
    [np.zeros((Nid,Nx)),np.eye(Nid)]]).A
Baug = np.vstack((B,np.zeros((Nid,Nu))))
Caug = np.hstack((C,Cd))

# Check rank condition for augmented system.
# See Lemma 1.8 from Rawlings and Mayne (2009).
svds = linalg.svdvals(np.bmat([[np.eye(Nx) - A,
                                -Bd],[C,Cd]]))
rank = sum(svds > 1e-10)
if rank < Nx + Nid:
    print("*Warning: system not detectable!")

#<<ENDCHUNK>>

# Build augmented penalty matrices for KF.
Qw = eps*np.eye(Nx + Nid)
Qw[-1,-1] = 1
Rv = eps*np.diag(xs**2)

# Get Kalman filter.
[L, P] = mpc.util.dlqe(Aaug, Caug, Qw, Rv)
Lx = L[:Nx,:]
Ld = L[Nx:,:]

#<<ENDCHUNK>>

# Now simulate things.
Nsim = 50
t = np.arange(Nsim+1)*Delta
x = np.zeros((Nsim+1,Nx))
u = np.zeros((Nsim,Nu))
y = np.zeros((Nsim+1,Ny))
err = y.copy()
v = y.copy()
xhat = x.copy() # State estimate after measurement.
xhatm = xhat.copy() # ... before measurement.
dhat = np.zeros((Nsim+1,Nid))
dhatm = dhat.copy()

#<<ENDCHUNK>>

# Pick disturbance and setpoint.
d = np.zeros((Nsim,Nd))
d[:,0] = (t[:-1] >= 10)*.1*F0s
ysp = np.zeros(y.shape)
contVars = [0,2] # Concentration and height.

#<<ENDCHUNK>>

# Steady-state target selector matrices.
H = np.zeros((Nu,Ny))
H[range(len(contVars)),contVars] = 1
Ginv = np.array(np.bmat([
    [np.eye(Nx) - A, -B],
    [H.dot(C), np.zeros((H.shape[0], Nu))]
]).I) # Take inverse.

#<<ENDCHUNK>>

for n in range(Nsim + 1):
    # Take plant measurement.
    y[n,:] = C.dot(x[n,:]) + v[n,:]
    
    # Update state estimate with measurement.
    err[n,:] = (y[n,:] - C.dot(xhatm[n,:])
        - Cd.dot(dhatm[n,:]))
    xhat[n,:] = xhatm[n,:] + Lx.dot(err[n,:])
    dhat[n,:] = dhatm[n,:] + Ld.dot(err[n,:])
    
    # Make sure we aren't at the last timestep.
    if n == Nsim: break

    #<<ENDCHUNK>>

    # Steady-state target.
    rhs = np.concatenate((Bd.dot(dhat[n,:]),
        H.dot(ysp[n,:] - Cd.dot(dhat[n,:]))))
    qsp = Ginv.dot(rhs)
    xsp = qsp[:Nx]
    usp = qsp[Nx:]
    
    #<<ENDCHUNK>>    
    
    # Regulator.
    u[n,:] = K.dot(xhat[n,:] - xsp) + usp
    
    #<<ENDCHUNK>>    
    
    # Simulate with nonlinear model.
    x[n+1,:] = cstr.sim(x[n,:] + xs, u[n,:] + us,
        d[n,:] + ds) - xs
    
    #<<ENDCHUNK>>    
    
    # Advance state estimate.
    xhatm[n+1,:] = (A.dot(xhat[n,:])
        + Bd.dot(dhat[n,:]) + B.dot(u[n,:]))
    dhatm[n+1,:] = dhat[n,:]

#<<ENDCHUNK>>

# Define plotting function.
def cstrplot(x,u,ysp=None,contVars=[],title=None):
    u = np.concatenate((u,u[-1:,:]))
    t = np.arange(0,x.shape[0])*Delta
    ylabelsx = ["$c$ (mol/L)", "$T$ (K)", "$h$ (m)"]
    ylabelsu = ["$T_c$ (K)", "$F$ (kL/min)"]
    
    gs = gridspec.GridSpec(Nx*Nu,2)    
    
    fig = plt.figure(figsize=(10,6))
    for i in range(Nx):
        ax = fig.add_subplot(gs[i*Nu:(i+1)*Nu,0])
        ax.plot(t,x[:,i] + xs[i],'-ok')
        if i in contVars:
            ax.step(t,ysp[:,i] + xs[i],'-r',
                    where="post")
        ax.set_ylabel(ylabelsx[i])
        mpc.plots.zoomaxis(ax,yscale=1.1)
        mpc.plots.prettyaxesbox(ax)
        mpc.plots.prettyaxesbox(ax,
            facecolor="white",front=False)
    ax.set_xlabel("Time (min)")
    for i in range(Nu):
        ax = fig.add_subplot(gs[i*Nx:(i+1)*Nx,1])
        ax.step(t,u[:,i] + us[i],'-k',where="post")
        ax.set_ylabel(ylabelsu[i])
        mpc.plots.zoomaxis(ax,yscale=1.25)
        mpc.plots.prettyaxesbox(ax)
        mpc.plots.prettyaxesbox(ax,
            facecolor="white",front=False)
    ax.set_xlabel("Time (min)")
    fig.tight_layout(pad=.5)
    if title is not None:
        fig.canvas.set_window_title(title)
    return fig
fig = cstrplot(x,u,ysp=None,contVars=[],title=None) 
mpc.plots.showandsave(fig,"cstr_python.pdf",facecolor="none")
