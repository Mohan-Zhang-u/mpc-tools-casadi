# Example 2.8 from Rawlings and Mayne (2009)

# Pick whether to enforce terminal x as constraint or with a penalty. Setting
# this to False will make the problem harder and less likely to solve, but it
# also reduces "cheating".
usePf = True

# Imports.
import numpy as np
import mpctools as mpc
import mpctools.plots as mpcplots
import matplotlib.pyplot as plt

Nt = 3
Nx = 2
Nu = 1
N = dict(t=Nt, x=Nx, u=Nu)

# Model and stage cost.
def F(x,u):
    return [x[0] + u[0], x[1] + u[0]**3]
def l(x,u):
    return mpc.util.mtimes(x.T,x) + mpc.util.mtimes(u.T,u)
def Pf(x):
    return 1000*mpc.mtimes(x.T,x) # Huge terminal penalty.

def nlsim(x0, u, N):
    """
    Simulates N periods of control with controls in u.
    """
    x = np.zeros((N+1, len(x0)))
    x[0,:] = x0
    for i in range(1,N+1):
        x[i,:] = F(x[i-1,:],u[i-1,:])
    return x

Fcasadi = mpc.getCasadiFunc(F, [Nx,Nu], ["x","u"], "F")
lcasadi = mpc.getCasadiFunc(l, [Nx,Nu], ["x","u"], "l")
if usePf:
    Pfcasadi = mpc.getCasadiFunc(Pf, [Nx], ["x"], "Pf")
else:
    Pfcasadi = None # We'll use a terminal constraint, not a penalty.

# Bounds for x.
xub = np.inf*np.ones((Nt+1, Nx))
if not usePf:
    xub[Nt,:] = np.zeros((Nx,)) # Must terminate at origin.
xlb = -xub

lb = dict(x=xlb)
ub = dict(x=xub)
Npts = 101
theta = np.linspace(-1*np.pi, np.pi, Npts)
uopt = np.zeros(theta.shape)

# Define a bunch of functions to build a guess.
def uselastguess(var, prevguess, x0):
    if "status" not in var or var["status"] != "Solve_Succeeded":
        return prevguess
    else:
        return {"x" : var["x"], "u" : var["u"]}
def alternateuguess(var, prevguess, x0, u0):
    u = np.zeros((Nt,Nu))
    u[:,0] = u0
    for k in range(1,Nt):
        u[k,:] = -1*u[k-1,:]
    x = nlsim(x0, u, Nt)
    return {"x" : x, "u" : u}

guessfuncs = [
    ("None", lambda var, prevguess,x0 : {}, "k"),
    ("$1$", lambda var, prevguess,x0 : alternateuguess(var,prevguess,x0,[1]), "r"),
    ("$-1$", lambda var, prevguess,x0 : alternateuguess(var,prevguess,x0,[-1]), "g"),
    ("Previous", uselastguess, "b"),
]
Nguessmethods = len(guessfuncs)

verb = 0
uopt = {}
phiopt = {}
figure = plt.figure(figsize=(6,8))
guessmethod = 0
print("\n-----------------\nGuess Method (run)\n-----------------")
for guessmethod in range(Nguessmethods):
    guess = {}
    opt = {}
    f = guessfuncs[guessmethod][0]
    getguess = guessfuncs[guessmethod][1]
    uopt[f] = np.zeros(theta.shape)
    phiopt[f] = np.zeros(theta.shape)
    for i in range(Npts):
        if i % 25 == 0:        
            print("%s (%3d of %d)" % (f,i+1,Npts))
        x0 = [np.cos(theta[i]), np.sin(theta[i])]
        guess = getguess(opt,guess,x0)
        opt = mpc.callSolver(mpc.nmpc(Fcasadi, lcasadi, N, x0, lb, ub, guess,
                                      Pf=Pfcasadi, verbosity=verb))
        if opt["status"] == "Solve_Succeeded":
            uopt[f][i] = opt["u"][0,0]
            phiopt[f][i] = opt["obj"]
        else:
            uopt[f][i] = np.NaN
            phiopt[f][i] = np.NaN
    
    # Plot optimal u[0].
    ax = figure.add_subplot(Nguessmethods, 2, 2*guessmethod + 1)
    c = guessfuncs[guessmethod][2]
    ax.plot(theta/np.pi,uopt[f],"-o",color=c,label=f,markerfacecolor=c,
            markeredgecolor="none",markersize=3)
    ax.set_ylabel(r"Guess: %s" % f)
    if guessmethod == 0:
        plt.title("Control Input ($u_0$)")
    elif guessmethod == Nguessmethods - 1:
        ax.set_xlabel(r"$\theta/\pi$")
    
    # Plot optimal cost.    
    ax = figure.add_subplot(Nguessmethods, 2, 2*guessmethod + 2)
    c = guessfuncs[guessmethod][2]
    ax.plot(theta/np.pi,phiopt[f],"-o",color=c,label=f,markerfacecolor=c,
            markeredgecolor="none",markersize=3)
    ax.set_ylabel(r"Guess: %s" % f)
    if guessmethod == 0:
        plt.title("Objective Function ($V$)")
    elif guessmethod == Nguessmethods - 1:
        ax.set_xlabel(r"$\theta/\pi$")
plt.tight_layout(pad=.5)
mpcplots.showandsave(plt.gcf(),"example2-8%s.pdf" % ("Pf" if usePf else "",))
