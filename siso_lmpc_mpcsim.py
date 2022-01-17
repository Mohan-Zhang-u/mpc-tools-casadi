# This is a siso linear mpc example.
#
# Note that we have to be careful about what is a 1 by 1 matrix vs. a 1 vector
# vs. a scalar. In Matlab, they are all the same, but in NumPy, they are not.

from mpctools import mpcsim as sim
import numpy as np
import mpctools as mpc
import random as rn

def runsim(k, simcon, opnclsd):

    print("runsim: iteration %d -----------------------------------" % k)

    # unpack stuff from simulation container

    mvlist = simcon.mvlist
    cvlist = simcon.cvlist
    xvlist = simcon.xvlist
    oplist = simcon.oplist
    deltat = simcon.deltat

    # get mv, cv, xv and options

    mv = mvlist[0]
    cv = cvlist[0]
    xv = xvlist[0]
    nf = oplist[0]
    av = oplist[1]
    vrlist = [mv,cv,xv,nf,av]

    # check for changes
    chsum = 0
    for var in vrlist:
        chsum += var.chflag
        var.chflag = 0

    # initialize values on first execution or when something changes

    if (k == 0 or chsum > 0):

        print("runsim: initialization")

        # Define continuous time model.
        Acont = np.array([[av.value]]) # Models are matrices.
        Bcont = np.array([[10.0]])
        Nx = Acont.shape[0] # Number of states.
        Nu = Bcont.shape[1] # Number of control elements

        # Discretize.

        (Adisc,Bdisc) = mpc.util.c2d(Acont,Bcont,deltat)

        # Define Q and R matrices and q penalty for periodic solution.

        Q = np.diag([cv.qvalue])
        R = np.diag([mv.rvalue])

        # Create discrete process
        def Fproc(x, u):
            return mpc.mtimes(Adisc,x) + mpc.mtimes(Bdisc,u)
        simcon.proc = Fproc

        # Create discrete controller (possibly with model mismatch)

        Amod = Adisc
        Bmod = Bdisc
        simcon.gain = Bmod[0,0]/(1-Amod[0,0]) # Scalar.
        def Fmod(x, u):
            """Model F function (possibly with mismatch)."""
            return mpc.mtimes(Amod, x) + mpc.mtimes(Bmod, u)
        simcon.mod = Fmod
        simcon.F = mpc.getCasadiFunc(Fmod,[Nx,Nu],["x","u"],"F")
        def l(x, u):
            """Quadratic stage cost."""
            return mpc.mtimes(x.T, Q, x) + mpc.mtimes(u.T, R, u)
        simcon.l = mpc.getCasadiFunc(l,[Nx,Nu],["x","u"],"l")
        def Pf(x):
            """Quadratic terminal cost."""
            return mpc.mtimes(x.T, Q, x)
        simcon.Pf = mpc.getCasadiFunc(Pf,[Nx],["x"],"Pf")

        # initialize the state
        simcon.xmk = np.array([cv.value]) # State is a 1 vector.

    # Get reference points and state disturbance as 1 vectors.
    xref = np.array([xv.ref])
    uref = np.array([mv.ref])
    w = np.array([cv.dist])
    
    # increment the process.
    xpkm1 = np.array([xv.value])
    ukm1 = np.array([mv.value + mv.dist])
    dxpkm1 = xpkm1 - xref
    dukm1 = ukm1 - uref
    dxpk = simcon.proc(dxpkm1, dukm1) + w
    xpk  = dxpk + xref
    ypk = xpk + w

    # zero out the disturbances
    mv.dist = 0.0
    xv.dist = 0.0

    # add noise if desired

    if nf.value > 0.0:
        xpk += np.array([nf.value*rn.uniform(-xv.noise, xv.noise)])
        ypk += np.array([nf.value*rn.uniform(-cv.noise, cv.noise)]) 
    
    # store values
    xv.value = xpk[0] # These values should be scalars.
    cv.value = ypk[0]

    # increment the model
    uk = ukm1    
    xmkm1 = simcon.xmk
    dxmkm1 = xmkm1 - xref
    dukm1 = ukm1 - uref
    dxmk = simcon.mod(dxmkm1, dukm1)
    xmk = dxmk + xref
    ymk = xmk
    simcon.xmk = xmk

    # simple bias feedback
    cv.bias = (ypk - ymk)[0] # Needs to be a scalar.

    # update future predictions
    mv.olpred[0] = uk[0]
    xv.olpred[0] = xmk[0] + cv.bias
    cv.olpred[0] = ymk[0] + cv.bias
    xv.est = xmk[0]
    cv.est = ymk[0] + cv.bias
    mv.clpred[0] = uk[0]
    xv.clpred[0] = xmk[0] + cv.bias
    cv.clpred[0] = ymk[0] + cv.bias
    duk = uk - uref # This guy is a vector.

    for i in range(xv.Nf -1):
       dxmkp1 = simcon.mod(dxmk, duk)
       mv.olpred[i+1] = uk
       xv.olpred[i+1] = dxmkp1[0] + xv.ref + cv.bias
       cv.olpred[i+1] = dxmkp1[0] + cv.ref + cv.bias
       mv.clpred[i+1] = uk[0]
       xv.clpred[i+1] = dxmkp1[0] + xv.ref + cv.bias
       cv.clpred[i+1] = dxmkp1[0] + cv.ref + cv.bias
       dxmk = dxmkp1

    # set xv target, limits same as cv limits
    xv.maxlim = cv.maxlim
    xv.minlim = cv.minlim

    # calculate mpc input adjustment in control is on
    if opnclsd.status.get() == 1:
        # calculate steady state
        cv.sstarg = cv.setpoint - cv.bias
        xv.sstarg = cv.sstarg
        mv.sstarg = (xv.sstarg - xv.ref)/simcon.gain + mv.ref
        
        # set mv, xv bounds
        ulb = np.array([mv.minlim - mv.sstarg])
        uub = np.array([mv.maxlim - mv.sstarg])
          
        xlb = np.array([xv.minlim - xv.sstarg - cv.bias])
        xub = np.array([xv.maxlim - xv.sstarg - cv.bias])
        
        # Loop to verify sizes. Before, we were a bit sloppy with scalars vs.
        # vectors, so we add this loop just to check. - MJR (2/18/2016)
        for thing in [ulb, uub, xlb, xub]:
            assert thing.shape == (1,)
        
        N = {"t" : xv.Nf, "x" : 1, "u" : 1}
        lb = {"x" : np.tile(xlb, (N["t"] + 1, 1)),
              "u" : np.tile(ulb, (N["t"], 1))}
        ub = {"x" : np.tile(xub, (N["t"] + 1, 1)),
              "u" : np.tile(uub, (N["t"],1))}

        # solve for new input and state
        dxss = xmk - xv.sstarg
        alg = mpc.nmpc(simcon.F, simcon.l, N, dxss, Pf=simcon.Pf, lb=lb, ub=ub,
                       verbosity=0)
        alg.solve()
        sol = mpc.util.casadiStruct2numpyDict(alg.var)
        sol["status"] = alg.stats["status"]
        duk = sol["u"][0,:] 
        uk = duk + np.array([mv.sstarg])
        
        # update future predictions
        mv.clpred = sol["u"][:,0] + mv.sstarg
        xv.clpred = sol["x"][:-1,0] + xv.sstarg + cv.bias
        cv.clpred = sol["x"][:-1,0] + cv.sstarg + cv.bias

        print("runsim: control status - %s" % sol["status"])

    # load current input
    mv.value = uk[0] # Needs to be a scalar.

# set up siso mpc example

simname = 'SISO LMPC Example'

# define variables

MVmenu = ["value", "rvalue", "maxlim", "minlim", "pltmax", "pltmin"]
XVmenu = ["noise", "pltmax", "pltmin"]
CVmenu = ["setpoint", "qvalue", "maxlim", "minlim", "noise", "pltmax",
          "pltmin"]

MV = sim.MVobj(name='MV', desc='manipulated variable', units='m3/h  ', 
               pltmin=0.0, pltmax=4.0, minlim=0.5, maxlim=3.5,
               value=2.0, Nf=60, menu=MVmenu)

CV = sim.CVobj(name='CV', desc='controlled variable ', units='degC  ', 
               pltmin=0.0, pltmax=50.0, minlim=5.0, maxlim=45.0,
               value=25.0, setpoint=25.0, noise=0.1, Nf=60, menu=CVmenu)

XV = sim.XVobj(name='XV', desc='state variable ', units='degC  ', 
               pltmin=0.0, pltmax=50.0, 
               value=25.0, setpoint=25.0, noise=0.1, Nf=60, menu=XVmenu)

# define options

NF = sim.Option(name='NF', desc='Noise Factor', value=0.0)

AV = sim.Option(name='AV', desc='A Value', value=-0.9)

# load up variable lists

MVlist = [MV]
DVlist = []
CVlist = [CV]
XVlist = [XV]
OPlist = [NF,AV]
DeltaT = .10
N = 120
refint = 10.0
simcon = sim.SimCon(simname=simname, mvlist=MVlist, dvlist=DVlist,
                    cvlist=CVlist, xvlist=XVlist, oplist=OPlist, N=N,
                    refint=refint, runsim=runsim, deltat=DeltaT)

# build the GUI and start it up

sim.makegui(simcon)
