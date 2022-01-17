# This is a CSTR NMHE NMPC simulation based on Example 1.11 from 
# Rawlings and Mayne.

from mpctools import mpcsim as sim
import mpctools as mpc
import numpy as np
from scipy import linalg

useCasadiSX = True

def runsim(k, simcon, opnclsd):

    print("runsim: iteration %d -----------------------------------" % k)

    # unpack stuff from simulation container

    mvlist = simcon.mvlist
    dvlist = simcon.dvlist
    cvlist = simcon.cvlist
    xvlist = simcon.xvlist
    oplist = simcon.oplist
    deltat = simcon.deltat
    #TODO: remove hard-coded numbers.
    vrlist = [mvlist[0], mvlist[1], dvlist[0], xvlist[0], xvlist[1],
              xvlist[2], cvlist[0], cvlist[1], cvlist[2], oplist[0],
              oplist[1]]
    nf = oplist[0]
    dh = oplist[1]
    doolpred = oplist[2]

    # check for changes

    chsum = 0

    for var in vrlist:
        chsum += var.chflag
        var.chflag = 0
    
    # Grab bounds.
    uub = [mvlist[0].maxlim, mvlist[1].maxlim]
    ulb = [mvlist[0].minlim, mvlist[1].minlim]
    yub = [cvlist[0].maxlim, cvlist[1].maxlim, cvlist[2].maxlim]
    ylb = [cvlist[0].minlim, cvlist[1].minlim, cvlist[2].minlim]
    
    # initialize values on first execution or when something changes

    if (k == 0 or chsum > 0):

        print("runsim: initialization")

        # Define problem size parameters

        Nx   = 3            # number of states
        Nu   = 2            # number of inputs
        Nd   = 1            # number of unmeasured disturbances
        Ny   = Nx           # number of outputs
        Nid  = Ny           # number of integrating disturbances

        Nw   = Nx + Nid     # number of augmented states
        Nv   = Ny           # number of output measurements
        Nf   = xvlist[0].Nf # length of NMPC future horizon
        Nmhe = 60           # length of MHE past horizon

        psize = [Nx,Nu,Nd,Ny,Nid,Nw,Nv,Nf,Nmhe]

        # Define sample time in minutes
        
        Delta = deltat


        # Define model parameters
        
        T0 = 350
        c0 = 1
        r = .219
        k0 = 7.2e10
        E = 8750
        U = 54.94
        rho = 1000
        Cp = .239
        dH = dh.value

        # Define the cstr model

        def cstrmodel(c,T,h,Tc,F,F0):
            # ODE for CSTR.
            rate = k0*c*np.exp(-E/T)

            dxdt = np.array([
                F0*(c0 - c)/(np.pi*r**2*h) - rate,
                F0*(T0 - T)/(np.pi*r**2*h)
                    - dH/(rho*Cp)*rate
                    + 2*U/(r*rho*Cp)*(Tc - T),    
                (F0 - F)/(np.pi*r**2)
            ])
            return dxdt

                # Define ode for CSTR simulation

        def ode(x,u,d):
            # Grab the states, controls, and disturbance.    
            [c, T, h] = x[0:Nx]
            [Tc, F] = u[0:Nu]
            [F0] = d[0:Nd]
            
            c = x[0]
            T = x[1]
            h = x[2]
            Tc = u[0]
            F = u[1]
            F0 = d[0]
            return cstrmodel(c,T,h,Tc,F,F0)

        # Create casadi function and simulator.

        ode_casadi = mpc.getCasadiFunc(ode,[Nx,Nu,Nd],["x","u","d"],"ode")
        cstr = mpc.DiscreteSimulator(ode, Delta, [Nx,Nu,Nd], ["x","u","d"])

        # Set the steady-state values.

        cs = .878
        Ts = 324.5
        hs = .659
        Fs = .1
        Tcs = 300
        F0s = .1

        # Calculate steady-state values to high precision
        
        for i in range(10):
            [cs,Ts,hs] = cstr.sim([cs,Ts,hs],[Tcs,Fs],[F0s]).tolist()
        xs = np.array([cs,Ts,hs])
        xaugs = np.concatenate((xs,np.zeros((Nid,))))
        us = np.array([Tcs,Fs])
        ds = np.array([F0s])
        
        # Define augmented model for state estimation.    

        # We need to define two of these because Ipopt isn't smart enough to throw out
        # the 0 = 0 equality constraints. ode_disturbance only gives dx/dt for the
        # actual states, and ode_augmented appends the zeros so that dx/dt is given for
        # all of the states. It's possible that Ipopt may figure this out by itself,
        # but we should just be explicit to avoid any bugs.    
        def ode_disturbance(x,u,d=ds):
            # Grab states, estimated disturbances, controls, and actual disturbance.
            [c, T, h] = x[0:Nx]
            dhat = x[Nx:Nx+Nid] # Actually, this guy does work.
            [Tc, F] = u[0:Nu]
            [F0] = d[0:Nd]
            
            dxdt = cstrmodel(c,T,h,Tc,F+dhat[2],F0)
            return dxdt

        def ode_augmented(x,u,d=ds):
            # Need to add extra zeros for derivative of disturbance states.
            dxdt = np.concatenate((ode_disturbance(x,u,d),np.zeros((Nid,))))
            return dxdt
 
        cstraug = mpc.DiscreteSimulator(ode_augmented, Delta,
                                        [Nx+Nid,Nu,Nd], ["xaug","u","d"])

        def measurement(x,d=ds):
            [c, T, h] = x[0:Nx]
            dhat = x[Nx:Nx+Nid]
            return np.array([c + dhat[0], T + dhat[1], h])
        ys = measurement(xaugs)

        # Turn into casadi functions.
        ode_disturbance_casadi = mpc.getCasadiFunc(ode_disturbance,
                                   [Nx+Nid,Nu,Nd],["xaug","u","d"],"ode_disturbance")
        ode_augmented_casadi = mpc.getCasadiFunc(ode_augmented,
                               [Nx+Nid,Nu,Nd],["xaug","u","d"],"ode_augmented")
        ode_augmented_rk4_casadi = mpc.getCasadiFunc(ode_augmented,
                                   [Nx+Nid,Nu,Nd],["xaug","u","d"],"ode_augmented_rk4",
                                     rk4=True,Delta=Delta,M=2)

        def ode_estimator_rk4(x,u,w=np.zeros((Nx+Nid,)),d=ds):
            return ode_augmented_rk4_casadi(x, u, d) + w

        def ode_estimator(x,u,w=np.zeros((Nx+Nid,)),d=ds):
            return ode_augmented_casadi(x, u, d) + w

        ode_estimator_rk4_casadi = mpc.getCasadiFunc(ode_estimator_rk4,
                                   [Nx+Nid,Nu,Nw,Nd], ["xaug","u","w","d"],
                                   "ode_estimator_rk4")

        measurement_casadi = mpc.getCasadiFunc(measurement,
                             [Nx+Nid,Nd], ["xaug","d"], "measurement")

        # Weighting matrices for controller.

        Q = np.diag([cvlist[0].qvalue, 0.0, cvlist[2].qvalue])
        R = np.diag([mvlist[0].rvalue, mvlist[1].rvalue])
        S = np.diag([mvlist[0].svalue, mvlist[1].svalue])

        # Now get a linearization at this steady state and calculate Riccati cost-to-go.

        ss = mpc.util.getLinearizedModel(ode_casadi, [xs,us,ds], ["A","B","Bp"], Delta)
        A = ss["A"]
        B = ss["B"]
        [K, Pi] = mpc.util.dlqr(A,B,Q,R)

        # Define control stage cost

        def stagecost(x,u,xsp,usp,Deltau):
            dx = x[:Nx] - xsp[:Nx]
            du = u - usp
            return (mpc.mtimes(dx.T,Q,dx) + .1*mpc.mtimes(du.T,R,du)
                + mpc.mtimes(Deltau.T,S,Deltau))

        largs = ["x","u","x_sp","u_sp","Du"]
        l = mpc.getCasadiFunc(stagecost,
            [Nx+Nid,Nu,Nx+Nid,Nu,Nu],largs,funcname="l")

        # Define cost to go.

        def costtogo(x,xsp):
            dx = x[:Nx] - xsp[:Nx]
            return mpc.mtimes(dx.T,Pi,dx)
        Pf = mpc.getCasadiFunc(costtogo,[Nx+Nid,Nx+Nid],["x","s_xp"],
                               funcname="Pf")

        # Build augmented estimator matrices.

        Qw = np.diag([xvlist[0].mnoise, xvlist[1].mnoise, xvlist[2].mnoise,
                    1, 1, 1])
        Rv = np.diag([cvlist[0].mnoise, cvlist[1].mnoise, cvlist[2].mnoise])
        Qwinv = linalg.inv(Qw)
        Rvinv = linalg.inv(Rv)

        # Define stage costs for estimator.

        def lest(w,v):
            return mpc.mtimes(w.T,Qwinv,w) + mpc.mtimes(v.T,Rvinv,v)
                      
        lest = mpc.getCasadiFunc(lest,[Nw,Nv],["w","v"],"l")

        # Don't use a prior.
        lxest = None
        x0bar = None

        # Check if the augmented system is detectable. (Rawlings and Mayne, Lemma 1.8)
        Aaug = mpc.util.getLinearizedModel(ode_augmented_casadi,[xaugs, us, ds],
                                       ["A","B","Bp"], Delta)["A"]
        Caug = mpc.util.getLinearizedModel(measurement_casadi,[xaugs, ds],
                                       ["C","Cp"])["C"]
        Oaug = np.vstack((np.eye(Nx,Nx+Nid) - Aaug[:Nx,:], Caug))
        svds = linalg.svdvals(Oaug)
        rank = sum(svds > 1e-8)
        if rank < Nx + Nid:
            print("***Warning: augmented system is not detectable!")

        # Make NMHE solver.

        uguess = np.tile(us,(Nmhe,1))
        xguess = np.tile(xaugs,(Nmhe+1,1))
        yguess = np.tile(ys,(Nmhe+1,1))
        nmheargs = {
            "f" : ode_estimator_rk4_casadi,
            "h" : measurement_casadi,
            "u" : uguess,
            "y" : yguess,
            "l" : lest,
            "N" : {"x":Nx + Nid, "u":Nu, "y":Ny, "p":Nd, "t":Nmhe},
            "lx" : lxest,
            "x0bar" : x0bar,
            "p" : np.tile(ds,(Nmhe+1,1)),
            "verbosity" : 0,
            "guess" : {"x":xguess, "y":yguess, "u":uguess},
            "timelimit" : 60,
            "casaditype" : "SX" if useCasadiSX else "MX",                       
        }
        estimator = mpc.nmhe(**nmheargs)

        # Declare ydata and udata. Note that it would make the most sense to declare
        # these using collection.deques since we're always popping the left element or
        # appending a new element, but for these sizes, we can just use a list without
        # any noticable slowdown.

        if (k == 0):
            ydata = [ys]*Nmhe
            udata = [us]*(Nmhe-1)
        else:
            ydata = simcon.ydata
            udata = simcon.udata

        # Make steady-state target selector.

        contVars = [0,2]
        Rss = np.zeros((Nu,Nu))
        Qss = np.zeros((Ny,Ny))
        Qss[contVars,contVars] = 1 # Only care about controlled variables.

        def sstargobj(y,y_sp,u,u_sp,Q,R):
            dy = y - y_sp
            du = u - u_sp
            return mpc.mtimes(dy.T,Q,dy) + mpc.mtimes(du.T,R,du)

        phiargs = ["y","y_sp","u","u_sp","Q","R"]
        phi = mpc.getCasadiFunc(sstargobj, [Ny,Ny,Nu,Nu,(Ny,Ny),(Nu,Nu)],
                                phiargs)

        sstargargs = {
            "f" : ode_disturbance_casadi,
            "h" : measurement_casadi,
            "lb" : {"u" : np.tile(ulb, (1,1)), "y" : np.tile(ylb, (1,1))},
            "ub" : {"u" : np.tile(uub, (1,1)), "y" : np.tile(yub, (1,1))},
            "guess" : {
                "u" : np.tile(us, (1,1)),
                "x" : np.tile(np.concatenate((xs,np.zeros((Nid,)))), (1,1)),
                "y" : np.tile(xs, (1,1)),
            },
            "p" : np.tile(ds, (1,1)), # Parameters for system.
            "N" : {"x" : Nx + Nid, "u" : Nu, "y" : Ny, "p" : Nd, "f" : Nx},
            "phi" : phi,
            "funcargs" : dict(phi=phiargs),
            "extrapar" : {"R" : Rss, "Q" : Qss, "y_sp" : ys, "u_sp" : us},
            "verbosity" : 0,
            "discretef" : False,
            "casaditype" : "SX" if useCasadiSX else "MX",
        }
        targetfinder = mpc.sstarg(**sstargargs)

        # Make NMPC solver.

        duub = [mvlist[0].roclim, mvlist[1].roclim]
        dulb = [-mvlist[0].roclim, -mvlist[1].roclim]
        lb = {"u" : np.tile(ulb, (Nf,1)), "Du" : np.tile(dulb, (Nf,1)),
              "y" : np.tile(ylb, (Nf,1))}
        ub = {"u" : np.tile(uub, (Nf,1)), "Du" : np.tile(duub, (Nf,1)),
              "y" : np.tile(yub, (Nf,1))}
        N = {"x":Nx+Nid, "u":Nu, "p":Nd, "t":Nf}
        p = np.tile(ds, (Nf,1)) # Parameters for system.
        sp = {"x" : np.tile(xaugs, (Nf+1,1)), "u" : np.tile(us, (Nf,1))}
        guess = sp.copy()
        xaug0 = xaugs
        nmpcargs = {
            "f" : ode_augmented_rk4_casadi,
            "l" : l,
            "funcargs" : dict(l=largs),
            "N" : N,
            "x0" : xaug0,
            "uprev" : us,
            "lb" : lb,
            "ub" : ub,
            "guess" : guess,
            "Pf" : Pf,
            "sp" : sp,
            "p" : p,
            "verbosity" : 0,
            "timelimit" : 60,
            "casaditype" : "SX" if useCasadiSX else "MX",
        }
        controller = mpc.nmpc(**nmpcargs)

        if k == 0:
            # Initialize variables
            x_k      = np.zeros(Nx)
            xhat_k   = np.zeros(Nx)
            dhat_k   = np.zeros(Nid)

            # Store initial values for variables
            xvlist.vecassign(xs)
            xvlist.vecassign(xs, "est")
            mvlist.vecassign(us)
            dvlist.vecassign(dhat_k[0], "est")

        # Store values in simulation container
        simcon.proc = [cstr]
        simcon.mod = []
        simcon.mod.append(us)
        simcon.mod.append(xs)
        simcon.mod.append(ys)
        simcon.mod.append(ds)
        simcon.mod.append(estimator)
        simcon.mod.append(targetfinder)
        simcon.mod.append(controller)
        simcon.mod.append(cstraug)
        simcon.mod.append(measurement)
        simcon.mod.append(psize)
        simcon.ydata = ydata
        simcon.udata = udata

    # Get stored values
    #TODO: these should be dictionaries.
    cstr          = simcon.proc[0]
    us            = simcon.mod[0]
    xs            = simcon.mod[1]
    ys            = simcon.mod[2]
    ds            = simcon.mod[3]
    estimator     = simcon.mod[4]
    targetfinder  = simcon.mod[5]
    controller    = simcon.mod[6]
    cstraug       = simcon.mod[7]
    measurement   = simcon.mod[8]
    psize         = simcon.mod[9]
    Nx            = psize[0]
    Nu            = psize[1]
    Nd            = psize[2]
    Ny            = psize[3]
    Nid           = psize[4]
    Nw            = psize[5]
    Nv            = psize[6]
    Nf            = psize[7]
    Nmhe          = psize[8]
    ydata         = simcon.ydata
    udata         = simcon.udata

    # Get variable values
    x_km1 = xvlist.asvec()
    u_km1 = mvlist.asvec()
    d_km1 = dvlist.asvec()    

    # Advance the process

    x_k = cstr.sim(x_km1, u_km1, d_km1)

    # Take plant measurement

    y_k = measurement(np.concatenate((x_k,np.zeros((Nid,)))))

    if (nf.value > 0.0):

        for i in range(0, Ny):
            y_k[i] += nf.value*np.random.normal(0.0, cvlist[i].noise)
    
    # Do Nonlinear MHE.

    ydata.append(y_k)
    udata.append(u_km1) 
    estimator.par["y"] = ydata
    estimator.par["u"] = udata
    estimator.solve()
    estsol = mpc.util.casadiStruct2numpyDict(estimator.var)        

    print("runsim: estimator status - %s" % (estimator.stats["status"]))
    xaughat_k = estsol["x"][-1,:]
    xhat_k = xaughat_k[:Nx]
    dhat_k = xaughat_k[Nx:]

    yhat_k = measurement(np.concatenate((xhat_k, dhat_k)))
    ydata.pop(0)
    udata.pop(0)    
    estimator.saveguess()        

    # Initialize the input
    u_k = u_km1

    # Update open and closed-loop predictions
    for field in ["olpred", "clpred"]:
        mvlist.vecassign(u_k, field, index=0)
        dvlist.vecassign(d_km1, field, index=0)
        xvlist.vecassign(xhat_k, field, index=0)
        cvlist.vecassign(yhat_k, field, index=0)
    
    xof_km1 = np.concatenate((xhat_k,dhat_k))

    # Need to be careful about this forecasting. Temporarily aggressive control
    # could cause the system to go unstable if continued indefinitely, and so
    # this simulation might fail. If the integrator fails at any step, then we
    # just return NaNs for future predictions. Also, if the user doesn't want
    # predictions, then we just always skip them.
    predictionOkay = bool(doolpred.value)
    for i in range(0,(Nf - 1)):
        if predictionOkay:
            try:
                xof_k = cstraug.sim(xof_km1, u_km1, ds)
            except RuntimeError: # Integrator failed.
                predictionOkay = False
        if predictionOkay:
            # Take measurement.
            yof_k = measurement(xof_k)
            
            # Stop forecasting if bounds are exceeded.
            if np.any(yof_k > yub) or np.any(yof_k < ylb):
                predictionOkay = False
        else:
            xof_k = np.NaN*np.ones((Nx+Nid,))
            yof_k = np.NaN*np.ones((Ny,))

        for field in ["olpred", "clpred"]:
            mvlist.vecassign(u_k, field, index=(i + 1))
            dvlist.vecassign(d_km1, field, index=(i + 1))
            xvlist.vecassign(xof_k[:Nx], field, index=(i + 1)) # Note [:Nx].
            cvlist.vecassign(yof_k[:Ny], field, index=(i + 1))
    
        xof_km1 = xof_k
    
    # calculate mpc input adjustment in control is on

    if (opnclsd.status.get() == 1):

        # Use nonlinear steady-state target selector

        ysp_k = [cvlist[0].setpoint, cvlist[1].setpoint, cvlist[2].setpoint]
        usp_k = [mvlist[0].target, mvlist[1].target]
        xtarget = np.concatenate((ysp_k,dhat_k))

        # Previously had targetfinder.par["p",0] = d_km1, but this shouldn't
        # be because the target finder should be using the same model as the
        # controller and doesn't get to know the real disturbance.
        targetfinder.guess["x",0] = xtarget
        targetfinder.fixvar("x", 0, dhat_k, np.arange(Nx, Nx+Nid))
        targetfinder.par["y_sp",0] = ysp_k
        targetfinder.par["u_sp",0] = usp_k
        targetfinder.guess["u",0] = u_km1
        targetfinder.solve()

        xaugss = np.squeeze(targetfinder.var["x",0,:])
        uss = np.squeeze(targetfinder.var["u",0,:])

        print("runsim: target status - %s (Obj: %.5g)" % (targetfinder.stats["status"],targetfinder.obj)) 
        
        # Now use nonlinear MPC controller.

        controller.par["x_sp"] = [xaugss]*(Nf + 1)
        controller.par["u_sp"] = [uss]*Nf
        controller.par["u_prev"] = [u_km1]
        controller.fixvar("x",0,xaughat_k)            
        controller.solve()
        print("runsim: controller status - %s (Obj: %.5g)" % (controller.stats["status"],controller.obj)) 

        controller.saveguess()
        u_k = np.squeeze(controller.var["u",0])

        # Update closed-loop predictions

        sol = mpc.util.casadiStruct2numpyDict(controller.var)

        mvlist.vecassign(u_k, "clpred", index=0)
        xvlist.vecassign(xhat_k, "clpred", index=0)
        cvlist.vecassign(yhat_k, "clpred", index=0)

        for i in range(Nf - 1):
            mvlist.vecassign(sol["u"][i+1,:], "clpred", index=(i + 1))
            xvlist.vecassign(sol["x"][i+1,:Nx], "clpred", index=(i + 1))
            xcl_k = sol["x"][i+1,:]
            ycl_k = measurement(xcl_k)
            cvlist.vecassign(ycl_k[:Ny], "clpred", index=(i + 1))

    else:
        # Track the cv setpoints if the control is not on. Can't use vecassign
        # here because we skip cvlist[1].
        cvlist[0].setpoint = y_k[0]
        cvlist[2].setpoint = y_k[2]

    # Store variable values
    mvlist.vecassign(u_k)
    xvlist.vecassign(x_k)
    xvlist.vecassign(xhat_k, "est")
    dvlist.vecassign(dhat_k[0], "est") # Only want the first component.
    cvlist.vecassign(y_k)
    cvlist.vecassign(yhat_k, "est")
    simcon.ydata    = ydata
    simcon.udata    = udata

# set up cstr mpc example

simname = 'CSTR NMPC Example'

# define variables

MVmenu=["value","rvalue","svalue","maxlim","minlim","roclim","pltmax","pltmin"]
DVmenu=["value","pltmax","pltmin"]
XVmenu=["mnoise","noise","pltmax","pltmin"]
CVmenu=["setpoint","qvalue","maxlim","minlim","mnoise","noise","pltmax","pltmin"]
FVmenu=["mnoise","noise","pltmax","pltmin"]

MV1 = sim.MVobj(name='Tc', desc='mv - coolant temp.', units='(K)', 
               pltmin=250.0, pltmax=350.0, minlim=255.0, maxlim=345.0,
               value=300.0, target=300.0, Nf=60, menu=MVmenu)

MV2 = sim.MVobj(name='F', desc='mv - outlet flow', units='(kL/min)', 
               pltmin=0.0, pltmax=0.3, minlim=0.01, maxlim=0.29,
               value=0.1, target=0.1, Nf=60, menu=MVmenu)

DV1 = sim.MVobj(name='F0', desc='dv - inlet flow', units='(kL/min)', 
               pltmin=0.0, pltmax=0.3,
               value=0.1, Nf=60, menu=DVmenu)

XV1 = sim.XVobj(name='c', desc='xv - concentration A', units='(mol/L)', 
               pltmin=0.0, pltmax=1.4, 
               value=0.877825, Nf=60, menu=XVmenu)

XV2 = sim.XVobj(name='T', desc='xv - temperature', units='(K)', 
               pltmin=250.0, pltmax=500.0, 
               value=324.496, Nf=60, menu=XVmenu)

XV3 = sim.XVobj(name='h', desc='xv - level', units='(m)', 
               pltmin=0.0, pltmax=1.0, 
               value=0.659, Nf=60, menu=XVmenu)

CV1 = sim.XVobj(name='c', desc='cv - concentration A', units='(mol/L)', 
               pltmin=0.0, pltmax=1.4, minlim=0.05, maxlim=1.35,
               noise =.0001,
               value=0.877825, setpoint=0.877825, Nf=60, menu=CVmenu)

CV2 = sim.XVobj(name='T', desc='fv - temperature', units='(K)', 
               pltmin=250.0, pltmax=500.0, noise=.1,
               value=324.496, setpoint=100.0, Nf=60, menu=FVmenu)

CV3 = sim.XVobj(name='h', desc='cv - level', units='(m)', 
               pltmin=0.0, pltmax=1.0, minlim=0.05, maxlim=0.95,
               noise=.01,
               value=0.659, setpoint=0.659, Nf=60, menu=CVmenu)

# define options

NF = sim.Option(name='NF', desc='Noise Factor', value=0.0)
DH = sim.Option(name='DH', desc='Heat Of Reaction', value=-5e4)
OL = sim.Option(name="OL Pred.", desc="Open-Loop Predictions", value=1)

# load up variable lists

MVlist = [MV1, MV2]
DVlist = [DV1]
XVlist = [XV1, XV2, XV3]
CVlist = [CV1, CV2, CV3]
OPlist = [NF, DH, OL]
DeltaT = 0.5
N      = 120
refint = 100


def dosimulation():
    """Create the GUI and run the simulation."""
    simcon = sim.SimCon(simname=simname,
                        mvlist=MVlist, dvlist=DVlist, cvlist=CVlist, xvlist=XVlist,
                        oplist=OPlist, N=N, refint=refint, runsim=runsim, deltat=DeltaT)
    
    # build the GUI and start it up
    
    sim.makegui(simcon)

if __name__ == "__main__":
    dosimulation()
