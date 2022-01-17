# This is a fired heater example.
#
# ToDO:
# 1. Add soft constraint weights to CV menu.
# 2. Write up a script 

from mpctools import mpcsim as sim
import mpctools as mpc
import numpy as np
from scipy import linalg
import casadi

useCasadiSX = True

def runsim(k, simcon, opnclsd):

    print("runsim: iteration %d -----------------------------------" % k)

    # Unpack stuff from simulation container.

    mvlist = simcon.mvlist
    dvlist = simcon.dvlist
    cvlist = simcon.cvlist
    xvlist = simcon.xvlist
    xilist = simcon.xilist
    oplist = simcon.oplist
    deltat = simcon.deltat
    nf = oplist[0]
    doolpred = oplist[1]

    #TODO:  change this to not reinitialize - just change stuff

    # Check for changes.

    chsum = 0

    for var in mvlist:
        chsum += var.chflag
        var.chflag = 0
    
    for var in dvlist:
        chsum += var.chflag
        var.chflag = 0
    
    for var in cvlist:
        chsum += var.chflag
        var.chflag = 0
    
    for var in oplist:
        chsum += var.chflag
        var.chflag = 0
    
    # Grab bounds.

    uub = [mvlist[0].maxlim, mvlist[1].maxlim, mvlist[2].maxlim]
    ulb = [mvlist[0].minlim, mvlist[1].minlim, mvlist[2].minlim]
    yub = [cvlist[0].maxlim, cvlist[1].maxlim, cvlist[2].maxlim, cvlist[3].maxlim, cvlist[4].maxlim]
    ylb = [cvlist[0].minlim, cvlist[1].minlim, cvlist[2].minlim, cvlist[3].minlim, cvlist[4].minlim]
    
    # Initialize values on first execution or when something changes.

    if (k == 0 or chsum > 0):

        print("runsim: initialization")

        # Define problem size parameters.

        Nx   = 10           # number of states
        Nu   = 3            # number of inputs
        Nd   = 2            # number of measured disturbances
        Ny   = 5            # number of outputs
        Nid  = Ny           # number of integrating disturbances

        Nw   = Nx + Nid     # number of augmented states
        Nv   = Ny           # number of output measurements
        Nf   = cvlist[0].Nf # length of NMPC future horizon
        Nmhe = 60           # length of MHE past horizon

        psize = [Nx,Nu,Nd,Ny,Nid,Nw,Nv,Nf,Nmhe]

        # Define sample time in minutes.
        
        Delta = deltat

        # Read in the fired heater model continuous state-space matrices.
        # The matrices were generated in Matlab.
        
        sys = mpc.util.c2d(A=ssfhtr_Ax, B=ssfhtr_Bu, Bp=ssfhtr_Bd, Delta=Delta,
                           asdict=True)
        Ax = sys["A"]
        Bu = sys["B"]
        Bd = sys["Bp"]
        Cx = ssfhtr_Cx

        # Set the initial values.

        y0 = np.zeros((Ny,))
        x0 = np.zeros((Nx,))
        u0 = np.zeros((Nu,))
        d0 = np.zeros((Nd,))
        for i in range(Ny):
#            print "cvlist[i] = ", cvlist[i].value
            y0[i] = cvlist[i].value
        for i in range(Nu):
#            print "mvlist[i] = ", mvlist[i].value
            u0[i] = mvlist[i].value
        for i in range(Nd):
#            print "dvlist[i] = ", dvlist[i].value
            d0[i] = dvlist[i].value
            
#        print "y0 = ", y0
#        print "x0 = ", x0
#        print "u0 = ", u0
#        print "d0 = ", d0
            
        # Define ode for the fired heater simulation.

        def model(x, u, d):

            # Discrete-time evolution for for fired heater.

            deltax = x - x0
            deltau = u - u0
            deltad = d - d0
            xplus = (mpc.mtimes(Ax, deltax) + mpc.mtimes(Bu, deltau)
                     + mpc.mtimes(Bd, deltad))  
            return xplus

        # Create a DummySimulator.

        htr = mpc.tools.DummySimulator(model, [Nx, Nu, Nd], ["x", "u", "d"])

        # Initialize the steady-state values

        ys = y0
        xs = x0
        us = u0
        ds = d0
        xaugs = np.concatenate((xs,np.zeros((Nid,))))
        
        # Define augmented model for state estimation.    

        # We need to define two of these because Ipopt isn't smart enough to throw out
        # the 0 = 0 equality constraints. ode_disturbance only gives dx/dt for the
        # actual states, and ode_augmented appends the zeros so that dx/dt is given for
        # all of the states. It's possible that Ipopt may figure this out by itself,
        # but we should just be explicit to avoid any bugs.    

        def model_augmented(x, u, d=ds):

            # Need to add identity for integrating disturbance evolution.
            xnom = x[:Nx]
            dhat = x[Nx:]
            xaugplus = mpc.vcat([model(xnom, u, d), dhat])
            return xaugplus
        
        def model_estimator(x, u, w=np.zeros(Nx + Nid), d=ds):
            
            # Model with additive state noise.
            return model_augmented(x, u, d) + w

        def model_sstarg(x, u, d=ds):
            
            # Model for steady state in terms of nominal states.
            xaugplus = model_augmented(x, u, d)
            return xaugplus[:Nx] - x[:Nx]


        # Turn into casadi functions.
        
        model_augmented_casadi = mpc.getCasadiFunc(model_augmented,
                                                   [Nx+Nid,Nu,Nd],
                                                   ["x","u","p"], "faug")

        model_estimator_casadi = mpc.getCasadiFunc(model_estimator,
                                                   [Nx+Nid,Nu,Nx+Nid,Nd],
                                                   ["x","u","w","p"], "fest")
        
        model_sstarg_casadi = mpc.getCasadiFunc(model_sstarg,
                                                [Nx+Nid,Nu,Nd],
                                                ["x","u","p"], "fdist")
        
        # Also define DummySimulator object.
        
        htraug = mpc.tools.DummySimulator(model_augmented, [Nx + Nid, Nu, Nd],
                                          ["x", "u", "d"])
        
        
        # Define measurement function.
        
        def measurement_augmented(x, d=ds):

            # For this case all of the integrating disturbances are output
            # disturbances.

            deltax = x[:Nx] - x0
            dhat   = x[Nx:]
            deltay = mpc.mtimes(Cx, deltax) + dhat
            ym     = deltay + y0
            return ym
        
        measurement_augmented_casadi = mpc.getCasadiFunc(measurement_augmented,
                                                         [Nx+Nid,Nd],
                                                         ["x","p"],
                                                         "measurement")

        # Weighting matrices for controller.

        Qy  = np.diag([cvlist[0].qvalue, cvlist[1].qvalue, cvlist[2].qvalue,
                       cvlist[3].qvalue, cvlist[4].qvalue])
        Qx  = mpc.mtimes(Cx.T,Qy,Cx)
        R   = np.diag([mvlist[0].rvalue, mvlist[1].rvalue, mvlist[2].rvalue])
        S   = np.diag([mvlist[0].svalue, mvlist[1].svalue, mvlist[2].svalue])

        # Now calculate the cost-to-go.

        [K, Pi] = mpc.util.dlqr(Ax, Bu, Qx, R)

        # Define control stage cost.

        def stagecost(x, u, xsp, usp, Deltau, s):
            dx = x[:Nx] - xsp[:Nx]
            du = u - usp
            return (mpc.mtimes(dx.T,Qx,dx) + .1*mpc.mtimes(du.T,R,du)
                + mpc.mtimes(Deltau.T,S,Deltau) + 1000*casadi.sum1(s))

        largs = ["x", "u", "x_sp", "u_sp", "Du", "s"]
        l = mpc.getCasadiFunc(stagecost, [Nx+Nid,Nu,Nx+Nid,Nu,Nu,Ny], largs,
                              funcname="l")

        # Define cost to go.

        def costtogo(x,xsp):
            dx = x[:Nx] - xsp[:Nx]
            return mpc.mtimes(dx.T, Pi, dx)
        Pf = mpc.getCasadiFunc(costtogo, [Nx+Nid,Nx+Nid], ["x", "x_sp"],
                               funcname="Pf")

        # Define output constraints for the controller.
        
        def outputconstraints(xaug, s):
            y = measurement_augmented(xaug)
            terms = [
                y - np.array(yub) - s,
                np.array(ylb) - y - s,
            ]
            return mpc.vcat(terms)
        e = mpc.getCasadiFunc(outputconstraints, [Nx + Nid, Ny], ["x", "s"],
                              funcname="e")

        # Build augmented estimator matrices.

        Qw = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        Rv = np.diag([cvlist[0].mnoise, cvlist[1].mnoise, cvlist[2].mnoise,
                      cvlist[3].mnoise, cvlist[4].mnoise])
        Qwinv = linalg.inv(Qw)
        Rvinv = linalg.inv(Rv)

        # Define stage costs for estimator.

        def lest(w, v):
            return mpc.mtimes(w.T, Qwinv, w) + mpc.mtimes(v.T, Rvinv, v)
                      
        lest = mpc.getCasadiFunc(lest, [Nw,Nv], ["w","v"], "l")

        # Don't use a prior.

        lxest = None
        x0bar = None

        # Check if the augmented system is detectable. (Rawlings and Mayne, Lemma 1.8)

        Aaug = mpc.util.getLinearizedModel(model_augmented_casadi,
                                           [xaugs, us, ds],
                                           ["A", "B", "Bd"])["A"]
        Caug = mpc.util.getLinearizedModel(measurement_augmented_casadi,
                                           [xaugs, ds],
                                           ["C", "Cd"])["C"]
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
            "f" : model_estimator_casadi,
            "h" : measurement_augmented_casadi,
            "u" : uguess,
            "y" : yguess,
            "l" : lest,
            "N" : {"x":Nx + Nid,"u":Nu, "y":Ny, "p":Nd, "t":Nmhe},
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
        
        Rss = R
#        Rss  = np.zeros((Nu,Nu))
        Qyss = Qy
#        Qyss = np.zeros((Ny,Ny))
#        Qyss[contVars,contVars] = 1 # We want to control all outputs
#        Qxss = mpc.mtimes(Cx.T,Qyss,Cx)

        def sstargobj(y, y_sp, u, u_sp):
            dy = y - y_sp
            du = u - u_sp
            return mpc.mtimes(dy.T, Qyss, dy) + mpc.mtimes(du.T, Rss, du)

        phiargs = ["y", "y_sp", "u", "u_sp"]
        phi = mpc.getCasadiFunc(sstargobj, [Ny, Ny, Nu, Nu], phiargs)

        sstargargs = {
            "f" : model_sstarg_casadi,
            "h" : measurement_augmented_casadi,
            "lb" : {"u" : np.tile(ulb, (1,1)), "y" : np.tile(ylb, (1,1))},
            "ub" : {"u" : np.tile(uub, (1,1)), "y" : np.tile(yub, (1,1))},
            "guess" : {
                "u" : np.tile(us, (1,1)),
                "x" : np.tile(np.concatenate((xs,np.zeros((Nid,)))), (1,1)),
                "y" : np.tile(ys, (1,1)),
            },
            "p" : np.tile(ds, (1,1)), # Parameters for system.
            "N" : {"x" : Nx + Nid, "u" : Nu, "y" : Ny, "p" : Nd, "f" : Nx},
            "phi" : phi,
            "inferargs" : True,
            "extrapar" : {"y_sp" : ys, "u_sp" : us},
            "verbosity" : 0,
            "discretef" : False,
            "casaditype" : "SX" if useCasadiSX else "MX",
        }
        targetfinder = mpc.sstarg(**sstargargs)

        # Make NMPC solver.

        duub = [ mvlist[0].roclim,  mvlist[1].roclim,  mvlist[2].roclim]
        dulb = [-mvlist[0].roclim, -mvlist[1].roclim, -mvlist[2].roclim]
        lb = {"u" : np.tile(ulb, (Nf,1)), "Du" : np.tile(dulb, (Nf,1))}
        ub = {"u" : np.tile(uub, (Nf,1)), "Du" : np.tile(duub, (Nf,1))}
        N = {"x": Nx+Nid, "u": Nu, "p": Nd, "t": Nf, "e": 2*Ny, "s": Ny}
        p = np.tile(ds, (Nf,1)) # Parameters for system.
        sp = {"x" : np.tile(xaugs, (Nf+1,1)), "u" : np.tile(us, (Nf,1))}
        guess = sp.copy()
        xaug0 = xaugs
        nmpcargs = {
            "f" : model_augmented_casadi,
            "l" : l,
            "inferargs" : True,
            "e" : e,
            "N" : N,
            "x0" : xaug0,
            "uprev" : us,
            "lb" : lb,
            "ub" : ub,
            "guess" : guess,
            "Pf" : Pf,
            "p" : p,
            "sp" : sp,
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
            xilist.vecassign(xs)
            xilist.vecassign(xs, "est")
            mvlist.vecassign(us)
            dvlist.vecassign(ds)
            cvlist.vecassign(dhat_k, "dist")
            #            dvlist[0].est = dhat_k

        # Store values in simulation container
        simcon.proc = [htr]
        simcon.mod = []
        simcon.mod.append(us)
        simcon.mod.append(xs)
        simcon.mod.append(ys)
        simcon.mod.append(ds)
        simcon.mod.append(estimator)
        simcon.mod.append(targetfinder)
        simcon.mod.append(controller)
        simcon.mod.append(htraug)
        simcon.mod.append(measurement_augmented)
        simcon.mod.append(psize)
        simcon.ydata = ydata
        simcon.udata = udata

    # Get stored values
    #TODO: these should be dictionaries.
    htr          = simcon.proc[0]
    us            = simcon.mod[0]
    xs            = simcon.mod[1]
    ys            = simcon.mod[2]
    ds            = simcon.mod[3]
    estimator     = simcon.mod[4]
    targetfinder  = simcon.mod[5]
    controller    = simcon.mod[6]
    htraug        = simcon.mod[7]
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
    x_km1 = xilist.asvec()
    u_km1 = mvlist.asvec()
    d_km1 = dvlist.asvec()    

#    print "d_km1 = ", d_km1
#    print "u_km1 = ", u_km1

    # Advance the process

    x_k = htr.sim(x_km1, u_km1, d_km1)

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
        xilist.vecassign(xhat_k, field, index=0)
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
                xof_k = htraug.sim(xof_km1, u_km1, ds)
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
            xilist.vecassign(xof_k[:Nx], field, index=(i + 1)) # Note [:Nx].
            cvlist.vecassign(yof_k[:Ny], field, index=(i + 1))
    
        xof_km1 = xof_k
    
    # calculate mpc input adjustment in control is on

    if (opnclsd.status.get() == 1):

        # Use nonlinear steady-state target selector

        ysp_k = [cvlist[0].setpoint, cvlist[1].setpoint, cvlist[2].setpoint,
                 cvlist[3].setpoint, cvlist[4].setpoint]
        usp_k = [mvlist[0].target, mvlist[1].target, mvlist[2].target]
        xtarget = np.concatenate((x_km1,dhat_k))

        # Previously had targetfinder.par["p",0] = d_km1, but this shouldn't
        # be because the target finder should be using the same model as the
        # controller and doesn't get to know the real disturbance.
        targetfinder.guess["x",0] = xtarget
        targetfinder.fixvar("x",0,dhat_k,range(Nx,Nx+Nid))
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
        xilist.vecassign(xhat_k, "clpred", index=0)
        cvlist.vecassign(yhat_k, "clpred", index=0)

        for i in range(Nf - 1):
            mvlist.vecassign(sol["u"][i+1,:], "clpred", index=(i + 1))
            xilist.vecassign(sol["x"][i+1,:Nx], "clpred", index=(i + 1))
            xcl_k = sol["x"][i+1,:]
            ycl_k = measurement(xcl_k)
            cvlist.vecassign(ycl_k[:Ny], "clpred", index=(i + 1))

    else:
        # Track the cv setpoints if the control is not on.
        cvlist[0].setpoint = y_k[0]
        cvlist[1].setpoint = y_k[1]
        cvlist[2].setpoint = y_k[2]

    # Store variable values
    mvlist.vecassign(u_k)
    xilist.vecassign(x_k)
    xilist.vecassign(xhat_k, "est")
#    dvlist[0].est = dhat_k
    cvlist.vecassign(y_k)
    cvlist.vecassign(yhat_k, "est")
    cvlist.vecassign(dhat_k, "dist")
    simcon.ydata    = ydata
    simcon.udata    = udata

# set up htr mpc example

simname = 'Fired Heater NMPC Example'

# define variables

MVmenu=["value","rvalue","svalue","maxlim","minlim","roclim","pltmax","pltmin"]
DVmenu=["value","pltmax","pltmin"]
XVmenu=["mnoise","noise","pltmax","pltmin"]
CVmenu=["setpoint","qvalue","maxlim","minlim","mnoise","noise","pltmax","pltmin"]
FVmenu=["mnoise","noise","pltmax","pltmin"]

MV1 = sim.MVobj(name='f1sp', desc='mv - pass 1 flow setpoint', units='(bph)',
            pltmin=80.0, pltmax=120.0, minlim=81.0, maxlim=119.0, svalue=1.0,
            value=100.0, target=100.0, Nf=60, menu=MVmenu)

MV2 = sim.MVobj(name='f2sp', desc='mv - pass 2 flow setpoint', units='(bph)', 
            pltmin=80.0, pltmax=120.0, minlim=81.0, maxlim=119.0, svalue=1.0,
            value=100.0, target=100.0, Nf=60, menu=MVmenu)

MV3 = sim.MVobj(name='fgsp', desc='fg - fuel gas flow setpoint', units='(scfh)', 
            pltmin=80.0, pltmax=120.0, minlim=81.0, maxlim=119.0, svalue=5.0,
            value=100.0, target=100.0, Nf=60, menu=MVmenu)

DV1 = sim.MVobj(name='t1in', desc='dv - pass 1 inlet temp', units='(degf)', 
            pltmin=520.0, pltmax=560.0,
            value=540.0, Nf=60, menu=DVmenu)

DV2 = sim.MVobj(name='t2in', desc='dv - pass 2 inlet temp', units='(degf)', 
            pltmin=520.0, pltmax=560.0,
            value=540.0, Nf=60, menu=DVmenu)

CV1 = sim.XVobj(name='toc', desc='cv - combined outlet temp', units='(degf)', 
            pltmin=700.0, pltmax=800.0, minlim=705.0, maxlim=795.0, noise=1.0,
            value=750.0, setpoint=750.0, Nf=60, menu=CVmenu)

CV2 = sim.XVobj(name='foc', desc='cv - combined outlet flow', units='(bph)', 
            pltmin=150.0, pltmax=250.0, minlim=155.0, maxlim=245.0, noise=1.0,
            value=200.0, setpoint=200.0, Nf=60, menu=CVmenu)

CV3 = sim.XVobj(name='dpt', desc='cv - delta pass temp', units='(degf)', 
            pltmin=-10.0, pltmax=10.0, minlim=-9.5, maxlim=9.5, noise=0.1,
                value=0.0, setpoint=0.0, qvalue=0.1, Nf=60, menu=CVmenu)

CV4 = sim.XVobj(name='t1s', desc='cv - pass 1 tubeskin temp', units='(degf)', 
            pltmin=870, pltmax=930, minlim=800.0, maxlim=920.0, noise=0.5,
                value=900.0, setpoint=900.0, qvalue=0.0, Nf=60, menu=CVmenu)

CV5 = sim.XVobj(name='t2s', desc='cv - pass 2 tubeskin temp', units='(degf)', 
            pltmin=870, pltmax=930, minlim=800.0, maxlim=920.0, noise=0.5,
                value=900.0, setpoint=900.0, qvalue=0.0, Nf=60, menu=CVmenu)

XI1  = sim.XIobj(name='x1',  desc='state 1',  value=0.0, Nf=60)
XI2  = sim.XIobj(name='x2',  desc='state 2',  value=0.0, Nf=60)
XI3  = sim.XIobj(name='x3',  desc='state 3',  value=0.0, Nf=60)
XI4  = sim.XIobj(name='x4',  desc='state 4',  value=0.0, Nf=60)
XI5  = sim.XIobj(name='x5',  desc='state 5',  value=0.0, Nf=60)
XI6  = sim.XIobj(name='x6',  desc='state 6',  value=0.0, Nf=60)
XI7  = sim.XIobj(name='x7',  desc='state 7',  value=0.0, Nf=60)
XI8  = sim.XIobj(name='x8',  desc='state 8',  value=0.0, Nf=60)
XI9  = sim.XIobj(name='x9',  desc='state 9',  value=0.0, Nf=60)
XI10 = sim.XIobj(name='x10', desc='state 10', value=0.0, Nf=60)

#XV1 = sim.XVobj(name='x1', desc='state 1', units='', 
#               pltmin=-100, pltmax=100, 
#               value=0.0, Nf=60, menu=XVmenu)
#
#XV2 = sim.XVobj(name='x2', desc='state 2', units='', 
#               pltmin=-100, pltmax=100, 
#               value=0.0, Nf=60, menu=XVmenu)
#
#XV3 = sim.XVobj(name='x3', desc='state 3', units='', 
#               pltmin=-100, pltmax=100, 
#               value=0.0, Nf=60, menu=XVmenu)
#
#XV4 = sim.XVobj(name='x4', desc='state 4', units='', 
#               pltmin=-100, pltmax=100, 
#               value=0.0, Nf=60, menu=XVmenu)
#
#XV5 = sim.XVobj(name='x5', desc='state 5', units='', 
#               pltmin=-100, pltmax=100, 
#               value=0.0, Nf=60, menu=XVmenu)
#
#XV6 = sim.XVobj(name='x6', desc='state 6', units='', 
#               pltmin=-100, pltmax=100, 
#               value=0.0, Nf=60, menu=XVmenu)
#
#XV7 = sim.XVobj(name='x7', desc='state 7', units='', 
#               pltmin=-100, pltmax=100, 
#               value=0.0, Nf=60, menu=XVmenu)
#
#XV8 = sim.XVobj(name='x8', desc='state 8', units='', 
#               pltmin=-100, pltmax=100, 
#               value=0.0, Nf=60, menu=XVmenu)
#
#XV9 = sim.XVobj(name='x9', desc='state 9', units='', 
#               pltmin=-100, pltmax=100, 
#               value=0.0, Nf=60, menu=XVmenu)
#
#XV10 = sim.XVobj(name='x10', desc='state 10', units='', 
#               pltmin=-100, pltmax=100, 
#               value=0.0, Nf=60, menu=XVmenu)

# define options

NF = sim.Option(name='NF', desc='Noise Factor', value=0.0)
OL = sim.Option(name="OL Pred.", desc="Open-Loop Predictions", value=1)

# load up variable lists

MVlist = [MV1, MV2, MV3]
DVlist = [DV1, DV2]
#XVlist = [XV1, XV2, XV3, XV4, XV5, XV6, XV7, XV8, XV9, XV10]
XVlist = []
XIlist = [XI1, XI2, XI3, XI4, XI5, XI6, XI7, XI8, XI9, XI10]
CVlist = [CV1, CV2, CV3, CV4, CV5]
OPlist = [NF, OL]
DeltaT = 1.0
N      = 120
refint = 100

# Define state-space matrices.
ssfhtr_Ax = np.array([
    [-3.0355482e-01, -9.4687895e-03, -2.5058438e-02, -7.0840563e-02, -5.0403247e-02, -1.8009285e-01, 1.1615073e-01, -2.5449122e-02, -6.7227306e-02, -4.6340924e-02],
    [4.2311708e-02, -7.3733419e-01, -4.6395958e-01, 6.6167445e-02, -4.0390266e-02, -3.3079095e-01, -8.3244322e-02, -3.9568920e-01, -2.4187538e-02, 4.1467954e-01],
    [-1.3299723e-02, 1.2621385e-02, -6.6231486e-01, 1.8486721e-03, 2.0530459e-01, 2.5768956e-01, -1.3697364e-01, -3.2933144e-03, -4.8252458e-02, -3.2291359e-01],
    [-7.2852359e-02, 3.2653051e-03, -1.6760562e-02, -2.5214563e-01, -3.5984133e-02, 2.1724470e-01, -1.2213044e-01, 2.5433870e-02, 4.5180118e-02, 1.2883216e-01],
    [-6.0323166e-02, 2.8840844e-01, 5.1406874e-01, -8.8476112e-02, -3.9102929e-01, 4.0722413e-01, 1.0124434e-01, -4.0870947e-02, 1.3927141e-01, -1.3936518e-01],
    [5.2782090e-02, -2.5659089e-01, -3.0062301e-01, 7.3552546e-02, -2.6303223e-01, -1.1588637e+00, 8.9395526e-02, 3.0668870e-01, -1.8727076e-01, 6.4019085e-02],
    [4.6386856e-02, -2.2522432e-01, -2.7396142e-01, 6.4893510e-02, -2.1866248e-01, -1.3865560e-01, -7.0756655e-01, 4.0031819e-01, -4.6027020e-02, 2.1779572e-01],
    [-2.0127948e-02, 9.9231876e-02, 6.5929159e-02, -2.6787203e-02, 1.5400933e-01, 6.1381712e-02, 9.6442036e-02, -1.0005813e+00, 6.8391056e-02, 7.1099606e-02],
    [-4.4841417e-02, 2.1285325e-01, 4.3622234e-01, -6.7169602e-02, 1.9979247e-02, 1.3009674e-01, 1.8128800e-01, -2.7544823e-01, -3.4747581e-01, 1.5897158e-01],
    [-7.9940997e-03, 4.1808992e-02, -5.8239471e-02, -8.4527513e-03, 1.5544784e-01, 2.6319107e-02, 4.7392659e-02, 7.9361893e-02, -1.1922762e-01, -9.7246722e-01],
])
            
ssfhtr_Bu = np.array([
    [9.3662232e-02, -7.8203259e-02, -3.5269595e-01],
    [-2.8832883e-01, -8.1735306e-02, -1.5928428e+00],
    [-5.8112677e-01, -2.1531239e-01, -1.1842419e+00],
    [-6.8495670e-02, 1.0765663e-01, 1.4634518e-02],
    [7.5756563e-02, 1.1694449e-01, 1.9691788e-01],
    [6.1514014e-01, -4.0972591e-01, -2.3744012e+00],
    [-7.9899097e-01, 1.4099062e-01, -9.3913025e-01],
    [-3.6762570e-01, -7.0357056e-01, 9.2546105e-01],
    [1.2768712e-01, -1.8466076e-01, 1.8963478e-01],
    [-2.0174781e-01, -9.2070482e-01, 6.8093529e-01],
])

ssfhtr_Bd = np.array([
    [4.4194612e-03, -5.4186514e-03],
    [5.6831596e-02, -2.6778227e-01],
    [2.5537917e-01, 4.0678375e-01],
    [-8.4833809e-03, -2.2615464e-02],
    [-4.9774661e-01, -4.5753829e-01],
    [1.9748539e-02, -1.1690683e-01],
    [-1.9320783e-01, 3.9676095e-02],
    [6.6823598e-01, -6.1718808e-01],
    [1.9161547e-01, 2.2758691e-01],
    [-5.6162635e-01, 5.0761204e-01],
])

ssfhtr_Cx = np.array([
    [1.3396477e+00, -3.6878574e-01, -6.8114063e-01, 8.7606379e-01, -4.6768349e-01, 1.3993241e-01, 7.3406388e-01, 1.3067031e-01, 1.8262439e-01, 3.4929682e-02],
    [-9.2186016e-04, 2.6716492e-02, -3.5855690e-02, 5.3214858e-04, -5.5698826e-02, -5.2814796e-02, -4.6559706e-01, -3.3686752e-01, -2.3554643e-01, -2.8711349e-01],
    [5.7258761e-01, -2.6913153e-01, 1.2789075e-01, -8.8485981e-01, 8.4953612e-02, -4.0000104e-02, 1.0430771e-01, 2.8622374e-02, -1.9509270e-01, -9.6295469e-02],
    [-4.3176751e-01, 2.3635803e-01, -2.6210683e-01, -4.0091055e-01, -1.4456333e+00, -3.2105879e-01, 8.7805840e-01, 5.4458178e-01, -7.2190928e-01, -1.4753603e-01],
    [-4.0975288e-01, 6.8432380e-02, -5.6246447e-01, -3.6622004e-01, -1.4926083e+00, 4.1723975e-01, 1.8132373e-01, 1.0962275e-01, -3.1289443e-01, 1.0522818e+00],
])            

def dosimulation():
    """Create the GUI and run the simulation."""
    simcon = sim.SimCon(simname=simname,
                    mvlist=MVlist, dvlist=DVlist, cvlist=CVlist, xvlist=XVlist,
                    xilist=XIlist, oplist=OPlist, N=N, refint=refint,
                    runsim=runsim, deltat=DeltaT)
    # build the GUI and start it up.
    plotspacing = dict(left=0.075, top=0.95, bottom=0.05, right=0.99,
                       hspace=0.5)
    sim.makegui(simcon, plotspacing=plotspacing)

if __name__ == "__main__":
    dosimulation()
