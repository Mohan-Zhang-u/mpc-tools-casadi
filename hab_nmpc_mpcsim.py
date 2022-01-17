# This is a hot-air balloon example

from mpctools import mpcsim as sim
import mpctools as mpc
import numpy as np
from scipy import linalg

useCasadiSX = True

# Define system sizes.
Nx   = 3            # number of states
Nu   = 2            # number of inputs
Ny   = 3            # number of outputs
Nid  = Ny           # number of integrating disturbances

Nw   = Nx + Nid     # number of augmented states
Nv   = Ny           # number of output measurements

# Define scaling factors
h0 = 1.1e4         # altitude scaling factor            (m)
T0 = 288.2         # air temperature at takeoff         (K)
f0 = 3672          # fuel flowrate at takeoff           (sccm)
p0 = 100.0         # vent position                      (%)
t0 = 33.49         # time scale factor                  (s)

# Define parameters for hot-air balloon
alpha  = 2.549     # balloon number
omega  = 20.0      # drag coefficient
beta   = 0.1116    # heat transfer coefficient
gamma  = 5.257     # atmosphere number
delta  = 0.2481    # temperature drop-off coefficient
lambde = 1.00      # vent coefficient

# Define ode and measurement function for the hot-air balloon.
def ode(x, u):
    """ODE for hot air balloon system."""
    f     = (1 + 0.03*u[0])*100/f0
    term1 = alpha*(1 - delta*x[0])**(gamma - 1)
    term2 = (1 - (1 - delta*x[0])/x[2])
    term3 = beta*(x[2] - 1 + delta*x[0])
    term4 = (1 + lambde*u[1]/p0)
    term5 = omega*x[1]*np.fabs(x[1])
    dx1dt = x[1]
    dx2dt = term1*term2 - 0.5 - term5
    dx3dt = -term3*term4 + f
    dxdt = np.array([dx1dt, dx2dt, dx3dt])
    return dxdt

def measurement(x):
    """Augmented system measurement (all output disturbances)."""
    ym = x[:Nx] + x[Nx:] # Nominal states plus output disturbances.
    ym = ym*np.array([h0, h0/t0, T0]) # Apply scaling.
    ym[2] -= 273.2 # Convert from K to C.
    return ym

# Define initial conditions for x, u, and y.
x_init = np.array([0, 0, 1.244, 0, 0, 0])
u_init = np.array([0, 0])
y_init = measurement(x_init)

# Define simulation functions.
def runsim(k, simcon, opnclsd):

    print("runsim: iteration %d -----------------------------------" % k)

    # Unpack stuff from simulation container.

    mvlist = simcon.mvlist
    cvlist = simcon.cvlist
    xvlist = simcon.xvlist
    oplist = simcon.oplist
    deltat = simcon.deltat
    nf = oplist[0]
    doolpred = oplist[1]
    fuelincrement = oplist[2].value
    usenonlinmpc = bool(oplist[3].value)
    usenonlinmhe = usenonlinmpc

    # Check for changes.

    chsum = 0

    for var in mvlist:
        chsum += var.chflag
        var.chflag = 0
    
    for var in xvlist:
        chsum += var.chflag
        var.chflag = 0
    
    for var in cvlist:
        chsum += var.chflag
        var.chflag = 0
    
    for var in oplist:
        chsum += var.chflag
        var.chflag = 0
    
    # Grab bounds.

    uub = [mvlist[0].maxlim, mvlist[1].maxlim]
    ulb = [mvlist[0].minlim, mvlist[1].minlim]
    yub = [cvlist[0].maxlim, cvlist[1].maxlim, cvlist[2].maxlim]
    ylb = [cvlist[0].minlim, cvlist[1].minlim, cvlist[2].minlim]
    xlb = [-0.1,-0.12, 1.1,-3500.0,-30.0,-30.0] # Range of model validity.
    xub = [1.0, 0.12, 1.5, 3500.0, 30.0, 30.0] # Set at ~50% above and below plot limits
    
    # Initialize values on first execution or when something changes.

    if (k == 0 or chsum > 0):

        print("runsim: initialization")

        # Define other problem size parameters.
        
        Nf   = cvlist[0].Nf # length of NMPC future horizon
        Nmhe = 60           # length of MHE past horizon
        psize = (Nf, Nmhe)

        # Define sample time in minutes.
        
        Delta = deltat

        # Set the initial values.

        y0 = np.zeros((Ny,))
        x0 = np.zeros((Nx,))
        u0 = np.zeros((Nu,))
        for i in range(Ny):
            y0[i] = cvlist[i].value
        for i in range(Nx):
            x0[i] = xvlist[i].value
        for i in range(Nu):
            u0[i] = mvlist[i].value
        xaug0 = np.concatenate((x0, np.zeros(Nid)))

        # Create simulator.

        hab = mpc.DiscreteSimulator(ode, Delta, [Nx,Nu], ["x","u"])

        # Initialize the steady-state values

        ys = y0
        xs = x0
        us = u0
        xaugs = np.concatenate((xs, np.zeros(Nid)))
        
        # Define augmented model for state estimation.

        def ode_augmented(x, u):

            # Need to add extra zeros for derivative of disturbance states.

            dxdt = mpc.vcat([ode(x[:Nx], u), np.zeros((Nid,))])
            return dxdt
 
        habaug = mpc.DiscreteSimulator(ode_augmented, Delta,
                                        [Nx+Nid,Nu], ["x","u"])

        # Turn into casadi functions.
        ode_augmented_casadi = mpc.getCasadiFunc(ode_augmented,
                                                 [Nx + Nid, Nu], ["x", "u"],
                                                 funcname="ode_augmented")

        ode_augmented_rk4_casadi = mpc.getCasadiFunc(ode_augmented,
                                   [Nx+Nid,Nu],["x","u"],"ode_augmented_rk4",
                                     rk4=True,Delta=Delta,M=2)

        def ode_estimator_rk4(x,u,w=np.zeros((Nx+Nid,))):
            return ode_augmented_rk4_casadi(x, u) + w

        ode_estimator_rk4_casadi = mpc.getCasadiFunc(ode_estimator_rk4,
                                   [Nx+Nid,Nu,Nw], ["x","u","w"],
                                   "ode_estimator_rk4")

        measurement_casadi = mpc.getCasadiFunc(measurement,
                             [Nx+Nid], ["x"], "measurement")        
        
        # Get linearized model.
        xlin = x_init
        ulin = u_init
        
        linmodelpar = mpc.util.getLinearizedModel(ode_augmented_rk4_casadi,
                                                  [xlin, ulin],
                                                  ["A", "B"])
        linmodelpar["xlin"] = xlin
        linmodelpar["ulin"] = ulin
        
        linmodelpar["C"] = mpc.util.getLinearizedModel(measurement_casadi,
                                                       [xlin], ["C"])["C"]
        linmodelpar["ylin"] = measurement(xlin)
        
        # Build augmented estimator matrices.

        Qw = np.diag([xv.mnoise for xv in xvlist] + [xv.dnoise for xv in xvlist])
        Rv = np.diag([cv.mnoise for cv in cvlist])
        Qwinv = linalg.inv(Qw)
        Rvinv = linalg.inv(Rv)

        # Define stage costs for estimator.

        def lest(w,v):
            return mpc.mtimes(w.T,Qwinv,w) + mpc.mtimes(v.T,Rvinv,v)
                      
        lest = mpc.getCasadiFunc(lest, [Nw,Nv], ["w","v"], "l")

        # Don't use a prior.

        lxest = None
        x0bar = None

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
            "N" : {"x":Nx + Nid, "u":Nu, "y":Ny, "t":Nmhe},
            "lb" : {"x" : np.tile(xlb, (Nmhe + 1, 1))},
            "ub" : {"x" : np.tile(xub, (Nmhe + 1, 1))},
            "lx" : lxest,
            "inferargs" : True,
            "x0bar" : x0bar,
            "verbosity" : 0,
            "guess" : {"x":xguess, "y":yguess, "u":uguess},
            "timelimit" : 10,
            "casaditype" : "SX" if useCasadiSX else "MX",
        }
        estimator = mpc.nmhe(**nmheargs)

        # Calculate Kalman Filter.
        kalmanfilter = {k : linmodelpar[k] for k in ["A", "C"]}
        kalmanfilter["Q"] = Qw
        kalmanfilter["R"] = Rv
        [kalmanfilter["L"], kalmanfilter["P"]] = mpc.util.dlqe(**kalmanfilter)
        kalmanfilter.update({k : linmodelpar[k] for k in ["B", "xlin", "ulin", "ylin"]})
        kalmanfilter["xhat"] = xaug0
        
        def kf_filter(kf, u, y):
            """
            Performs a Kalman Filtering step.
            
            First argument shoudl be a dict with fields A, B, L, and xhat
            (with xhat giving xhat(k - 1 | k - 1). Other arguments u(k), and
            y(k).
            
            Note that the kf dict is updated with the new xhat.
            """
            xhat = kf["xhat"]
            xlin = kf["xlin"]
            ulin = kf["ulin"]
            ylin = kf["ylin"]
            xhatm = kf["A"].dot(xhat - xlin) + kf["B"].dot(u - ulin) + xlin
            yhatm = kf["C"].dot(xhatm - xlin) + ylin
            xhat = xhatm + kf["L"].dot(y - yhatm)
            return xhat
        kalmanfilter["filter"] = kf_filter

        # Declare ydata and udata. Note that it would make the most sense to declare
        # these using collection.deques since we're always popping the left element or
        # appending a new element, but for these sizes, we can just use a list without
        # any noticable slowdown.

        if (k == 0):
            ydata = [y_init.copy() for i in range(Nmhe)]
            udata = [u_init.copy() for i in range(Nmhe - 1)]
        else:
            ydata = simcon.ydata
            udata = simcon.udata

        # Weighting matrices for controller.

        Qy  = np.diag([cvlist[0].qvalue, cvlist[1].qvalue, cvlist[2].qvalue])
        Cx = linmodelpar["C"][:,:Nx]
        Qx  = mpc.mtimes(Cx.T, Qy, Cx)
        R   = np.diag([mvlist[0].rvalue, mvlist[1].rvalue])
        S   = np.diag([mvlist[0].svalue, mvlist[1].svalue])

        # Make steady-state target selector.

        Rss = R
        Qyss = Qy
        
        lbslack = np.array([[cv.lbslack for cv in cvlist]])
        ubslack = np.array([[cv.ubslack for cv in cvlist]])
        
        def sstargobj(y, y_sp, u, u_sp, Q, R, s):
            dy = y - y_sp
            du = u - u_sp
            slb = s[:Ny]
            sub = s[Ny:]
            slack = mpc.mtimes(lbslack, slb) + mpc.mtimes(ubslack, sub)
            return mpc.mtimes(dy.T,Q,dy) + mpc.mtimes(du.T,R,du) + slack

        phiargs = ["y", "y_sp", "u", "u_sp", "Q", "R", "s"]
        phi = mpc.getCasadiFunc(sstargobj, [Ny,Ny,Nu,Nu,(Ny,Ny),(Nu,Nu),2*Ny],
                                phiargs)

        # Add slacked output constraints.
        def outputcon(x, s):
            """Softened output constraints."""
            y = measurement(x)
            slb = s[:Ny]
            sub = s[Ny:]
            terms = (
                np.array(ylb) - y - slb,
                y - np.array(yub) - sub,
            )
            return np.concatenate(terms)
        outputcon_casadi = mpc.getCasadiFunc(outputcon, [Nx + Nid, 2*Ny],
                                             ["x", "s"], funcname="e")
        
        sstargargs = {
            "f" : ode_augmented_casadi,
            "ignoress" : range(Nx, Nx + Nid), # Ignore integrating disturbances.
            "h" : measurement_casadi,
            "lb" : {"u" : np.tile(ulb, (1,1)), "x" : np.tile(xlb, (1, 1))},
            "ub" : {"u" : np.tile(uub, (1,1)), "x" : np.tile(xub, (1, 1))},
            "guess" : {
                "u" : np.tile(us, (1,1)),
                "x" : np.tile(np.concatenate((xs,np.zeros((Nid,)))), (1,1)),
                "y" : np.tile(ys, (1,1)),
            },
            "N" : {"x" : Nx + Nid, "u" : Nu, "y" : Ny, "f" : Nx, "e" : 2*Ny,
                   "s" : 2*Ny},
            "phi" : phi,
            "inferargs" : True,
            "e" : outputcon_casadi,
            "extrapar" : {"R" : Rss, "Q" : Qyss, "y_sp" : ys, "u_sp" : us},
            "verbosity" : 0,
            "discretef" : False,
            "timelimit" : 10,
            "casaditype" : "SX" if useCasadiSX else "MX",
        }
        targetfinder = mpc.sstarg(**sstargargs)

        # Define control stage cost.
        def stagecost(x, u, xsp, usp, Deltau, s):
            dx = x[:Nx] - xsp[:Nx]
            du = u - usp
            slb = s[:Ny]
            sub = s[Ny:]
            slack = mpc.mtimes(lbslack, slb) + mpc.mtimes(ubslack, sub)
            return (mpc.mtimes(dx.T,Qx,dx) + mpc.mtimes(du.T,R,du)
                + mpc.mtimes(Deltau.T,S,Deltau) + slack)

        largs = ["x","u","x_sp","u_sp","Du","s"]
        l = mpc.getCasadiFunc(stagecost,
            [Nx+Nid,Nu,Nx+Nid,Nu,Nu,2*Ny],largs,funcname="l")

        # Define cost to go.

        def costtogo(x,xsp):
            dx = x[:Nx] - xsp[:Nx]
            return mpc.mtimes(dx.T, Qx, dx)
        Pf = mpc.getCasadiFunc(costtogo,[Nx+Nid,Nx+Nid],["x","x_sp"],
                               funcname="Pf")
    
        # Make NMPC solver.

        duub = [ mvlist[0].roclim,  mvlist[1].roclim]
        dulb = [-mvlist[0].roclim, -mvlist[1].roclim]
        lb = {"u" : np.tile(ulb, (Nf,1)), "Du" : np.tile(dulb, (Nf,1)),
              "x" : np.tile(xlb, (Nf + 1, 1))}
        ub = {"u" : np.tile(uub, (Nf,1)), "Du" : np.tile(duub, (Nf,1)),
              "x" : np.tile(xub, (Nf + 1, 1))}
        N = {"x": Nx + Nid, "u": Nu, "t": Nf, "s": 2*Ny, "e": 2*Ny}
        sp = {"x" : np.tile(xaugs, (Nf+1,1)), "u" : np.tile(us, (Nf,1))}
        guess = sp.copy()
        nmpcargs = {
            "f" : ode_augmented_rk4_casadi,
            "l" : l,
            "inferargs" : True,
            "N" : N,
            "x0" : xaug0,
            "uprev" : us,
            "lb" : lb,
            "ub" : ub,
            "guess" : guess,
            "Pf" : Pf,
            "sp" : sp,
            "e" : outputcon_casadi,
            "verbosity" : 0,
            "timelimit" : 10,
            "casaditype" : "SX" if useCasadiSX else "MX",
        }
        controller = mpc.nmpc(**nmpcargs)

        # Calculate LQR.
        
        lqr = dict()
        lqr["A"] = linalg.block_diag(linmodelpar["A"][:Nx,:Nx], np.zeros((Nu, Nu)))
        lqr["B"] = np.vstack((linmodelpar["B"][:Nx,:], np.eye(Nu)))
        lqr["Q"] = linalg.block_diag(Qx, S)
        lqr["R"] = R + S
        lqr["M"] = np.vstack((np.zeros((Nx, Nu)), -S))
        [lqr["K"], lqr["P"]] = mpc.util.dlqr(**lqr)
        lqr["xlin"] = np.concatenate((xlin[:Nx], np.zeros(Nu)))
        lqr["ulin"] = ulin

        # Store values in simulation container
        simcon.proc = [hab]
        simcon.mod = (us, xs, ys, estimator, targetfinder, controller, habaug,
                      psize)
        simcon.ydata = ydata
        simcon.udata = udata
        simcon.extra["quanterr"] = 0
        simcon.extra["mpcfunc"] = ode_augmented_rk4_casadi
        simcon.extra["mhefunc"] = ode_estimator_rk4_casadi
        simcon.extra["kalmanfilter"] = kalmanfilter
        simcon.extra["lqr"] = lqr

    # Get stored values
    #TODO: these should be dictionaries or NamedTuples.
    hab           = simcon.proc[0]
    (us, xs, ys, estimator, targetfinder, controller, habaug, psize) = simcon.mod
    (Nf, Nmhe) = psize
    ydata         = simcon.ydata
    udata         = simcon.udata

    # Get variable values
    x_km1 = xvlist.asvec()
    u_km1 = mvlist.asvec()

    # Advance the process

    x_k = hab.sim(x_km1, u_km1)

    # Constrain the altitude and velocity states.
    
    if x_k[0] <= 0:
        x_k[1] = max(x_k[1], 0) # No negative velocity on the ground.
    x_k[0] = max(x_k[0], 0) # No negative altitude.

    # Take plant measurement

    y_k = measurement(np.concatenate((x_k,np.zeros((Nid,)))))

    if nf.value > 0.0:
        for i in range(0, Ny):
            y_k[i] += nf.value*np.random.normal(0.0, cvlist[i].noise)
    
    # Do Nonlinear MHE.

    ydata.append(y_k)
    udata.append(u_km1) 
    estimator.par["y"] = ydata
    estimator.par["u"] = udata
    
    if usenonlinmhe:
        estimator.solve()
        estsol = mpc.util.casadiStruct2numpyDict(estimator.var)
        status = estimator.stats["status"]
        xaughat_k = estsol["x"][-1,:]
        estimator.saveguess()
    else:
        kalmanfilter = simcon.extra["kalmanfilter"]
        xaughat_k = kalmanfilter["filter"](kalmanfilter, u_km1, y_k)
        xaughat_k = np.clip(xaughat_k, xlb, xub) # Make sure estimate is valid.
        status = "Kalman Filter"
    simcon.extra["kalmanfilter"]["xhat"] = xaughat_k

    print("runsim: estimator status - %s" % status)
    xhat_k = xaughat_k[:Nx]
    dhat_k = xaughat_k[Nx:]

    yhat_k = measurement(np.concatenate((xhat_k, dhat_k)))
    ydata.pop(0)
    udata.pop(0)

    # Initialize the input
    u_k = u_km1

    # Update open and closed-loop predictions
    for field in ["olpred", "clpred"]:
        mvlist.vecassign(u_k, field, index=0)
        xvlist.vecassign(xhat_k, field, index=0)
        cvlist.vecassign(yhat_k, field, index=0)
    
    xof_km1 = np.concatenate((xhat_k,dhat_k))

    # Need to be careful about this forecasting. Temporarily aggressive control
    # could cause the system to go unstable if continued indefinitely, and so
    # this simulation might fail. If the integrator fails at any step, then we
    # just return NaNs for future predictions. Also, if the user doesn't want
    # predictions, then we just always skip them.
    predictionOkay = bool(doolpred.value)
    xmask = np.array([1]*Nx + [0]*Nid)
    y_lb_safe = 0.5*(ylb + measurement(xmask*xlb))
    y_ub_safe = 0.5*(yub + measurement(xmask*xub))
    for i in range(0,(Nf - 1)):
        if predictionOkay:
            try:
                xof_k = habaug.sim(xof_km1, u_km1)
            except RuntimeError: # Integrator failed.
                predictionOkay = False
        if predictionOkay:
            # Take measurement.
            yof_k = measurement(xof_k)
            
            # Stop forecasting if bounds are exceeded.
            if np.any(yof_k > y_ub_safe) or np.any(yof_k < y_lb_safe):
                predictionOkay = False
        else:
            xof_k = np.NaN*np.ones((Nx+Nid,))
            yof_k = np.NaN*np.ones((Ny,))

        for field in ["olpred", "clpred"]:
            mvlist.vecassign(u_k, field, index=(i + 1))
            xvlist.vecassign(xof_k[:Nx], field, index=(i + 1)) # Note [:Nx].
            cvlist.vecassign(yof_k[:Ny], field, index=(i + 1))
        xof_km1 = xof_k
    
    # Find a valid steady state to use as the origin for the linearized model.
    
    ysp_k = [cvlist[0].setpoint, cvlist[1].setpoint, cvlist[2].setpoint]
    usp_k = [mvlist[0].target, mvlist[1].target]
    xtarget = np.concatenate((x_km1,dhat_k))
    
    targetfinder.guess["x",0] = xtarget
    targetfinder.fixvar("x", 0, dhat_k, range(Nx,Nx+Nid))
    targetfinder.par["y_sp",0] = ysp_k
    targetfinder.par["u_sp",0] = usp_k
    targetfinder.guess["u",0] = u_km1
    if opnclsd.status.get() == 0:
        # Want to find the steady state for the given input.
        targetfinder.fixvar("u", 0, u_km1)
    else:
        # Let the target finder pick u.
        targetfinder.lb["u",0] = ulb
        targetfinder.ub["u",0] = uub
    targetfinder.solve()

    xaugss = np.squeeze(targetfinder.var["x",0,:])
    uss = np.squeeze(targetfinder.var["u",0,:])

    print("runsim: target status - %s (Obj: %.5g)" % (targetfinder.stats["status"],targetfinder.obj))
    
    # Update Kalman Filter steady state.
    simcon.extra["kalmanfilter"].update(xlin=xaugss, ulin=uss,
                                        ylin=measurement(xaugss))    
    
    # calculate mpc input adjustment if control is on

    if (opnclsd.status.get() == 1):
        
        # Choose model for simulation.
        if usenonlinmpc:
            mpcmodel_ = simcon.extra["mpcfunc"]
        else:
            kf = simcon.extra["kalmanfilter"]
            def mpcmodel_(x, u):
                """Linear model for mpc."""
                return (kf["A"].dot(x - xaugss)
                        + kf["B"].dot(u - uss)
                        + xaugss)
        mpcmodel = mpc.tools.DummySimulator(mpcmodel_, [Nx + Nid, Nu], ["x", "u"])
        
        # Now use nonlinear MPC controller.
        if usenonlinmpc:
            controller.par["x_sp"] = [xaugss]*(Nf + 1)
            controller.par["u_sp"] = [uss]*Nf
            controller.par["u_prev"] = [u_km1]
            controller.fixvar("x",0,xaughat_k)           
            controller.solve()
            status = controller.stats["status"]
            obj = controller.obj
            controller.saveguess()
            sol = mpc.util.casadiStruct2numpyDict(controller.var)
        else:
            sol = dict(u=np.zeros((Nf, Nu)), x=np.zeros((Nf + 1, Nx + Nid)))
            sol["x"][0,:] = xaughat_k
            sol["x"][1:,Nx:] = np.tile(xaughat_k[Nx:], (Nf, 1))
            lqr = simcon.extra["lqr"]
            uprev = u_km1
            for t in range(Nf):
                z_ = np.concatenate((sol["x"][t,:Nx] - xaugss[:Nx], uprev - uss))
                if t == 0:
                    obj = mpc.mtimes(z_.T, lqr["P"], z_)
                sol["u"][t,:] = (lqr["K"].dot(z_) + uss).clip(ulb, uub)
                sol["x"][t + 1,:] = mpcmodel.sim(sol["x"][t,:], sol["u"][t,:])
                uprev = sol["u"][t,:]
            status = "LQR"
        print("runsim: controller status - %s (Obj: %.5g)" % (status, obj)) 
        
        # Apply quantization.
        
        if fuelincrement > 1e-3:
            minfuel = ulb[0]
            maxfuel = uub[0]
            
            # Use cumulative rounding strategy.
            quantum = (maxfuel - minfuel)*fuelincrement
            wantfuel = sol["u"][:,0]/quantum
            Nmax = np.floor(maxfuel/quantum)
            Nmin = np.ceil(minfuel/quantum)
            wantsofar = 0
            getfuel = []
            getsofar = simcon.extra["quanterr"]/quantum
            for want in wantfuel:
                wantsofar += want
                get = np.clip(round(wantsofar - getsofar), Nmin, Nmax)
                getfuel.append(get)
                getsofar += get
            simcon.extra["quanterr"] += (getfuel[0] - wantfuel[0])*quantum
            sol["u"][:,0] = np.array(getfuel)*quantum
            
            # Re-simulate the x trajectory.
            for i in range(sol["u"].shape[0]):
                sol["x"][i + 1,:] = mpcmodel.sim(sol["x"][i,:], sol["u"][i,:])
        else:
            simcon.extra["quanterr"] = 0
        
        print("runsim: quantization offset: %g" % simcon.extra["quanterr"])
        u_k = np.squeeze(sol["u"][0,:])

        # Update closed-loop predictions

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
        # Track the cv setpoints if the control is not on.
        cvlist[0].setpoint = y_k[0]
        cvlist[1].setpoint = y_k[1]
        cvlist[2].setpoint = y_k[2]

    # Store variable values
    mvlist.vecassign(u_k)
    xvlist.vecassign(x_k)
    xvlist.vecassign(xhat_k, "est")
#    dvlist[0].est = dhat_k
    cvlist.vecassign(y_k)
    cvlist.vecassign(yhat_k, "est")
    cvlist.vecassign(dhat_k, "dist")
    simcon.ydata    = ydata
    simcon.udata    = udata

# set up hab mpc example

simname = 'Hot-Air Ballon Example'

# define variables

MVmenu=["value","rvalue","svalue","target","maxlim","minlim","roclim","pltmax","pltmin"]
XVmenu=["mnoise","dnoise","pltmax","pltmin"]
CVmenu=["setpoint","qvalue","maxlim","minlim","mnoise","noise","pltmax","pltmin","lbslack","ubslack"]
DVmenu=["value","pltmax","pltmin"]

MV1 = sim.MVobj(name='f', desc='fuel flow setpoint', units='(%)',
            pltmin=-5.0, pltmax=105.0, minlim=0.0, maxlim=100.0, svalue=9.0e-4,
            rvalue=9.0e-7, value=u_init[0], target=0.0, Nf=60, menu=MVmenu)

MV2 = sim.MVobj(name='p', desc='top vent position', units='(%)', 
            pltmin=0.0, pltmax=100.0, minlim=1.0, maxlim=99.0, svalue=1.0e-4,
            rvalue=1.0e-4, value=u_init[1], target=0.0, Nf=60, menu=MVmenu)

CV1 = sim.CVobj(name='h', desc='altitude', units='(m)', 
            pltmin=-300.0, pltmax=7300.0, minlim=0.0, maxlim=7000.0, qvalue=1e-6, noise=1.0,
            value=y_init[0], setpoint=0.0, Nf=60, lbslack=1000, ubslack=1000, menu=CVmenu)

CV2 = sim.CVobj(name='v', desc='vertical velocity', units='(m/s)', 
            pltmin=-25.0, pltmax=25.0, minlim=-23.0, maxlim=23.0, qvalue=0.0, noise=1.0,
            value=y_init[1], setpoint=0.0, Nf=60, lbslack=1000, ubslack=1000, menu=CVmenu)

CV3 = sim.CVobj(name='T', desc='bag temperature', units='(degC)', 
            pltmin=55.0, pltmax=125.0, minlim=60.0, maxlim=120.0, qvalue=0.0, noise=0.1,
            value=y_init[2], setpoint=85.0, Nf=60, lbslack=1000, ubslack=1000, menu=CVmenu)

XV1 = sim.XVobj(name='h', desc='dim. altitude', units='', 
            pltmin=-0.1, pltmax=0.7, mnoise=1, dnoise=0.001,
            value=x_init[0], Nf=60, menu=XVmenu)

XV2 = sim.XVobj(name='v', desc='dim. vertical velocity', units='', 
            pltmin=-0.08, pltmax=0.08, mnoise=1, dnoise=0.001,
            value=x_init[1], Nf=60, menu=XVmenu)

XV3 = sim.XVobj(name='T', desc='dim. bag temperature', units='', 
            pltmin=1.19, pltmax=1.4, mnoise=1, dnoise=0.001,
            value=x_init[2], Nf=60, menu=XVmenu)

# define options

NF = sim.Option(name='NF', desc='Noise Factor', value=0.0)
OL = sim.Option(name="OL Pred.", desc="Open-Loop Predictions", value=1)
fuel = sim.Option(name="Fuel increment", desc="Fuel increment", value=0)
nonlinmpc = sim.Option("Nonlin. MPC", desc="Nonlinear MPC", value=False)

# load up variable lists

MVlist = [MV1, MV2]
XVlist = [XV1, XV2, XV3]
CVlist = [CV1, CV2, CV3]
OPlist = [NF, OL, fuel, nonlinmpc]
DeltaT = 0.5
N      = 120
refint = 100
simcon = sim.SimCon(simname=simname,
                    mvlist=MVlist, cvlist=CVlist, xvlist=XVlist,
                    oplist=OPlist, N=N, refint=refint, runsim=runsim, deltat=DeltaT)

# build the GUI and start it up.
plotspacing = dict(left=0.075, top=0.95, bottom=0.05, right=0.99, hspace=0.5)

def dosimulation():
    """Build the GUI and start the simulation."""
    sim.makegui(simcon, plotspacing=plotspacing)

if __name__ == "__main__":
    dosimulation()
