import numpy as np
import casadi
import casadi.tools as ctools
import warnings

# Other things from our package.
from . import util
from . import solvers
from . import colloc

"""
Functions for solving MPC problems using Casadi and Ipopt.
"""

# Using a numpy array of casadi MX symbols is almost always a bad idea, so we
# warn the user if they request it. Users who know what they are doing can
# disable the warning via this constant (see __getCasadiFunc).
WARN_NUMPY_MX = True

# =================================
# MPC and MHE
# =================================

# Note that using dictionaries as default arguments is okay in these functions
# because we always (should) make sure that the dictionary is copied internally
# before modifying it.

def nmpc(f=None, l=None, N={}, x0=None, lb={}, ub={}, guess={}, g=None,
         Pf=None, sp={}, p=None, uprev=None, verbosity=5, timelimit=60,
         Delta=None, funcargs={}, extrapar={}, e=None, ef=None, periodic=False,
         discretel=True, isQP=False, casaditype="SX", infercolloc=None,
         solver=None, udiscrete=None, inferargs=False):
    """
    Solves nonlinear MPC problem.
    
    N muste be a dictionary with at least entries "x", "u", and "t". If 
    parameters are present, you must also specify a "p" entry, and if algebraic
    states are present, you must provide a "z" entry.
    
    If provided, p must be a 2D array with the time dimension first. It should
    have N["t"] rows and N["p"] columns. This is for time-varying parameters.
    Note that they must be specified as a vector. For time-invariant parameters
    that may have weird sizes (e.g., a terminal penalty matrix that you may
    want to change), specify the numerical value in an entry of extrapar. Note
    that the names in extrapar must not conflict with default variable names
    like 'u', 'u_sp', 'Du', etc.
    
    lb and ub should be dictionaries of bounds for the various variables. Each
    entry should have time as the first index (i.e. lb["x"] should be a
    N["t"] + 1 by N["x"] array). If the bounds are not time-varying, you may
    omit the first dimension. guess should have the same structure. For bounds
    on the final state x, you may also specify an "xf" entry in lb or ub, which
    should be a N["x"] array.
    
    sp is a dictionary that holds setpoints for x and u. If supplied, the stage
    cost is assumed to be a function l(x,u,x_sp,u_sp). If not supplied, l is
    l(x,u). To explicitly specify a different order of arguments or something
    else, e.g. a dependence on parameters, specify a list of input variables
    in funcargs as described in the next paragraph. Similarly, Pf is assumed to
    be Pf(x,x_sp) if a setpoint for x is supplied, and it is left as Pf(x)
    otherwise.    
    
    Function argument are assumed to be the "usual" order, i.e. f(x,u), l(x,u),
    and Pf(x). If you wish to override any of these, specify a list of variable
    names in the corresponding entry of funcargs. E.g., for a stage cost
    l(x,u,x_sp,u_sp,Du), specify funcargs={"l" : ["x","u","x_sp","u_sp","Du"]}.
    Alternatively, inferargs decides whether to infer argument names from the
    Casadi Functions. Any functions not given in funcargs will have their named
    arguments inferred. Terminal constraints can be specified in ef, but
    arguments must be given, and u cannot be included since there is no u(N)
    variable.
    
    To include rate of change penalties or constraints, set uprev so a vector
    with the previous u entry. Bound constraints can then be entered with the
    key "Du" in the lb and ub structs. "Du" can also be specified as in largs.    
    
    The functions e and ef are constraints and terminal constraints,
    respectively. These should be Casadi functions, and the feasible region is
    defined by e <= 0 and ef <= 0. If either functions are specified, you must
    also specify the arguments to each in funcargs or set inferargs=True.
    To soften the e constraints, you can specify an N["s"] entry to specify
    how many slacks you need. These variables are nonnegative and should be
    penalized in the stage cost l. Similarly, for ef, specify N["sf"], and
    penalize it in Pf.    
    
    solver is a string specifying which solver to use. By default, the solver
    is chosen based on the problem type: qpoases if isQP, bonmin if any
    component of u is discrete, and ipopt otherwise.    
    
    udiscrete should be an Nu vector of True and False to say whether u has
    any discrete components. Note that this setting is not supported for all
    solvers.    
    
    The return value is a ControlSolver object. To actually solve the
    optimization, use ControlSolver.solve().
    """
    # Copy dictionaries so we don't change the user inputs.
    N = N.copy()
    guess = guess.copy()
    funcargs = funcargs.copy()
    
    # Also, make sure certain inputs are dictionaries of numpy arrays. This
    # also forces a copy, so we don't have to worry about modifying the user's
    # dictionaries.    
    lb = util.ArrayDict(lb)
    ub = util.ArrayDict(ub)
    guess = util.ArrayDict(guess)
    sp = util.ArrayDict(sp)
    extrapar = util.ArrayDict(extrapar)
    if infercolloc is None:
        infercolloc = ("x" in guess and "xc" not in guess)
   
    # Check specified sizes.
    try:
        for i in ["t","x"]:
            if N[i] <= 0:
                N[i] = 1
        if e is not None and N["e"] <= 0:
            N["e"] = 1
    except KeyError as err: 
        raise KeyError("Missing entries in N dictionary: %s" % (err.message,))
    
    # Make sure these elements aren't present.
    for i in ["y","v"]:
        N.pop(i, None)
    
    # Sort out extra parameters.
    extraparshapes = __getShapes(extrapar)
    
    # Now get the shapes of all the variables that are present.
    deltaVars = ["u"] if uprev is not None else []
    allShapes = __generalVariableShapes(N, setpoint=list(sp), delta=deltaVars,
                                        extra=extraparshapes)
    if "c" not in N:
        N["c"] = 0
    
    # Sort out bounds on x0 and xf.
    for (d,v) in [(lb,-np.inf), (ub,np.inf), (guess,0)]:
        if "x" not in d:
            d["x"] = v*np.ones((N["t"]+1,N["x"]))
    if x0 is not None or any(["xf" in d for d in [lb, ub]]):
        # First, need to check if time-varying bounds were supplied. If not,
        # we need to make them time-varying for x so that we can change x0 and
        # xf separately. Other vars are handled in __optimalControlProblem.
        for d in [lb, ub, guess]:
            x = d["x"]
            if x.shape in [(N["x"],), (N["x"], 1), (1, N["x"])]:
                x = np.reshape(x, (1, N["x"]))
                d["x"] = np.tile(x, (N["t"]+1, 1))
        if x0 is not None:
            lb["x"][0,...] = x0
            ub["x"][0,...] = x0
            guess["x"][0,...] = x0
        for d in [lb, ub, guess]:
            if "xf" in d:
                xf = d.pop("xf")
                if xf.shape != (N["x"],):
                    raise ValueError("Incorrect size for xf.")
                d["x"][-1,...] = xf
    
    # Build Casadi symbolic structures. These need to be separate because one
    # is passed as a set of variables and one is a set of parameters. Note that
    # if this ends up empty, we just set it to None.
    parNames = set(["p"] + [k + "_sp" for k in sp]
        + [k + "_prev" for k in deltaVars] + list(extrapar))
    parStruct = __casadiSymStruct(allShapes, parNames, casaditype)
    if len(parStruct.keys()) == 0:
        parStruct = None
        
    varNames = set(["x", "u", "z", "xc", "zc", "s", "sf"]
                   + ["D" + k for k in deltaVars])
    varStruct = __casadiSymStruct(allShapes, varNames, casaditype)

    # Add parameters and setpoints to the guess structure.
    guess["p"] = p
    for v in sp:
        guess[v + "_sp"] = sp[v]
    if uprev is not None:
        # Need uprev to have shape (1, N["u"]).
        uprev = np.array(uprev).flatten()
        uprev.shape = (1,uprev.size)
        guess["u_prev"] = uprev
    for v in extrapar:
        thispar = np.array(extrapar[v])[np.newaxis,...]
        guess[v] = thispar
    
    # Need to decide about algebraic constraints.
    if "z" in allShapes:
        N["g"] = N["z"]
    else:
        N["g"] = None
    N["f"] = N["x"]
    
    # Decide if u has discrete components.
    discretevar = dict()
    if udiscrete is not None:
        discretevar["u"] = udiscrete
    
    # Make initial objective term.    
    if Pf is not None:
        if "Pf" not in funcargs:
            if inferargs:
                funcargs["Pf"] = __getargnames(Pf)
            elif "x" in sp:
                funcargs["Pf"] = ["x", "x_sp"]
            else:
                funcargs["Pf"] = ["x"]
        args = __getArgs(funcargs["Pf"], N["t"], varStruct, parStruct)
        obj = Pf(*args)
    else:
        obj = None
    
    # Terminal constraint (if present).
    if ef is not None:
        if "ef" not in funcargs:
            if inferargs:
                funcargs["ef"] = __getargnames(ef)    
            else:
                raise KeyError("Must provide an 'ef' entry in funcargs!")
        args = __getArgs(funcargs["ef"], N["t"], varStruct, parStruct)
        con = [ef(*args)]
        Nef = np.prod(con[0].shape) # Figure out number of entries.
        conlb = -np.inf*np.ones((Nef,))
        conub = np.zeros((Nef,))
    else:
        con = None
        conlb = None
        conub = None
    
    # Decide arguments of l.
    if "l" not in funcargs and not inferargs:
        largs = ["x","u"]
        if "x" in sp:
            largs.append("x_sp")
        if "u" in sp:
            largs.append("u_sp")
        funcargs["l"] = largs
    
    # Pick solver.
    if solver is None:
        if udiscrete is not None and np.any(udiscrete):
            solver = "bonmin"
        elif isQP:
            solver = "qpoases"
        if solver not in util.listAvailableSolvers(categorize=False):
            solver = None
        if solver is None:
            solver = "ipopt" # Default choice.
    
    # Build list of arguments for optimal control type.
    args = [N, varStruct, parStruct, lb, ub, guess, obj]
    kwargs = dict(f=f, g=g, h=None, l=l, e=e, funcargs=funcargs, Delta=Delta,
                  con=con, conlb=conlb, conub=conub,periodic=periodic,
                  verbosity=verbosity, timelimit=timelimit,
                  deltaVars=deltaVars, isQP=isQP,
                  casaditype=casaditype, discretel=discretel,
                  infercolloc=infercolloc, solver=solver,
                  discretevar=discretevar, inferargs=inferargs)
    return __optimalControlProblem(*args, **kwargs)

def nmhe(f, h, u, y, l, N, lx=None, x0bar=None, lb={}, ub={}, guess={}, g=None,
         p=None, verbosity=5, largs=None, funcargs={}, timelimit=60, Delta=None,
         wAdditive=False, casaditype="SX", inferargs=False, extrapar={}):
    """
    Solves nonlinear MHE problem.
    
    N muste be a dictionary with at least entries "x", "y", and "t". "w" may be
    specified, but it is assumed to be equal to "x" if not given. "v" is always
    taken to be equal to "y". If time-varying parameters are present, you must
    also specify a "p" entry. Time-invariant parameters can be specified in
    extrapar.
    
    u, y, and p must be 2D arrays with the time dimension first. Note that y
    and p should have N["t"] + 1 rows, while u should have N["t"] rows.
    
    lb and ub should be dictionaries of bounds for the various variables. Each
    entry should have time as the first index (i.e. lb["x"] should be a
    N["t"] + 1 by N["x"] array). guess should have the same structure.    
    
    Set wAddivitve=True to make the model

        x^+ = f(x,u,p) + w
        
    Otherwise, the model must take a "w" argument.
    
    The return value is a ControlSolver object.
    """
    # Copy dictionaries so we don't change the user inputs.
    N = N.copy()
    funcargs = funcargs.copy()
    
    # Also make sure some things are arrays of numpy dicts.
    lb = util.ArrayDict(lb)
    ub = util.ArrayDict(ub)
    guess = util.ArrayDict(guess)    
    
    # Check specified sizes.
    try:
        for i in ["x","y"]:
            if N[i] <= 0:
                N[i] = 1
            if N["t"] < 0:
                N["t"] = 0
        if "w" not in N:
            N["w"] = N["x"]
        N["v"] = N["y"]
    except KeyError:
        raise KeyError("Invalid or missing entries in N dictionary!")
    
    # Handle prior for x0bar.
    extrapar = extrapar.copy()    
    if lx is not None or x0bar is not None:
        if lx is None or x0bar is None:
            raise ValueError("Both or none of lx and x0bar must be given!")
        includeprior = True
        extrapar["x0bar"] = x0bar
        guess["x0bar"] = x0bar # Store parameters in guess struct.
    else:
        includeprior = False
    extraparshapes = __getShapes(extrapar)    
    
    
    # Now get the shapes of all the variables that are present.
    allShapes = __generalVariableShapes(N, finalx=True, finaly=True,
                                        extra=extraparshapes)
    if "c" not in N:
        N["c"] = 0    
    
    # Build Casadi symbolic structures. These need to be separate because one
    # is passed as a set of variables and one is a set of parameters.
    parNames = set(["u","p","y"] + list(extrapar))
    parStruct = __casadiSymStruct(allShapes, parNames, casaditype)
        
    varNames = set(["x","z","w","v","xc","zc"])
    varStruct = __casadiSymStruct(allShapes, varNames, casaditype)

    # Now we fill up the parameters in the guess structure.
    for (name,val) in [("u",u),("p",p),("y",y)]:
        guess[name] = val
    for v in extrapar:
        thispar = np.array(extrapar[v])[np.newaxis,...]
        guess[v] = thispar

    # Need to decide about algebraic constraints.
    if "z" in allShapes:
        N["g"] = N["z"]
    else:
        N["g"] = None
    N["h"] = N["y"]
    N["f"] = N["x"]
    
    # Make initial objective term.
    if "l" not in funcargs:
        funcargs["l"] = __getargnames(l) if inferargs else ["w", "v"]
    finallargs = []   
    for k in funcargs["l"]:
        if k == "w":
            finallargs.append(np.zeros(N["w"]))
        elif k in parStruct.keys():
            finallargs.append(parStruct[k,-1])
        elif k in varStruct.keys():
            finallargs.append(varStruct[k,-1])
        else:
            raise KeyError("l argument %s is invalid!" % k)
    obj = l(*finallargs)  
    if includeprior:
        lxargs = funcargs.get("lx", None)
        if lxargs is None and inferargs:
            lxargs = __getargnames(lx)
        if lxargs is not None:
            args = __getArgs(lxargs, 0, varStruct, parStruct)
        else:
            args = [varStruct["x",0] - parStruct["x0bar",0]]
        obj += lx(*args)
    
    # Decide if w is inside the model or additive.
    fErrorVars = []    
    if wAdditive:
        fErrorVars.append("w")
    args = [N, varStruct, parStruct, lb, ub, guess, obj]
    kwargs = dict(f=f, g=g, h=h, l=l, funcargs=funcargs, Delta=Delta,
                  verbosity=verbosity, casaditype=casaditype,
                  timelimit=timelimit, fErrorVars=fErrorVars,
                  inferargs=inferargs)
    return __optimalControlProblem(*args, **kwargs)


def sstarg(f, h, N, phi=None, lb={}, ub={}, guess={}, g=None, p=None,
           funcargs={}, extrapar={}, e=None, discretef=True, verbosity=5,
           timelimit=60, casaditype="SX", inferargs=False, udiscrete=None,
           ignoress=None):
    """
    Solves nonlinear steady-state target problem.
    
    N must be a dictionary with at least entries "x" and "y". If parameters
    are present, you must also specify a "p" entry.
    
    If given, ignoress is list of integers giving the indices of states to
    ignore in the steady-state constraint. The intended purpose is to screen
    out pure integrator states to avoid 0 = 0 constraints (which often cause
    problems during optimization).
    
    For descriptions of other arguments, refer to the help for nmpc.   
    """
    
    # Copy dictionaries so we don't change the user inputs.
    N = N.copy()
    funcargs = funcargs.copy()    
    
    # Make sure certain dictionaries have numpy arrays.
    lb = util.ArrayDict(lb)
    ub = util.ArrayDict(ub)
    guess = util.ArrayDict(guess)
    extrapar = util.ArrayDict(extrapar)       
    
    # Check specified sizes.
    try:
        for i in ["x","y"]:
            if N[i] <= 0:
                N[i] = 1
    except KeyError:
        raise KeyError("Invalid or missing entries in N dictionary!")
    
    # Sort out extra parameters.
    extraparshapes = __getShapes(extrapar)    
    
    # Now get the shapes of all the variables that are present.
    N["t"] = 1
    allShapes = __generalVariableShapes(N,finalx=False,finaly=False,
                                        extra=extraparshapes)
    if "c" not in N:
        N["c"] = 0
    
    # Build Casadi symbolic structures. These need to be separate because one
    # is passed as a set of variables and one is a set of parameters.
    parNames = set(["p"] + list(extrapar))
    parStruct = __casadiSymStruct(allShapes, parNames, casaditype)
    
    varNames = set(["x", "z", "u", "y", "s"])
    varStruct = __casadiSymStruct(allShapes, varNames, casaditype)

    # Now we fill up the parameters in the guess structure.
    guess["p"] = p
    for v in extrapar:
        thispar = np.array(extrapar[v])
        thispar.shape = (1,) + thispar.shape # Prepend dummy time dimension.
        guess[v] = thispar
    
    # Need to decide about algebraic constraints.
    if "z" in allShapes:
        N["g"] = N["z"]
    else:
        N["g"] = None
    N["h"] = N["y"]
    if "f" not in N:
        N["f"] = N["x"]
    
    # Screen out ignored states.
    if ignoress is not None:
        if inferargs and "f" not in funcargs:
            funcargs["f"] = __getargnames(f)
        if "f" in funcargs:
            xind = dict(zip(funcargs["f"], range(len(funcargs["f"])))).get("x")
        else:
            xind = 0
        keep = np.ones(N["x"], dtype=bool)
        for i in ignoress:
            keep[i] = False
        mask = np.eye(N["x"])[keep,:] 
        _f = f
        _discretef = discretef
        def wrappedf(*args):
            """Returns only a subset of of model equations."""
            dx = _f(*args)
            if _discretef:
                dx -= args[xind]
            return util.mtimes(mask, dx)
        f = getCasadiFunc(wrappedf, wraps=_f)
        discretef = False
    
    # Make objective term.
    if phi is not None:
        if "phi" not in funcargs:
            if inferargs:
                funcargs["phi"] = __getargnames(phi)
            else:
                raise KeyError("Must provide funcargs['phi'] or set inferargs"
                               "=Trueif phi is given!")
        args = __getArgs(funcargs["phi"], 0, varStruct, parStruct)
        obj = phi(*args)
    else:
        obj = None
    
    # Decide if u has discrete components.
    discretevar = dict()
    if udiscrete is not None:
        discretevar["u"] = udiscrete
    
    # Get controller arguments.    
    args = [N, varStruct, parStruct, lb, ub, guess, obj]
    kwargs = dict(f=f, g=g, h=h, funcargs=funcargs, verbosity=verbosity,
                  discretef=discretef, finalpoint=False, casaditype=casaditype,
                  timelimit=timelimit, inferargs=inferargs, e=e,
                  discretevar=discretevar)
    return __optimalControlProblem(*args, **kwargs)


def __optimalControlProblem(N, var, par=None, lb={}, ub={}, guess={},
        obj=None, f=None, g=None, h=None, l=None, e=None, funcargs={},
        Delta=None, con=None, conlb=None, conub=None, periodic=False,
        discretef=True, deltaVars=None, finalpoint=True, verbosity=5,
        timelimit=60, casaditype="SX", discretel=True, fErrorVars=None,
        isQP=False, infercolloc=None, solver="ipopt", discretevar=None,
        inferargs=False):
    """
    General wrapper for an optimal control problem (e.g., mpc or mhe).
    
    var and par must both be casadi sym_structs.
    
    Note that only variable fields are taken from lb and ub, but parameter
    values must be specified in the guess dictionary.
    """

    # Initialize things.
    if discretevar is None:
        discretevar = {}
    varlb = var(-np.inf)
    varub = var(np.inf)
    varguess = var(0)
    vardiscretevar = var(False)
    dataAndStructure = [(guess,varguess,"guess"), (lb,varlb,"lb"),
                        (ub,varub,"ub"),
                        (discretevar,vardiscretevar,"discretevar")]
    if par is not None:
        parval = par(0) # guess dictionary also contains parameter values.
        dataAndStructure.append((guess,parval,"par"))  # See above.
    else:
        parval = None
    misc = {"N" : N.copy()}
    
    # Check timestep.
    if N.get("c", 0) > 0:
        if Delta is None:
            raise ValueError("Must provide Delta to use collocation!")
        misc["Delta"] = Delta
        
    # Sort out bounds and parameters.
    for (data,structure,name) in dataAndStructure:
        for v in set(data).intersection(structure.keys()): # .keys() is important!
            # Check sizes. We have to decide if the user supplied time-varying
            # bounds or not (as determined by the shape of data).
            vs = structure[v]
            d = data[v]
            tryshapes = set()
            if len(vs) > 0:
                s = vs[0].shape
                tryshapes.add(s)
                for n in [1,2]:
                    if s[-n:] == (1,)*n:
                        tryshapes.add(s[:-n])
            if d.shape in tryshapes:
                # d does not have a time dimension, so add one.
                d = d[np.newaxis,...]
                o = 0 # Offset multiplier.
            else:
                # d does have a time dimension, so check.           
                if len(vs) < d.shape[0]:
                    warnings.warn("Extra time points in %s['%s']. "
                        "Ignoring." % (name,v))
                elif len(vs) > d.shape[0]:
                    raise IndexError("Too few time points in %s['%s']!" %
                                     (name, v))
                o = 1 # Offset multiplier.
                
            # Grab data.            
            for t in range(len(structure[v])):
                structure[v,t] = d[o*t,...]   
    
    # Make sure slacks are nonnegative.
    for v in set(["s", "sf"]).intersection(var.keys()):
        varlb[v] = [np.zeros(bound.shape) for bound in varlb[v]]
        varub[v] = [np.inf*np.ones(bound.shape) for bound in varub[v]]
    
    # Smush together variables and parameters to get the constraints.
    struct = {}
    for k in var.keys():
        struct[k] = var[k]
    if par is not None:
        for k in par.keys():
            struct[k] = par[k]
    
    # Double-check some sizes and then get constraints.
    for (func,name) in [(f,"f"), (g,"g"), (h,"h"), (e,"e")]:
        if func is None:
            N[name] = 0
    if fErrorVars is None:
        fErrorVars = []
    if deltaVars is None:
        deltaVars = []
    constraints = __generalConstraints(struct, N["t"], f=f, Nf=N["f"],
        g=g, Ng=N["g"], h=h, Nh=N["h"], l=l, funcargs=funcargs, Ncolloc=N["c"],
        Delta=Delta, discretef=discretef, deltaVars=deltaVars,
        finalpoint=finalpoint, e=e, Ne=N["e"], discretel=discretel,
        fErrorVars=fErrorVars, inferargs=inferargs)
        
    # Save collocation weights and generate a guess for xc if not given.
    if "colloc" in constraints:
        misc["colloc"] = constraints["colloc"]
        if infercolloc is None:
            infercolloc = ("xc" not in guess and "x" in guess)
        if infercolloc:
            util._infercolloc(misc["colloc"]["r"], varguess)
    
    # Build up constraints.
    if con is None or conlb is None or conub is None:
        con = []
        conlb = np.array([])
        conub = np.array([])
    if periodic and "x" in list(struct.keys()):
        con.append(struct["x"][0] - struct["x"][-1])
        conlb = np.concatenate([conlb,np.zeros((N["x"],))])
        conub = np.concatenate([conub,np.zeros((N["x"],))])
    for f in ["state","measurement","algebra","delta","path"]:
        if f in list(constraints.keys()):
            con += util.flattenlist(constraints[f]["con"])
            conlb = np.concatenate([conlb,constraints[f]["lb"].flatten()])
            conub = np.concatenate([conub,constraints[f]["ub"].flatten()])
    con = casadi.vertcat(*con)
    
    if obj is None:
        try:
            obj = dict(SX=casadi.SX, MX=casadi.MX)[casaditype](0)
        except KeyError:
            raise ValueError("Unknown casaditype. Must be 'SX' or 'MX'.")
    
    if "cost" in list(constraints.keys()):
        obj = sum(util.flattenlist(constraints["cost"]),obj)
    
    # Build ControlSolver object and return that.
    args = [var, varlb, varub, varguess, obj, con, conlb, conub, par, parval]
    kwargs = dict(verbosity=verbosity, timelimit=timelimit, isQP=isQP,
                  casaditype=casaditype, misc=misc, discretevar=vardiscretevar,
                  solver=solver)
    solver = solvers.ControlSolver(*args, **kwargs)
    return solver

def __generalConstraints(var, Nt, f=None, Nf=0, g=None, Ng=0, h=None, Nh=0,
                         l=None, funcargs=None, Ncolloc=0, Delta=1,
                         discretef=True, deltaVars=None, finalpoint=True,
                         e=None, Ne=0, discretel=True, fErrorVars=None,
                         inferargs=False):
    """
    Creates general state evolution constraints for the following system:
    
       x^+ = f(x,z,u,w,p)                      \n
       g(x,z,w,p) = 0                          \n
       y = h(x,z,p) + v                        \n
       e(x,z,u,p) <= 0                         \n
       
    The variables are intended as follows:
    
        x: differential states                  \n
        z: algebraic states                     \n
        u: control actions                      \n
        w: unmodeled state disturbances         \n
        p: fixed system parameters              \n
        y: meadured outputs                     \n
        v: noise on outputs
    
    The arguments of any functions can be overridden by specifying a list of
    arguments in the appropriate entry of the dictionary funcargs. E.g., if
    your function f is f(p,y,z) then you would pass {"f" : ["p","y","z"]} for
    funcargs.    
    
    Also builds up a list of stage costs l(...). Note that if l is given, its
    arguments must be specified in funcargs as a list of variable names.
    
    In principle, you can use the variables for whatever you want, but they
    must show up in the proper order. We do very little checking of this, so
    if this function errors, make sure you are passing the proper variables.
    
    var should be a dictionary with entries "x", "u", "y", etc., that give
    either casadi variables or data values. Data must be accessed as follows:

        var["x"][t][k,...] gives the kth state of x at time t
        
    In particular, if struct is a casadi.tools.struct_symMX object, then
    setting var["x"] = struct["x"] will suffice. If you want to use your own
    data structure, so be it.

    If Ncolloc is not None, var must also have entries "xc" and "zc". Each
    entry must have Ncolloc as the size of the second dimension.
    
    deltaVars should be a list of variables to make constraints for time
    differences. For each entry in this list, var must have the appropriate
    keys, e.g. if deltaVars = ["u"], then var must have "u", "Du", and "u_prev"
    entries or else this will error.    
    
    Returns a dictionary with entries "state", "algebra", and "measurement".
    Note that the relevant fields will be missing if f, g, or h are set to
    None. Each entry in the return dictionary will be a list of lists, with
    each sublist corresponding to a single time segment worth of constraints.
    The list of stage costs is in "cost". This is also a list of lists, but
    each sub-list only has one element unless you are using a continuous
    objective function.
    """
    
    # Figure out what variables are supplied.
    if fErrorVars is None:
        fErrorVars = []
    if funcargs is None:
        funcargs = {}
    if deltaVars is None:
        deltaVars = []
    givenvars = set(var.keys())
    givenvarscolloc = givenvars.intersection(["x","z"])    
    
    # Decide function arguments.
    if inferargs:
        funcs = dict(f=f, g=g, h=h, l=l, e=e)
        args = {k : __getargnames(v) for (k, v) in funcs.items()}
        args.update(funcargs)
    else:
        args = funcargs.copy()
        if l is not None and "l" not in args:
            raise KeyError("Must supply arguments to l!")
    
    # Make sure user-defined arguments are valid.
    for k in args:
        try:
            okay = givenvars.issuperset(args[k])
        except TypeError:
            raise TypeError("funcargs['%s'] must be a list of strings!" % (k,))
        if not okay:
            badvars = set(args[k]).difference(givenvars)
            raise ValueError("Bad arguments for %s: %s."
                             % (k, repr(list(badvars))))
    
    # Now sort out defaults.
    defaultargs = {
        "f" : [k for k in ["x","z","u","w","p"] if k not in fErrorVars],
        "g" : ["x","z","w","p"],
        "h" : ["x","z","p"],
        "e" : ["x","z","u","p"],
    }
    def isGiven(v): # Membership function.
        return v in givenvars
    for k in set(defaultargs).difference(args):
        args[k] = [a for a in defaultargs[k] if isGiven(a)]
    
    # Also define inverse map to get positions of arguments.
    argsInv = {}
    for a in args:
        argsInv[a] = dict([(args[a][j], j) for j in range(len(args[a]))])
    
    # Define some helper functions/variables.    
    def getArgs(func,times,var):
        """Returns a list of casadi variables with index time."""
        allargs = []
        for t in times:
            allargs.append(__getArgs(args[func],t,var))
        return allargs
    tintervals = np.arange(Nt)
    tpoints = np.arange(Nt + bool(finalpoint))
    
    # Preallocate return dictionary.
    returnDict = {}    
    
    # Decide whether we got the correct stuff for collocation.
    if Ncolloc < 0 or round(Ncolloc) != Ncolloc:
        raise ValueError("Ncolloc must be a nonnegative integer if given.")
    if Ncolloc > 0:
        [r,A,B,q] = colloc.weights(Ncolloc, True, True) # Collocation weights.
        returnDict["colloc"] = {"r" : r, "A" : A, "B" : B, "q" : q}
        collocvar = {}
        for v in givenvarscolloc:
            # Make sure we were given the corresponding "c" variables.            
            if v + "c" not in givenvars:
                raise KeyError("Entry %sc not found in vars!" % (v,))
            collocvar[v] = []
            for k in range(Nt):
                errorVar = __getArgs(fErrorVars,k,var)
                collocvar[v].append([var[v][k]]
                    + [var[v+"c"][k][:,j] for j in range(Ncolloc)]
                    + [sum(errorVar,var[v][k+1 % len(var[v])])])
    
        def getCollocArgs(k,t,j):
            """
            Gets arguments for function k at time t and collocation point j.
            """
            thisargs = []
            for a in args[k]:
                if a in givenvarscolloc:
                    thisargs.append(collocvar[a][t][j])
                elif len(var[a]) == 1:
                    thisargs.append(var[a][0])     
                else:
                    thisargs.append(var[a][t])
            return thisargs
    
    # State evolution f.   
    if f is not None:
        if Nf <= 0:
            raise ValueError("Nf must be a positive integer!")
        fargs = getArgs("f",tintervals,var)
        state = []
        for t in tintervals:
            errorargs = __getArgs(fErrorVars,t,var)
            if Ncolloc == 0:
                # Just use discrete-time equations.
                thiscon = f(*fargs[t])
                if "x" in givenvars and discretef:
                    thiscon -= var["x"][t+1 % len(var["x"])]
                thiscon = sum(errorargs, thiscon)
                thesecons = [thiscon] # Only one constraint per timestep.
            else:
                # Need to do collocation stuff.
                thesecons = []
                for j in range(1,Ncolloc+2):
                    thisargs = getCollocArgs("f",t,j)
                    # Start with function evaluation.
                    thiscon = Delta*f(*thisargs)
                    
                    # Add collocation weights.
                    if "x" in givenvarscolloc:
                        for jprime in range(len(collocvar["x"][t])):
                            thiscon -= A[j,jprime]*collocvar["x"][t][jprime]
                    thesecons.append(thiscon)
            state.append(thesecons)
        lb = np.zeros((len(tintervals),Ncolloc+1,Nf))
        ub = lb.copy()
        returnDict["state"] = dict(con=state,lb=lb,ub=ub)
            
    # Algebraic constraints g.
    if g is not None:
        if Ng <= 0:
            raise ValueError("Ng must be a positive integer!")
        gargs = getArgs("g",tpoints,var)
        algebra = []
        for t in tpoints:
            if Ncolloc == 0 or t == Nt:
                thesecons = [g(*gargs[t])]
            else:
                thesecons = []
                for j in range(Ncolloc+1):
                    thisargs = getCollocArgs("g",t,j)
                    thiscon = g(*thisargs)
                    thesecons.append(thiscon)
            algebra.append(thesecons)
        lb = np.zeros(((len(tpoints)-1)*(Ncolloc+1)+1,Ng))
        ub = lb.copy()
        returnDict["algebra"] = dict(con=algebra,lb=lb,ub=ub)
        
    # Measurements h.
    if h is not None:
        if Nh <= 0:
            raise ValueError("Nh must be a positive integer!")
        hargs = getArgs("h",tpoints,var)
        measurement = []
        for t in tpoints:
            thiscon = h(*hargs[t])
            if "y" in givenvars:
                thiscon -= var["y"][t]
            if "v" in givenvars:
                thiscon += var["v"][t]
            measurement.append([thiscon])
        lb = np.zeros((len(measurement),Nh))
        ub = lb.copy()
        returnDict["measurement"] = dict(con=measurement,lb=lb,ub=ub)
    
    # Delta variable constraints.
    if len(deltaVars) > 0:
        deltaconstraints = []
        numentries = 0
        for v in deltaVars:
            if not set([v,"D"+v, v+"_prev"]).issubset(var.keys()):
                raise KeyError("Variable '%s' must also have entries 'D%s' "
                    "and '%s_prev'!" % (v,v,v))
            thisdelta = [var["D" + v][0] - var[v][0] + var[v + "_prev"][0]]
            for t in range(1,len(var[v])):
                thisdelta.append(var["D" + v][t] - var[v][t] + var[v][t-1])
            deltaconstraints.append(thisdelta)
            numentries += len(var[v])*np.product(var[v][0].shape)
        lb = np.zeros((numentries,))
        ub = lb.copy()
        returnDict["delta"] = dict(con=deltaconstraints,lb=lb,ub=ub)
          
    # Stage costs. Either discrete sum or quadrature via collocation.
    if l is not None:
        cost = []
        if discretel:
            largs = getArgs("l",tintervals,var)
            for t in tintervals:
                cost.append([l(*largs[t])])
        else:
            if Ncolloc == 0:
                raise ValueError("Must use collocation for continuous "
                    "objective!")
            for t in tintervals:
                thiscost = []
                for j in range(Ncolloc+2):
                    thisargs = getCollocArgs("l",t,j)
                    thiscost.append(Delta*q[j]*l(*thisargs))
                cost.append(thiscost)
        returnDict["cost"] = cost
    
    # Nonlinear path constraints.
    if e is not None:
        if Ne <= 0:
            raise ValueError("Ne must be a positive integer!")
        eargs = getArgs("e",tintervals,var)
        pathconstraints = []
        for t in tintervals:
            # Need to wrap e() in a list because only one call per timestep.
            pathconstraints.append([e(*eargs[t])])
        lb = -np.inf*np.ones((len(pathconstraints),Ne))
        ub = np.zeros((len(pathconstraints),Ne))
        returnDict["path"] = dict(con=pathconstraints,lb=lb,ub=ub)
    return returnDict


# =====================================
# Building CasADi Functions and Objects
# =====================================

def __generalVariableShapes(sizeDict, setpoint=[], delta=[], finalx=True,
                            finaly=False, extra={}):
    """
    Generates variable shapes from the size dictionary N.
    
    The keys of N must be a subset of
        ["x","z","u","d","w","p","y","v","c"]    
    
    If present, "c" will specify collocation, at which point extra variables
    will be created.
    
    Each entry in the returned dictionary will be a dictionary of keyword
    arguments repeat and shape to pass to __casadiSymStruct.
    
    Optional argument setpiont is a list of variables that have corresponding
    setpoint parameters. Any variable names in this list will have a
    corresponding entry suffixed with _sp. This is useful, e.g., for control
    problems where you may change the system setpoint.
    
    Optional argument delta is similar, but it defines decision variables to
    calculate the difference between successive time points. This is useful for
    rate-of-change penalties or constraints for control problems.
    
    If N["t"] is 0, then each variable will only have one entry. This is useful
    for steady-state target problems where you only want one x and z variable.
    """
    # Figure out what variables are supplied.
    allsizes = set(["x", "z", "u", "w", "p", "y", "v", "s", "sf", "c", "t"])
    givensizes = allsizes.intersection(sizeDict)
    
    # Make sure we were given a time.
    try:
        Nt = sizeDict["t"]
    except KeyError:
        raise KeyError("Entry 't' must be provided!")
    
    # Need to decide whether to include final point of various entries.
    finalx = bool(finalx)
    finaly = bool(finaly)
    
    # Now we're going to build a data structure that says how big each of the
    # variables should be. The first entry 1 if there should be N+1 copies of
    # the variable and 0 if there should be N. The second is a list of sizes.
    allvars = {
        "x" : (Nt+finalx, ["x"]),
        "z" : (Nt+finalx, ["z"]),
        "u" : (Nt, ["u"]),
        "w" : (Nt, ["w"]),
        "p" : (Nt + finaly, ["p"]),
        "y" : (Nt + finaly, ["y"]),
        "v" : (Nt + finaly, ["v"]),
        "s" : (Nt, ["s"]),
        "sf" : (1, ["sf"]),
        "xc": (Nt, ["x", "c"]),
        "zc": (Nt, ["z", "c"]),        
    }
    # Here we define any "extra" variables like setpoints. The syntax is a
    # tuple with the prefix string, the list of variables, the suffix string,
    # and the number of entries (None to use the default value).
    extraEntries = [
        ("", setpoint, "_sp", None), # These are parameters for sepoints.
        ("D", delta, "", None), # These are variables to calculate deltas.
        ("", delta, "_prev", 1), # This is a parameter for the previous value.    
    ]
    for (prefix, var, suffix, num) in extraEntries:
        for v in set(var).intersection(allvars.keys()):
            thisnum = num if num is not None else allvars[v][0]
            allvars[prefix + v + suffix] = (thisnum, allvars[v][1])
        
    # Now loop through all the variables, and if we've been given all of the
    # necessary sizes, then add that variable
    shapeDict = {}
    for (v, (t, shapeinds)) in allvars.items():
        if givensizes.issuperset(shapeinds):
            shape = [sizeDict[i] for i in shapeinds]
            if len(shape) == 0:
                shape = [1,1]
            elif len(shape) == 1:
                shape += [1]
            shapeDict[v] = {"repeat" : t, "shape" : tuple(shape)}
            
    # Finally, look through extra variables and raise an error if something
    # will overwrite a variable that is already there.
    for v in extra:
        if v in shapeDict:
            raise KeyError("Extra parameter '%s' shadows a reserved name. "
                "Please choose a different name." % (v,))
        else:
            shapeDict[v] = {"repeat" : 1, "shape" : tuple(extra[v])}
    
    return shapeDict


def __casadiSymStruct(allVars, theseVars=None, casaditype="SX"):
    """
    Returns a Casadi sym struct for the variables in allVars.
    
    To use only a subset of variables, set theseVars to the subset of variables
    that you need. If theseVars is not None, then only variable names appearing
    in allVars and theseVars will be created.
    """
    # Figure out what names we need.
    allVars = allVars.copy()
    varNames = set(allVars.keys())
    if theseVars is not None:
        for v in varNames.difference(theseVars):
            allVars.pop(v)
    
    # Choose type of struct.
    if casaditype == "SX":
        struct = ctools.struct_symSX
    elif casaditype == "MX":
        struct = ctools.struct_symMX
    else:
        raise ValueError("Invalid choice of casaditype. Must be 'SX' or 'MX'.")
    
    # Build casadi struct_symSX    
    structArgs = tuple([ctools.entry(name,**args) for (name,args)
        in allVars.items()])
    return struct([structArgs])

def __getShapes(vals, mindims=1, extra="prepend"):
    """
    Gets shapes for each entry of the dictionary vals.
    
    Each entry of vals must either have a shape attribute or be castable to a
    numpy array so that its shape can be determined. Any entry with fewer than
    ndims dimensions will have ones added depending on the value of extra as
    follows:
    
        "prepend" : add dimensions to the front
        "append" : add dimensions to the back
        "squeeze" : squeeze out all singletons
    """
    extras = {
        "prepend" : lambda s, s1 : s1 + s,
        "append" : lambda s, s1 : s + s1,
        "squeeze" : lambda s, s1 : __squeezeShape(s),
    }
    try:
        fixfunc = extras[extra]
    except KeyError:
        raise ValueError("Invalid choices for extra!")
    shapes = {}
    for (k, v) in vals.items():
        s = getattr(v, "shape", None)
        if s is None:
            try:
                s = np.array(v, dtype=float).shape
            except ValueError:
                raise ValueError("Entry '%s' does not have a shape andcannot "
                    "be converted to a numpy array!" % (k,))
        shapes[k] = fixfunc(s, (1,)*(mindims - len(s)))
    return shapes


def __squeezeShape(s, endonly=False):
    """
    Takes a shape tuple and squeezes out singleton dimensions.
    
    If endonly=True, only trailing dimensions are removed.
    """
    if endonly:
        while len(s) > 0 and s[-1] == 1:
            s = s[:-1]
    else:
        s = tuple([i for i in s if i != 1])
    return s


def __getArgs(names,t=0,*structs):
    """
    Returns the arguments in names at time t by searching through all structs.
    
    Raises a KeyError if any argument is not found.
    """
    thisargs = []
    for v in names:
        for (i, struct) in enumerate(structs):
            if struct is not None and v in struct.keys():
                found = True
                break
        else:
            found = False
        if not found:
            allkeys = []
            for struct in structs:
                if struct is not None:
                    allkeys.append(struct.keys())
            raise ValueError("Argument %s is invalid! Must be in [%s]!" 
                             % (v, ", ".join(util.flattenlist(allkeys))))
        if len(structs[i][v]) == 1:
            thisargs.append(structs[i][v][0])
        else:
            thisargs.append(structs[i][v][t])
    return thisargs

def getCasadiFunc(f, varsizes=None, varnames=None, funcname=None, rk4=False,
                  Delta=1, M=1, scalar=None, casaditype=None, wraps=None,
                  numpy=None):
    """
    Takes a function handle and turns it into a Casadi function.
    
    f should be defined to take a specified number of arguments and return a
    scalar, list, or numpy array. varnames, if specified, gives names to each
    of the inputs, but this is not required.
    
    sizes should be a list of how many elements are in each one of the inputs.
    
    Alternatively, instead of specifying varsizes, varnames, and funcname,
    you can pass a casadi.Function as wraps to copy these values from the other
    function.
    
    The numpy argument determines whether arguments are passed with numpy
    array semantics or not. By default, numpy=True, which means symbolic
    variables are passed as numpy arrays of Casadi scalar symbolics. This means
    your function should be written to accept (and should also return) numpy
    arrays. If numpy=False, the arguments are passed as Casadi symbolic
    vectors, which have slightly different semantics. Note that 'scalar'
    is a deprecated synonym for numpy.
    
    To choose what type of Casadi symbolic variables to use, pass
    casaditype="SX" or casaditype="MX". The default value is "SX" if
    numpy=True, and "MX" if numpy=True.
    """ 
    # Decide if user specified wraps.
    if wraps is not None:
        if not isinstance(wraps, casadi.Function):
            raise TypeError("wraps must be a casadi.Function!")
        if varsizes is None:
            varsizes = [wraps.size_in(i) for i in range(wraps.n_in())]
        if varnames is None:
            varnames = [wraps.name_in(i) for i in range(wraps.n_in())]
        if funcname is None:
            funcname = wraps.name()
    
    # Pass the buck to the sub function.
    if varsizes is None:
        raise ValueError("Must specify either varsizes or wraps!")
    if funcname is None:
        funcname = "f"
    if numpy is None and scalar is not None:
        numpy = scalar
        warnings.warn("Passing 'scalar' is deprecated. Replace with 'numpy'.")
    symbols = __getCasadiFunc(f, varsizes, varnames, funcname,
                              numpy=numpy, casaditype=casaditype,
                              allowmatrix=True)
    args = symbols["casadiargs"]
    fexpr = symbols["fexpr"]
    
    # Evaluate function and make a Casadi object.  
    fcasadi = casadi.Function(funcname, args, [fexpr], symbols["names"],
                              [funcname])
    
    # Wrap with rk4 if requested.
    if rk4:
        frk4 = util.rk4(fcasadi, args[0], args[1:], Delta, M)
        fcasadi = casadi.Function(funcname, args, [frk4], symbols["names"],
                                  [funcname])
    
    return fcasadi


def getCasadiIntegrator(f, Delta, argsizes, argnames=None, funcname="int_f",
                        abstol=1e-8, reltol=1e-8, wrap=True, verbosity=1,
                        scalar=None, casaditype=None, numpy=None):
    """
    Gets a Casadi integrator for function f from 0 to Delta.
    
    Argsizes should be a list with the number of elements for each input. Note
    that the first argument is assumed to be the differential variables, and
    all others are kept constant.
    
    The scalar, casaditype, and numpy arguments all have the same behavior as
    in getCasadiFunc. See getCasadiFunc documentation for more details.
    
    wrap can be set to False to return the raw casadi Integrator object, i.e.,
    with inputs x and p instead of the arguments specified by the user.
    """
    # First get symbolic expressions.
    if numpy is None and scalar is not None:
        numpy = scalar
        warnings.warn("Keyword argument 'scalar=...' is deprecated. Replace "
                      "with 'numpy=...'.")
    symbols = __getCasadiFunc(f, argsizes, argnames, funcname,
                              numpy=numpy, casaditype=casaditype,
                              allowmatrix=False)
    x0 = symbols["casadiargs"][0]
    par = symbols["casadiargs"][1:]
    fexpr = symbols["fexpr"]
    
    # Build ODE and integrator.
    ode = dict(x=x0, p=casadi.vertcat(*par), ode=fexpr)
    options = {
        "abstol" : abstol,
        "reltol" : reltol,
        "tf" : Delta,
        "disable_internal_warnings" : verbosity <= 0,
        "verbose" : verbosity >= 2,
    }
    integrator = casadi.integrator(funcname, "cvodes", ode, options)
    
    # Now do the subtle bit. Integrator has arguments x0 and p, but we need
    # arguments as given by the user. First we need MX arguments.
    if wrap:
        names = symbols["names"]
        sizes = symbols["sizes"]
        wrappedx0 = casadi.MX.sym(names[0], *sizes[0])
        wrappedpar = [casadi.MX.sym(names[i], *sizes[i]) for i
                      in range(1, len(sizes))]    
        wrappedIntegrator = integrator(x0=wrappedx0,
                                       p=casadi.vertcat(*wrappedpar))["xf"]
        integrator = casadi.Function(funcname, [wrappedx0] + wrappedpar,
                                     [wrappedIntegrator], symbols["names"],
                                     [funcname])
    return integrator


def __getCasadiFunc(f, varsizes, varnames=None, funcname="f", numpy=None,
                    casaditype=None, allowmatrix=True):
    """
    Core logic for getCasadiFunc and its relatives.
    
    casaditype chooses what type of casadi variable to use, while numpy chooses
    to wrap the casadi symbols in a NumPy array before calling f. Both
    numpy and casaditype are None by default; the table below shows what values
    are used in the various cases.
    
                  +----------------------+-----------------------+
                  |       numpy is       |       numpy is        |
                  |         None         |       not None        |
    +-------------+----------------------+-----------------------+
    | casaditype  | casaditype = "SX"    | casaditype = ("SX" if |
    |  is None    | numpy = True         |   numpy else "MX")    |
    +------------------------------------+-----------------------+
    | casaditype  | numpy = (False if    | warning issued if     |
    | is not None |   casaditype == "MX" |   numpy == True and   |
    |             |   else True)         |   casaditype == "MX"  |
    +------------------------------------+-----------------------+
    
    Returns a dictionary with the following entries:
        
    - casadiargs: a list of the original casadi symbolic primitives
    
    - numpyargs: a list of the numpy analogs of the casadi symbols. Note that
                 this is None if numpy=False.
    
    - fargs: the list of arguments passed to f. This is numpyargs if numpyargs
             is not None; otherwise, it is casadiargs.
     
    - fexpr: the casadi expression resulting from evaluating f(*fargs).
    
    - XX: is either casadi.SX or casadi.MX depending on what type was used
          to create casadiargs.
  
    - names: a list of string names for each argument.
    
    - sizes: a list of one- or two-element lists giving the sizes.
    """
    # Check names.
    if varnames is None:
        varnames = ["x%d" % (i,) for i in range(len(varsizes))]
    else:
        varnames = [str(n) for n in varnames]
    if len(varsizes) != len(varnames):
        raise ValueError("varnames must be the same length as varsizes!")
    
    # Loop through varsizes in case some may be matrices.
    realvarsizes = []
    for s in varsizes:
        goodInput = True
        try:
            s = [int(s)]
        except TypeError:
            if allowmatrix:
                try:
                    s = list(s)
                    goodInput = len(s) <= 2
                except TypeError:
                    goodInput = False
            else:
                raise TypeError("Entries of varsizes must be integers!")
        if not goodInput:
            raise TypeError("Entries of varsizes must be integers or "
                "two-element lists!")
        realvarsizes.append(s)
    
    # Decide which Casadi type to use and whether to wrap as a numpy array.
    # XX is either casadi.SX or casadi.MX.
    if numpy is None and casaditype is None:
        numpy = True
        casaditype = "SX"
    elif numpy is None:
        numpy = False if casaditype == "MX" else True
    elif casaditype is None:
        casaditype = "SX" if numpy else "MX"
    else:
        if numpy and (casaditype == "MX") and WARN_NUMPY_MX:
            warnings.warn("Using a numpy array of casadi MX is almost always "
                          "a bad idea. Consider refactoring to avoid.")
    XX = dict(SX=casadi.SX, MX=casadi.MX).get(casaditype, None)
    if XX is None:
        raise ValueError("casaditype must be either 'SX' or 'MX'!")
        
    # Now make the symbolic variables. Make numpy versions if requested.
    casadiargs = [XX.sym(name, *size)
                  for (name, size) in zip(varnames, realvarsizes)]
    if numpy:
        numpyargs = [__casadi_to_numpy(x) for x in casadiargs]
        fargs = numpyargs
    else:
        numpyargs = None
        fargs = casadiargs
    
    # Evaluate the function and return everything.
    fexpr = util.safevertcat(f(*fargs))
    return dict(fexpr=fexpr, casadiargs=casadiargs, numpyargs=numpyargs, XX=XX,
                names=varnames, sizes=realvarsizes)


def __casadi_to_numpy(x, matrix=False, scalar=False):
    """
    Converts casadi symbolic variable x to a numpy array of scalars.
    
    If matrix=False, the function will guess whether x is a vector and return
    the appropriate numpy type. To force a matrix, set matrix=True. To use
    a numpy scalar when x is scalar, use scalar=True.
    """
    shape = None
    if not matrix:
        if scalar and x.is_scalar():
            shape = ()
        elif x.is_vector():
            shape = (x.numel(),)
    if shape is None:
        shape = x.shape
    y = np.empty(shape, dtype=object)
    if y.ndim == 0:
        y[()] = x # Casadi uses different behavior for x[()].
    else:
        for i in np.ndindex(shape):
            y[i] = x[i]
    return y


def __getargnames(func):
    """
    Returns a list of named input arguments of a Casadi Function.
    
    For convenience, if func is None, an empty list is returned.
    """
    argnames = []
    if func is not None:
        try:
            for i in range(func.n_in()):
                argnames.append(func.name_in(i))
        except AttributeError:
            if not isinstance(func, casadi.Function):
                raise TypeError("func must be a casadi.Function!")
            else:
                raise
    return argnames


# =============
# Miscellany
# ============

class DummySimulator(object):
    """
    Wrapper class to simulate a generic discrete-time function.
    """
    @property
    def Nargs(self):
        return self.__Nargs
        
    @property
    def args(self):
        return self.__argnames
    
    def __init__(self, model, argsizes, argnames=None):
        """Initilize the simulator using a model function."""
        # Decide argument names.
        if argnames is None:
            argnames = ["x"] + ["p_%d" % (i,) for i in range(1,self.Nargs)]
        
        # Store names and model.
        self.__argnames = argnames
        self.__integrator = model
        self.__argsizes = argsizes
        self.__Nargs = len(argsizes)
    
    def call(self, *args):
        """
        Simulates one timestep and returns a vector.
        """
        self._checkargs(args)
        return self.__integrator(*args)

    def _checkargs(self, args):
        """Checks that the right number of arguments have been given."""
        if len(args) != self.Nargs:
            raise ValueError("Wrong number of arguments: %d given; %d "
                             "expected." % (len(args), self.Nargs))

    def __call__(self, *args):
        """
        Interface to self.call.
        """
        return self.call(*args)
    
    def sim(self, *args):
        """
        Simulate one timestep and returns a Numpy array.
        """
        return np.array(self.call(*args)).flatten()
    

class DiscreteSimulator(DummySimulator):
    """
    Simulates one timestep of a continuous-time system.
    """
    
    @property
    def Delta(self):
        return self.__Delta
        
    def __init__(self, ode, Delta, argsizes, argnames=None, verbosity=1,
                 casaditype=None, numpy=None, scalar=None):
        """
        Initialize by specifying model and sizes of everything.
        
        See getCasadiIntegrator for description of arguments.
        """
        # Call subclass constructor.
        super(DiscreteSimulator, self).__init__(None, argsizes, argnames)
        
        # Store names and Casadi Integrator object.
        self.__Delta = Delta
        self.verbosity = verbosity
        self.__integrator = getCasadiIntegrator(ode, Delta, argsizes, argnames,
                                                wrap=False, scalar=scalar,
                                                casaditype=casaditype,
                                                verbosity=verbosity)

    def call(self, *args):
        """
        Simulates one timestep and returns a Casadi vector (DM, SX, or MX).
        
        Useful if you are using this object to construct a new symbolic
        function. If you are just simulating with numeric values, see self.sim.
        """
        # Check arguments.
        self._checkargs(args)
        integratorargs = dict(x0=args[0])
        if len(args) > 1:
            integratorargs["p"] = util.safevertcat(args[1:])
        
        # Call integrator.
        nextstep = self.__integrator(**integratorargs)
        xf = nextstep["xf"]
        return xf
    
