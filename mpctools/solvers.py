import numpy as np
from . import util
import casadi
import time
import warnings

"""
Holds solver interfaces, wrappers, and class definitions.
"""

_MAX_VERBOSITY = 100

def setMaxVerbosity(verb=100):
    """
    Sets a module override for maximum verbosity setting.
    """
    global _MAX_VERBOSITY
    _MAX_VERBOSITY = verb


def callSolver(solver, verbosity=None):
    """
    Wrapper to ControlSolver.solve() that returns a single dictionary.
    
    This function is mainly for backward-compatibility now that the option
    runOptimization was removed from nmpc, nmhe, etc.
    
    Returns a dictionary with optimal variables as NumPy arrays. Additional
    keys "t", "obj" and "status" are also present. Finally, if the model used
    collocation, an entry "tc" is also present which is an Nt by Nc array of
    time points.
    """
    if verbosity is not None:
        solver.verbosity = verbosity
    solver.solve()

    returnDict = util.casadiStruct2numpyDict(solver.var)
    returnDict["obj"] = solver.obj
    returnDict["status"] = solver.stats["status"]
    
    Delta = solver.misc.get("Delta", 1)
    N = solver.misc["N"]
    returnDict["t"] = Delta*np.arange(N["t"] + 1)
    if "c" in N and N["c"] > 0:
        r = solver.misc["colloc"]["r"][1:-1] # Throw out endpoints.        
        r = r[np.newaxis,:]
        returnDict["tc"] = returnDict["t"][:-1,np.newaxis] + Delta*r    
    
    return returnDict


# Build a dictionary of names for the time limit setting.
_CPU_TIME_SETTING = util.ReadOnlyDict({"ipopt" : "max_cpu_time",
                                       "qpoases" : "CPUtime",
                                       "bonmin" : "max_cpu_time"})

class ControlSolver(object):
    """
    A simple class for holding a casadi solver object.
    
    Users have access parameter and guess fields to adjust parameters on the
    fly and reuse past trajectories in subsequent optimizations, etc.
    """
    
    # We use the attributes __varguess and __parval to store the values for
    # guesses and parameters, and the attributes __var and __par for the
    # casadi symbolic structures. Users should never need to access __var and
    # __par, so these aren't properties. We do expose __varguess and __parval.
    # After an optimization, __varval holds the optimal values of variables,
    # and this is exposed through the property var.
    
    @property
    def lb(self):
        return self.__lb
        
    @property
    def ub(self):
        return self.__ub
    
    @property
    def guess(self):
        return self.__guess    
    
    @guess.setter
    def guess(self, g):
        self.saveguess(g)
    
    @property
    def defaultguess(self):
        return self.__defaultguess
        
    @defaultguess.setter
    def defaultguess(self, g):
        self.__defaultguess = util.ArrayDict(g)
    
    @property
    def discretevar(self):
        return self.__discretevar
    
    @property
    def conlb(self):
        return self.__conlb
        
    @property
    def conub(self):
        return self.__conub
    
    @property
    def par(self):
        return self.__parval
    
    @property
    def var(self):
        return self.__varval
    
    @property
    def vardict(self):
        if self.__vardict is None:
            self.__vardict = util.casadiStruct2numpyDict(self.__varval)
        return self.__vardict
    
    @property
    def obj(self):
        return self.__objval
    
    @property
    def stats(self):
        return self.__stats
    
    @property
    def verbosity(self):
        return self.__settings["verbosity"]
        
    @verbosity.setter
    def verbosity(self, v):
        v = min(min(max(v, -1), 12), _MAX_VERBOSITY)
        self.__changesettings(verbosity=v)
    
    @property
    def name(self):
        return self.__settings["name"]
        
    @name.setter
    def name(self, n):
        self.__changesettings(name=n)
        
    @property
    def timelimit(self):
        return self.__settings["timelimit"]
        
    @timelimit.setter
    def timelimit(self, t):
        self.__changesettings(timelimit=t)
    
    @property
    def isQP(self):
        return self.__settings["isQP"]
    
    @isQP.setter
    def isQP(self, tf):
        self.__changesettings(isQP=tf)
    
    @property
    def solver(self):
        return self.__settings["solver"]
        
    @solver.setter
    def solver(self, solverstr):
        availablesolvers = util.listAvailableSolvers()
        if solverstr not in availablesolvers["NLP"] + availablesolvers["QP"]:
            errmsg = ("%s is not a valid solver. Available solvers:\n%s" %
                      (solverstr, util.listAvailableSolvers(asstring=True)))
            raise ValueError(errmsg)
        elif solverstr in availablesolvers["QP"] and not self.isQP:
            errmsg = ("%s is a QP solver and self.isQP is False. Please set "
                      "isQP to True or choose a QP solver. Available solvers:"
                      "\n%s" % (solverstr, util.listAvailableSolvers()))
            raise ValueError(errmsg)
        self.__changesettings(solver=solverstr)
    
    def __changesettings(self, **settings):
        """
        Changes fields of the settings dictionary and sets update flag.
        """
        self.__settings.update(settings)
        self.__changed = True
    
    @property
    def varsym(self):
        return self.__var
    
    @property
    def parsym(self):
        return self.__par
    
    def __init__(self, var, varlb, varub, varguess, obj, con, conlb, conub,
                 par=None, parval=None, verbosity=5, timelimit=60, isQP=False,
                 casaditype="SX", name="ControlSolver", casadioptions=None,
                 solveroptions=None, misc=None, solver="ipopt",
                 discretevar=None):
        """
        Initialize the solver object.
        
        Arguments are mostly self-explainatory. var, varlb, varub, and varguess
        should be casadi struct_symMX objects, e.g. the outputs of
        getCasadiVars. obj should be a scalar casadi MX object, and con should
        be a vector casadi MX object (possibly from calling casadi.vertcat on a
        list of constraints). conlb and conub should be numpy vectors of the
        appropriate size. misc is a read-only dictionary for storing to hold
        miscellaneous parameters that cannot be changed.
        
        Typically, it's easiest to build these objects using nmpc, nmhe, or
        sstarg from the tools module, all of which return ControlSolver
        objects.
        """
        
        # First store everybody to the object.
        self.__changed = True
        self.__var = var
        self.__varval = var(np.nan)
        self.__vardict = None # Lazy pudate flag.
        self.__lb = varlb
        self.__ub = varub
        if discretevar is None:
            discretevar = var(False) # Default all continuous variables.
        self.__discretevar = discretevar
        self.__guess = varguess
        self.defaultguess = util.casadiStruct2numpyDict(varguess)
        self.__obj = obj
        self.__con = con
        self.__conlb = conlb
        self.__conub = conub
        
        self.__par = par
        self.__parval = parval
        
        self.__sol = {}
        self.__stats = {}
        self.__settings = {} # Need to initialize this.
        self.__changesettings(isQP=isQP, name=name, verbosity=verbosity,
                              timelimit=timelimit, solver=solver)
        if misc is None:
            misc = {}
        self.misc = util.ReadOnlyDict(**misc)
        #TODO: better way of indicating collocation variables.
        
        # Now initialize the solver object.
        if casadioptions is None:
            casadioptions = {}
        if solveroptions is None:
            solveroptions = {}
        self.initialize(casadioptions, solveroptions)        
        
    def initialize(self, casadioptions=None, solveroptions=None):
        """
        Recreates the solver object completely.
        
        You shouldn't need to do this manually unless you are changing internal
        casadi or solver options (via the respective dictionaries).
        
        For a complete list of solver-specific options, use
        self.getSolverOptions. Note that most options are either strings or
        floats, and any boolean values will likely cause errors.
        """
        if casadioptions is None:
            casadioptions = {}
        else:
            casadioptions = casadioptions.copy()
        if solveroptions is None:
            solveroptions = {}
        else:
            solveroptions = solveroptions.copy()
        
        nlp = {
            "x" : self.__var,
            "f" : self.__obj,
            "g" : self.__con,
        }
        if self.__par is not None:
            nlp["p"] = self.__par
        
        # Print and time limit options.
        if self.solver in set(["ipopt", "bonmin"]):
            solveroptions["print_level"] =  min(12, max(0, self.verbosity))
        elif self.solver == "qpoases":
            solveroptions["verbose"] = self.verbosity >= 10
            if self.verbosity >= 9:
                plevel = "high"
            elif self.verbosity >= 6:
                plevel = "medium"
            elif self.verbosity >= 3:
                plevel = "low"
            else:
                plevel = "none"
            solveroptions["printLevel"] = plevel
        else:
            #TODO: add other solver-specific verbosity code.
            warnings.warn("Solver '%s' does not have a verbosity setting"
                          % self.solver)
        if self.timelimit is not None:
            timesetting = _CPU_TIME_SETTING.get(self.solver, None)
            if timesetting is None:
                warnings.warn("Solver '%s' does not support a time limit."
                              % self.solver)
            else:
                solveroptions[timesetting] = self.timelimit        
             
        # Choose different function whether QP or not.
        #TODO: Specify constant Lagrangian if isQP
        availablesolvers = util.listAvailableSolvers()
        if self.solver in availablesolvers["QP"]:
            solverfunc = casadi.qpsol
            casadioptions.update(solveroptions) #TODO: Verity API difference.
        elif self.solver in availablesolvers["NLP"]:
            if self.isQP:
                if self.solver == "ipopt":
                    for k in ["jac_c_constant", "jac_d_constant",
                              "hessian_constant"]:                    
                        solveroptions[k] = "yes"
                else:
                    warnings.warn("NLP solver '%s' selected for QP."
                                  % self.solver)
            solverfunc = casadi.nlpsol
            if "eval_errors_fatal" not in casadioptions:
                casadioptions["eval_errors_fatal"] = True
            casadioptions["print_time"] = self.verbosity > 2
            casadioptions[self.solver] = solveroptions
        else:
            raise ValueError("Invalid choice of solver: %s" % self.solver)
        
        # Set discrete variables.
        if self.solver in set(["bonmin", "gurobi", "cplex"]):
            discrete =  np.squeeze(np.array(self.discretevar.cat,
                                            dtype=bool)).tolist()
            casadioptions["discrete"] = discrete
        elif np.any(self.discretevar.cat):
            warnings.warn("Discrete variables not supported in %s!"
                          % self.solver)
        
        # Finally, save the solver and unset the changed flag.
        solver = solverfunc(self.name, self.solver, nlp, casadioptions)        
        self.__solver = solver
        self.__changed = False
    
    def getSolverOptions(self, display=True):
        """
        Lists options for the current solver.
        
        Options can be set using solveroptions in ControlSolver.initialize().
        See help in util.getSolverOptions for more details.
        """
        options = util.getSolverOptions(self.solver, display=display)
        if display:
            print("Options can be set using solveroptions in"
                " ControlSolver.initialize().\n")
        return options
    
    def solve(self):
        """
        Solve the current solver object.
        """
        # Solve the problem and get optimal variables.
        starttime = time.time()
        if self.__changed:
            self.initialize()
        solver = self.__solver
        
        # Now set guess and bounds.
        solverargs = {
            "x0" : self.guess,
            "lbx" : self.lb,
            "ubx" : self.ub,
            "lbg" : self.conlb,
            "ubg" : self.conub,
        }
        if self.par is not None:
            solverargs["p"] = self.par
        
        # Need something special to prevent c code from printing; in
        # particular, we want to suppress Ipopt's splash message if
        # verbosity <= -1. Note that this redirection can have some weird
        # side-effects, so that's why we don't do it for verbosity = 0.
        if self.verbosity <= -1:
            printcontext = util.stdout_redirected
        else:
            printcontext = util.dummy_context
        with printcontext():
            sol = solver(**solverargs)
            stats = solver.stats()
        self.__sol = sol
        self.__varval = self.__var(sol["x"])
        self.__vardict = None # Lazy update in getter.
        self.__objval = float(sol["f"])
        endtime = time.time()
        
        # Grab some stats.
        status = stats.get("return_status", "UNKNOWN")
        if self.verbosity > 0:
            print("Solver Status:", status)
            if status == "NonIpopt_Exception_Thrown":
                print("***Warning: NaN or Inf encountered during function "
                    "evaluation.")
        if self.verbosity > 1:
            print("Took %g s." % (endtime - starttime,))
        self.stats["status"] = status
        self.stats["time"] = endtime - starttime
         
    def saveguess(self, newguess=None, toffset=None, default=False,
                  infercolloc=True, pad=True):
        """
        Stores a given guess in one of three ways.
        
        The first way is to pass a dict of numpy arrays. Each entry should
        have time along the first dimension and appropriate variable size in
        following dimensions. For this case, toffset defaults to zero.
        
        The second way is to not pass a guess but instead set default=True.
        This method will revert to whatever was given as the default guess when
        the solver was originally created (note that you can manually change
        self.defaultguess). For this case, toffset defaults to zero.
        
        The third and final way is to not specify a guess and to keep
        default=False. In this case, the guess will be pulled from the previous
        optimization (i.e., self.var). In this case, toffset defaults to 1,
        which is useful for using the previous optimization as a guess for the
        next optimization.
        
        The keyword argument pad determines what to do for the "missing"
        elements when toffset is not zero. If pad=True (the default), then the
        missing entries will be filled by duplicating the first and/or last
        given entries to fill any gaps at the beginning and/or end. If
        pad=False, then the missing entries will not be changed. For example,
        suppose toffset=1 and the guess has the same number of time points as
        the solver. Thus, the final time point is missing. If pad=True, it will
        be filled using the second-to-last time point; otherwise, it will not
        be changed.
        """
        getguess = None
        if newguess is None:
            if default:
                # Just use default guess. toffset default is 0.
                if toffset is None:
                    toffset = 0
                newguess = self.__defaultguess
            else:
                # Use self.var. toffset default is 1. Need special getguess.
                newguess = self.var
                if toffset is None:
                    toffset = 1
                def getguess(k, t=None):
                    """Gets element from casadi struct."""
                    key = k if t is None else (k, t)
                    return newguess[key]
        else:
            # Guess supplied. toffset default is 0.
            if toffset is None:
                toffset = 0 
        
        # By default, getguess assumes ArrayDict-like structure.
        if getguess is None:
            def getguess(k, t=None):
                """Gets element from ArrayDict-like."""
                if t is None:
                    ret = newguess[k]
                else:
                    ret = newguess[k][t,...]
                return ret
        
        # Check for extra fields in guess. keys() is important!
        extra = set(newguess.keys()).difference(self.guess.keys())
        if len(extra) > 0:
            warnings.warn("Ignoring extra fields in guess: %r." % (extra,))
        
        # Now actually save guess.
        for k in self.guess.keys(): # keys() is important!
            Tguess = len(getguess(k))
            Tself = len(self.guess[k])
            if pad:
                tmin = 0
                tmax = Tself
            else:
                tmin = max(0, toffset)
                tmax = min(Tself, Tguess - toffset)
            
            # Now actually store the stuff.           
            for t in range(tmin, tmax):
                tself = t
                tguess = max(0, min(t + toffset, Tguess - 1))
                self.guess[k,tself] = getguess(k, tguess)
                
        # Finally, infer a collocation guess.
        if (infercolloc and "colloc" in self.misc
                and "xc" not in newguess.keys()): # keys() is important!
            self.infercollocguess()
    
    def newmeasurement(self, y, u=None, x0bar=None):
        """
        Adds new measurement for MHE.
        
        The current u and a new update for the prior x0bar can also be given.
        """
        parkeys = self.par.keys() # keys() is important!
        cycles = {}
        if "y" not in parkeys:
            raise TypeError("y is missing from par! Not from mhe().")
        cycles["y"] = y
        if u is not None:
            if "u" not in parkeys:
                raise TypeError("u is missing from par! Not from mhe().")
            cycles["u"] = u
        
        # Actually cycle things.
        # TODO: split out saveguess and use same logic.
        for (k, newval) in cycles.items():
            for t in range(len(self.par[k]) - 1):
                self.par[k,t] = self.par[k,t + 1]
            self.par[k,-1] = newval
            
        # Also do x0bar.
        if x0bar is not None:
            if "x0bar" not in parkeys:
                raise TypeError("Object does not accept x0bar!")
            self.par["x0bar",0] = x0bar
    
    def infercollocguess(self):
        """Infers a guess for "xc" based on the guess for "x"."""
        try:
            r = self.misc["colloc"]["r"]
        except KeyError:
            raise ValueError("No collocation variables are present!")
        util._infercolloc(r, self.guess)
    
    def fixvar(self,var,t,val,indices=None):
        """
        Fixes variable var at time t to val.
        
        Indices can be specified as a list to fix only a subset of values.
        """
        if indices is None:
            self.lb[var,t] = val
            self.ub[var,t] = val
            self.guess[var,t] = val
        else:
            self.lb[var,t,indices] = val
            self.ub[var,t,indices] = val
            self.guess[var,t,indices] = val      
    
    def addconstraints(self, newcon, ctype=None, lb=None, ub=None):
        """
        Adds a constraint to the optimization problem.
        
        con should be a CasADi symbolic expression, or a list of expressions
        that can be vertically concatenated into a vector to give new
        constraints.
        
        To specify the bounds on constraints, there are two options. First, you
        can specify ctype as '=', '<', or '>' to define equality or one-sided
        inequality constraints. Alternatively, you can specify lb or ub as
        scalars or vectors with the same concatenated size as con. Note that if
        either of lb or ub is provided, then ctype is ignored, and any missing
        bound is set to +/-infinity. If only con is provided, then the
        constraints are added as equalities.
        """
        newcon = util.safevertcat(newcon)
        if ctype is None:
            if lb is None and ub is None:
                ctype = "="
            else:
                if lb is None:
                    lb = -np.inf
                if ub is None:
                    ub = np.inf
        if ctype is not None:
            if ctype == "=":
                lb = 0
                ub = 0
            elif ctype == "<":
                lb = -np.inf
                ub = 0
            elif ctype == ">":
                lb = 0
                ub = np.inf
        
        def promote(x):
            """Promotes x to be a vector of the appropriate length."""
            x = np.array(x)
            if x.size == 1:
                x = x.flatten()*np.ones(newcon.numel())
            elif x.ndims != 1:
                raise ValueError("lb and ub must be vectors!")
            return x
        lb = promote(lb)
        ub = promote(ub)
        
        def cat(old, new):
            """Vertically concatenates two vector objects."""
            return util.safevertcat([old, new])
        self.__con = cat(self.__con, newcon)
        self.__conlb = cat(self.__conlb, lb)
        self.__conub = cat(self.__conub, ub)
        
        self.__changed = True

    def addtoobjective(self, newobj):
        """
        Adds a new term to the objective function.
        
        newobj must be a scalar CasADi expression.
        """
        if not hasattr(newobj, "shape") or newobj.shape() != (1, 1):
            raise ValueError("newobj must be a scalar expression!")
        self.__obj = self.__obj + newobj
        self.__changed = True
        
