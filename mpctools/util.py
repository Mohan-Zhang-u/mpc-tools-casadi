
import scipy.linalg
import casadi
import casadi.tools as ctools
import collections
import numpy as np
import pdb
import itertools
import sys
import os
import warnings
from contextlib import contextmanager
from .compat import execfile, reduce # analysis:ignore

# First, we grab a few things from the CasADi module.
DM = casadi.DM
MX = casadi.MX
vertcat = casadi.vertcat
sum1 = casadi.sum1

# Grab pdb function to emulate Octave/Matlab's keyboard().
keyboard = pdb.set_trace

# Also make a wrapper to numpy's array function that forces float64 data type.
def array(x, dtype=np.float64, **kwargs):
    """
    Wrapper to NumPy's array that forces floating point data type.
    
    Uses numpy.float64 as the default data type instead of trying to infer it.
    See numpy.array for other keyword arguments.
    """
    kwargs["dtype"] = dtype
    return np.array(x, **kwargs)


# Now give the actual functions.
def rk4(f,x0,par,Delta=1,M=1):
    """
    Does M RK4 timesteps of function f with variables x0 and parameters par.
    
    The first argument of f must be var, followed by any number of parameters
    given in a list in order.
    
    Note that var and the output of f must add like numpy arrays.
    """
    h = Delta/M
    x = x0
    j = 0
    while j < M: # For some reason, a for loop creates problems here.       
        k1 = f(x,*par)
        k2 = f(x + k1*h/2,*par)
        k3 = f(x + k2*h/2,*par)
        k4 = f(x + k3*h,*par)
        x = x + (k1 + 2*k2 + 2*k3 + k4)*h/6
        j += 1
    return x


def atleastnd(arr,n=2):
    """
    Adds an initial singleton dimension to arrays with fewer than n dimensions.
    """
    if len(arr.shape) < n:
        arr = arr.reshape((1,) + arr.shape)
    return arr


def getLinearizedModel(f,args,names=None,Delta=None,returnf=True,forcef=False):
    """
    Returns linear (affine) state-space model for f at the point in args.
    
    Note that f must be a casadi function (e.g., the output of getCasadiFunc).    
    
    names should be a list of strings to specify the dictionary entry for each
    element. E.g., for args = [xs, us] to linearize a model in (x,u), you
    might choose names = ["A", "B"]. These entries can then be accessed from
    the returned dictionary to get the linearized state-space model.
    
    If "f" is not in the list of names, then the return dict will also include
    an "f" entry with the actual value of f at the linearization point. To
    disable this, set returnf=False.
    """
    # Decide names.
    if names is None:
        names = ["A"] + ["B_%d" % (i,) for i in range(1,len(args))]
    
    # Evaluate function.
    fs = np.array(f(*args))    
    
    # Now do jacobian.
    jacobians = []
    for i in range(len(args)):
        jac = jacobianfunc(f, i) # df/d(args[i]).
        jacobians.append(np.array(jac(*args)))
    
    # Decide whether or not to discretize.
    if Delta is not None:
        (A, Bfactor) = c2d(jacobians[0],np.eye(jacobians[0].shape[0]),Delta)
        jacobians = [A] + [Bfactor.dot(j) for j in jacobians[1:]]
        fs = Bfactor.dot(fs)
    
    # Package everything up.
    ss = dict(zip(names, jacobians))
    if returnf and ("f" not in ss or forcef):
        ss["f"] = fs
    return ss    

    
def c2d(A, B, Delta, Bp=None, f=None, asdict=False):
    """
    Discretizes affine system (A, B, Bp, f) with timestep Delta.
    
    This includes disturbances and a potentially nonzero steady-state, although
    Bp and f can be omitted if they are not present.
    
    If asdict=True, return value will be a dictionary with entries A, B, Bp,
    and f. Otherwise, the return value will be a 4-element list [A, B, Bp, f]
    if Bp and f are provided, otherwise a 2-element list [A, B].
    """
    n = A.shape[0]
    I = np.eye(n)
    D = scipy.linalg.expm(Delta*np.vstack((np.hstack([A, I]),
                                     np.zeros((n, 2*n)))))
    Ad = D[:n,:n]
    Id = D[:n,n:]
    Bd = Id.dot(B)
    Bpd = None if Bp is None else Id.dot(Bp)
    fd = None if f is None else Id.dot(f)   
    
    if asdict:
        retval = dict(A=Ad, B=Bd, Bp=Bpd, f=fd)
    elif Bp is None and f is None:
        retval = [Ad, Bd]
    else:
        retval = [Ad, Bd, Bpd, fd]
    return retval


def c2dObjective(a,b,q,r,Delta):
    """
    Discretization with continuous objective.

    Converts from continuous-time objective
    
                 / \Delta
        l(x,u) = |        x'qx + u'ru  dt
                 / 0
        dx/dt = ax + bu
    
    to the equivalent
    
        L(x,u) = x'Qx + 2x'Mu + u'Qu
        x^+ = Ax + Bu
        
    in discrete time.
    
    Formulas from Pannocchia, Rawlings, Mayne, and Mancuso (2014).
    """
    # Make sure everything is a matrix.
    for m in [a,b,q,r]:
        try:
            if len(m.shape) != 2:
                raise ValueError("All inputs must be 2D arrays!")
        except AttributeError:
            raise TypeError("All inputs must have a shape attribute!")
            
    # Get sizes.
    Nx = a.shape[1]
    Nu = b.shape[1]
    for (m,s) in [(a,(Nx,Nx)), (b,(Nx,Nu)), (q,(Nx,Nx)), (r,(Nu,Nu))]:
        if m.shape != s:
            raise ValueError("Incorrect sizes for inputs!")
    
    # Now stack everybody up.
    i = [slice(j*Nx,(j+1)*Nx) for j in range(3)] + [slice(3*Nx,3*Nx+Nu)]
    c = np.zeros((3*Nx + Nu,)*2)
    c[i[0],i[0]] = -a.T
    c[i[1],i[1]] = -a.T
    c[i[2],i[2]] = a
    c[i[0],i[1]] = np.eye(Nx)
    c[i[1],i[2]] = q
    c[i[2],i[3]] = b
    
    # Now exponentiate and grab everybody.
    C = scipy.linalg.expm(c*Delta);
    F3 = C[i[2],i[2]]
    G3 = C[i[2],i[3]]
    G2 = C[i[1],i[2]]
    H2 = C[i[1],i[3]]
    K1 = C[i[0],i[3]]
    
    # Then, use formulas.
    A = F3
    B = G3
    Q = F3.T.dot(G2)
    M = F3.T.dot(H2)
    R = r*Delta + b.T.dot(F3.T.dot(K1)) + (b.T.dot(F3.T.dot(K1))).T
    
    return [A,B,Q,R,M]


def dlqr(A,B,Q,R,M=None):
    """
    Get the discrete-time LQR for the given system.
    
    Stage costs are
    
        x'Qx + 2*x'Mu + u'Qu
        
    with M = 0 if not provided.
    """
    # For M != 0, we can simply redefine A and Q to give a problem with M = 0.
    if M is not None:
        RinvMT = scipy.linalg.solve(R,M.T)
        Atilde = A - B.dot(RinvMT)
        Qtilde = Q - M.dot(RinvMT)
    else:
        Atilde = A
        Qtilde = Q
        M = np.zeros(B.shape)
    Pi = scipy.linalg.solve_discrete_are(Atilde,B,Qtilde,R)
    K = -scipy.linalg.solve(B.T.dot(Pi).dot(B) + R, B.T.dot(Pi).dot(A) + M.T)
    
    return [K, Pi]

    
def dlqe(A,C,Q,R):
    """
    Get the discrete-time Kalman filter for the given system.
    """
    P = scipy.linalg.solve_discrete_are(A.T,C.T,Q,R)
    L = scipy.linalg.solve(C.dot(P).dot(C.T) + R, C.dot(P)).T     
    
    return [L, P]

    
def mtimes(*args, **kwargs):
    """
    More flexible version casadi.tools.mtimes.
    
    Matrix multiplies all of the given arguments and returns the result. If any
    inputs are Casadi's SX or MX data types, uses Casadi's mtimes. Otherwise,
    uses a sequence of np.dot operations.
    
    Keyword arguments forcedot or forcemtimes can be set to True to pick one
    behavior or another.
    """
    # Get keyword arguments.
    forcemtimes = kwargs.pop("forcemtimes", None)
    forcedot = kwargs.pop("forcedot", False)
    if len(kwargs) > 0:
        raise TypeError("Invalid keywords: %s" % list(kwargs))    
    
    # Pick whether to use mul or dot.
    if forcemtimes:
        if forcedot:
            raise ValueError("forcemtimes and forcedot can't both be True!")
        useMul = True
    elif forcedot:
        useMul = False
    else:
        useMul = False
        symtypes = set(["SX", "MX"])
        for a in args:
            atype = getattr(a, "type_name", lambda : None)()
            if atype in symtypes:
                useMul = True
                break

    # Now actually do multiplication.
    ans = ctools.mtimes(args) if useMul else reduce(np.dot, args)
    return ans


def flattenlist(l,depth=1):
    """
    Flattens a nested list of lists of the given depth.
    
    E.g. flattenlist([[1,2,3],[4,5],[6]]) returns [1,2,3,4,5,6]. Note that
    all sublists must have the same depth.
    """
    for i in range(depth):
        l = list(itertools.chain.from_iterable(l))
    return l


def casadiStruct2numpyDict(struct, arraydict=False):
    """
    Takes a casadi struct and turns int into a dictionary of numpy arrays.
    
    Access patterns are now as follows:

        struct["var",t,...] = dict["var"][t,...]
        
    Note that if the struct entry is empty, then there will not be a
    corresponding key in the returned dictonary.
    
    If arraydict=True, then the return value will be an ArrayDict object, which
    is a dictionary 
    """ 
    npdict = {}
    for k in struct.keys():
        if len(struct[k]) > 0:
            npdict[k] = listcatfirstdim(struct[k])   
    return npdict


def listcatfirstdim(l):
    """
    Takes a list of numpy arrays, prepends a dimension, and concatenates.
    """
    newl = []
    for a in l:
        a = np.array(a)
        if len(a.shape) == 2 and a.shape[1] == 1:
            a.shape = (a.shape[0],)
        a.shape = (1,) + a.shape
        newl.append(a)
    return np.concatenate(newl)


def smushColloc(t=None, x=None, tc=None, xc=None, Delta=1, asdict=False):
    """
    Combines point x variables and interior collocation xc variables.
    
    The sizes of each input must be as follows:
     -  t: (Nt+1,)
     -  x: (Nt+1,Nx)
     - tc: (Nt,Nc)
     - xc: (Nt,Nx,Nc)
    with Nt the number of time periods, Nx the number of states in x, and Nc
    the number of collocation points on the interior of each time period. Note
    that if t or tc is None, then they are constructed using a timestep of
    Delta (with default value 1).
    
    Note that t and tc will be calculated if they are not provided.    
    
    Returns arrays T with size (Nt*(Nc+1) + 1,) and X with size 
    (Nt*(Nc+1) + 1, Nx) that combine the collocation points and edge points.
    Also return Tc and Xc which only contain the collocation points.

    If asdict=True, then results are returned in a dictionary. This contains
    fields "t" and "x" with interior collocation and edge points together,
    "tc" and "xc" with just the inter collocation points, and "tp" and "xp"
    which are only the edge points.         
    """
    # Make sure at least x and xc were supplied.
    if x is None or xc is None:
        raise TypeError("x and xc must both be supplied!")
    
    # Make copies.
    if t is not None:
        t = t.copy()
    if tc is not None:
        tc = tc.copy()
    x = x.copy()
    xc = xc.copy()
    
    # Build t and tc if not given.
    if t is None or tc is None:
        Nt = xc.shape[0]
        if t is None:                
            t = np.arange(0, Nt+1)*Delta
        else:
            t.shape = (t.size,)
        from . import colloc
        Nc = xc.shape[2]
        [r, _, _, _] = colloc.weights(Nc, include0=False, include1=False)
        r.shape = (r.size,1)
        tc = (t[:-1] + r*Delta).T.copy()
    
    # Add some dimensions to make sizes compatible.
    t.shape = (t.size,1)
    x.shape += (1,)
    
    # Begin the smushing.
    T = np.concatenate((t[:-1],tc),axis=1)    
    X = np.concatenate((x[:-1,...],xc),axis=2)
    
    # Have to do some permuting for X. Order is now (t,c,x).
    X = X.transpose((0,2,1))
    Xc = xc.transpose((0,2,1))
    
    # Now flatten.
    T.shape = (T.size,)
    Tc = tc.flatten()
    X = X.reshape((X.shape[0]*X.shape[1],X.shape[2]))
    Xc = Xc.reshape((Xc.shape[0]*Xc.shape[1],Xc.shape[2]))
    
    # Then add final elements.
    T = np.concatenate((T,t[-1:,0]))
    X = np.concatenate((X,x[-1:,:,0]))
    
    if asdict:
        ret = dict(t=T, x=X, tc=Tc, xc=Xc, tp=np.squeeze(t), xp=np.squeeze(x))
    else:
        ret = [T, X, Tc, Xc]
    return ret


@contextmanager
def nice_stdout():
    """
    Redirect C++ output to Python's stdout (or at least attempts to).
    
    Taken from casadi.tools.io with some modifications for Python 3
    compatbility.
    """
    (r, w) = os.pipe()
    sys.stdout.flush()
    backup = os.dup(1)
    os.dup2(w, 1)
    try:
        yield
    finally:
       os.dup2(backup, 1)
       os.write(w, b"x")
       sys.stdout.write(os.read(r, 2**20)[:-1]
                        .decode(sys.getdefaultencoding()))
       os.close(r)
       os.close(w)
       os.close(backup)


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    context to redirect all Python output, including C code.
    
    Used in a with statement, e.g.,

        with stdout_redirected(to=filename):
            print "from Python"
            ipopt.solve()
    
    will capture both the Python print output and any output of calls to C
    libraries (e.g., IPOPT).
    
    Note that this makes use of CasADi's tools.io.nice_stdout context, which
    means all C output is buffered and then returned all at once. Thus, this is
    only really useful if don't need to see output as it is created.
    """
    old_stdout = sys.stdout
    with open(to, "w") as new_stdout:
        with nice_stdout(): # Buffers C output to Python stdout.
            sys.stdout = new_stdout # Redefine Python stdout.
            try:
                yield # Allow code to be run with the redirected stdout.
            finally:
                sys.stdout.flush()
                sys.stdout = old_stdout # Reset stdout.

   
@contextmanager
def dummy_context(*args):
    """
    Dummy context for a with statement.
    """
    # We need this in solvers.py.
    yield


# Below, we don't inherit from dict because its methods sometimes don't use
# __setitem__, and so results can be inconsistent. Instead, we use an abstract
# base class from the collections module. See Section 8.3.6 in the docs at
# https://docs.python.org/2/library/collections.html
class ArrayDict(collections.MutableMapping):
    """
    Python dictionary of numpy arrays.

    When instantiating or when setting an item, calls np.array to convert
    everything.
    """
    def __init__(self, *args, **kwargs):
        """
        Creates a dictionary and then wraps everything in np.array.
        """
        self.dtype = float
        self.__arraydict__ = dict() # This is where we actually store things.
        self.update(dict(*args, **kwargs)) # We get this method for free.      
    
    def __setitem__(self, k, v):
        """
        Wraps v with np.array before setting.
        """
        self.__arraydict__[k] = np.array(v, dtype=self.dtype)
    
    def copy(self):
        """
        Returns a copy of self with each array copied as well.
        """
        return {k : v.copy() for (k, v) in self.__arraydict__.items()}
    
    # The rest of the methods just perform the corresponding dict action.
    def __getitem__(self, k):
        return self.__arraydict__[k]
    
    def __len__(self):
        return len(self.__arraydict__)
    
    def __iter__(self):
        return iter(self.__arraydict__)
        
    def __delitem__(self, k):
        del self.__arraydict__[k]
        
    def __repr__(self):
        return repr(self.__arraydict__)


class ReadOnlyDict(dict):
    """Read-only dictionary to prevent user changes."""
    def __readonly__(self, *args, **kwargs):
        raise NotImplementedError("Cannot modify ReadOnlyDict")
    __setitem__ = __readonly__
    __delitem__ = __readonly__
    pop = __readonly__
    popitem = __readonly__
    clear = __readonly__
    update = __readonly__
    setdefault = __readonly__
    del __readonly__


def strcolor(s, color=None, bold=False):
    """
    Adds ANSI escape sequences to colorize string s.
    
    color must be one of the eight standard colors (RGBCMYKW). Accepts full
    names or one-letter abbreviations.
    
    Keyword bold decides to make string bold.
    """
    colors = dict(_end="\033[0m", _bold="\033[1m", b="\033[94m", c="\033[96m",
        g="\033[92m", k="\033[90m", m="\033[95m", r="\033[91m", w="\033[97m",
        y="\033[93m")
    colors[""] = "" # Add a few defaults.
    colors[None] = ""
    
    # Decide what color user gave.
    c = color.lower()
    if c == "black":
        c = "k"
    elif len(c) > 0:
        c = c[0]
    try:
        c = colors[c]
    except KeyError:
        raise ValueError("Invalid color choice '%s'!" % (color,))
    
    # Build up front and back of string and return.
    front = (colors["_bold"] if bold else "") + c
    back = (colors["_end"] if len(front) > 0 else "")
    return "%s%s%s" % (front, s, back)


def ekf(f,h,x,u,w,y,P,Q,R,f_jacx=None,f_jacw=None,h_jacx=None):
    """
    Updates the prior distribution P^- using the Extended Kalman filter.
    
    f and h should be casadi functions. f must be discrete-time. P, Q, and R
    are the prior, state disturbance, and measurement noise covariances. Note
    that f must be f(x,u,w) and h must be h(x).
    
    If specified, f_jac and h_jac should be initialized jacobians. This saves
    some time if you're going to be calling this many times in a row, although
    it's really not noticable unless the models are very large. Note that they
    should return a single argument and can be created using
    mpctools.util.jacobianfunc.
    
    The value of x that should be fed is xhat(k | k-1), and the value of P
    should be P(k | k-1). xhat will be updated to xhat(k | k) and then advanced
    to xhat(k+1 | k), while P will be updated to P(k | k) and then advanced to
    P(k+1 | k). The return values are a list as follows
    
        [P(k+1 | k), xhat(k+1 | k), P(k | k), xhat(k | k)]
        
    Depending on your specific application, you will only be interested in
    some of these values.
    """
    
    # Check jacobians.
    if f_jacx is None:
        f_jacx = jacobianfunc(f, 0)
    if f_jacw is None:
        f_jacw = jacobianfunc(f, 2)
    if h_jacx is None:
        h_jacx = jacobianfunc(h, 0)
        
    # Get linearization of measurement.
    C = np.array(h_jacx(x))
    yhat = np.array(h(x)).flatten()
    
    # Advance from x(k | k-1) to x(k | k).
    xhatm = x                                          # This is xhat(k | k-1)    
    Pm = P                                             # This is P(k | k-1)    
    L = scipy.linalg.solve(C.dot(Pm).dot(C.T) + R, C.dot(Pm)).T          
    xhat = xhatm + L.dot(y - yhat)                     # This is xhat(k | k) 
    P = (np.eye(Pm.shape[0]) - L.dot(C)).dot(Pm)       # This is P(k | k)
    
    # Now linearize the model at xhat.
    w = np.zeros(w.shape)
    A = np.array(f_jacx(xhat, u, w))
    G = np.array(f_jacw(xhat, u, w))
    
    # Advance.
    Pmp1 = A.dot(P).dot(A.T) + G.dot(Q).dot(G.T)       # This is P(k+1 | k)
    xhatmp1 = np.array(f(xhat, u, w)).flatten()     # This is xhat(k+1 | k)    
    
    return [Pmp1, xhatmp1, P, xhat]


def _infercolloc(r, guess):
    """
    Infer a guess for collocation states "xc" based on the guess for "x".
    
    r should be the first output of colloc.weights, giving the multipliers
    between 0 and 1 for each collocation time point.
    """
    guesskeys = set(guess.keys())
    if not guesskeys.issuperset(["x", "xc"]):
        raise ValueError("Missing keys! Must have 'x' and 'xc'.")
    r = r[np.newaxis,1:-1]
    x1 = np.array(guess["x",0])
    for t in range(len(guess["x"]) - 1):
        x0 = x1
        x1 = np.array(guess["x",t + 1])
        guess["xc",t] = r*x0 + (1 - r)*x1


# Conveinence function for getting derivatives.
def getScalarDerivative(f, nargs=1, wrt=(0,), vectorize=True):
    """
    Returns a function that gives the derivative of the function scalar f.
    
    f must be a function that takes nargs scalar entries and returns a single
    scalar. Derivatives are taken with respect to the variables specified in
    wrt, which must be a tuple of integers. E.g., to take a second derivative
    with respect to the first argument, specify wrt=(0,0).
    
    vectorize is a boolean flag to determine whether or not the function should
    be wrapped with numpy's vectorize. Note that vectorized functions do not
    play well with Casadi symbolics, so set vectorize=False if you wish to
    use the function later on with Casadi symbolics.
    """
    x = [casadi.SX.sym("x" + str(n)) for n in range(nargs)]
    dfdx_expression = f(*x)
    for i in wrt:
        dfdx_expression = casadi.jacobian(dfdx_expression, x[i])
    dfcasadi = casadi.Function("dfdx", x, [dfdx_expression])
    def dfdx(*x):
        return dfcasadi(*x)
    if len(wrt) > 1:
        funcstr = "d^%df/%s" % (len(wrt), "".join(["x%d" % (i,) for i in wrt]))
    else:
        funcstr = "df/dx"
    dfdx.__doc__ = "\n%s = %s" % (funcstr, repr(dfdx_expression))
    if vectorize:    
        ret = np.vectorize(dfdx, otypes=[np.float])
    else:
        ret = dfdx
    return ret


# Function to list available Casadi plugins, e.g., for solving NLPs.
def getCasadiPlugins(keep=None):
    """
    Returns a dictionary of casadi plugin (name, type).

    If keep is not None, it should be a list of plugin types to keep. Only
    plugins of these types are in the return dictionary.    
    """
    for suffix in ["getPlugins", "plugins"]:
        func = 'CasadiMeta_' + suffix
        if hasattr(casadi, func):
            plugins = getattr(casadi, func)().split(";")
            break
    else:
        raise RuntimeError("Unable to get Casadi plugins!")
    plugins = dict(tuple(reversed(p.split("::"))) for p in plugins)
    if keep is not None:
        plugins = {k : v for (k, v) in plugins.items() if v in keep}
    return plugins


# Functions for turning solver documentation tables into a Python dict.
# TODO: package these functions into a class.
_DocCell = collections.namedtuple("DocCell", ["id", "default", "doc"])

def _getDocCell(lines, joins=("", "", "", " ")):
    """
    Returns a DocCell tuple for the set of lines.
    
    joins is a tuple of strings to say how to join multiple lines in a given
    cell. It must have exactly one entry for each cell    
    """
    Ncol = len(joins)
    fields = [[] for i in range(Ncol)]
    for line in lines:
        cells = line.split(" | ", Ncol - 1)
        cells[0] = cells[0].lstrip().lstrip("|")
        cells[-1] = cells[-1].rstrip().rstrip("|")
        if len(cells) != Ncol:
            raise ValueError("Wrong number of columns.")
        for (i, c) in enumerate(cells):
            fields[i].append(c.strip())
    fields = [j.join(f) for (j, f) in zip(joins, fields)]
    types = {"OT_INTEGER" : int, "OT_STRING" : str, "OT_REAL" : float,
             "OT_INT" : int, "OT_DICT" : dict,
             "OT_DOUBLE" : float, "OT_BOOL" : bool, "OT_STR" : str,
             "OT_INTVECTOR" : _LambdaType(lambda x : [int(i) for i in x],
                                          "list[int]"),
             "OT_STRINGVECTOR" : _LambdaType(lambda x : [str(i) for i in x],
                                             "list[str]"),
             }
    [thisid, thistype, thisdefault, thisdoc] = fields
    try:
        thisdefault = types[thistype](thisdefault)
    except (ValueError, TypeError):
        includetype = True
        typefunc = types[thistype]
        if thisdefault == "None" or thisdefault == "GenericType()":
            thisdefault = None
        elif typefunc is int:
            try:
                thisdefault = int(float(thisdefault))
                includetype = False
            except (ValueError, TypeError):
                pass
        if includetype:
            thisdefault = (thisdefault, typefunc)
    except KeyError:
        warnings.warn("Unknown type for '%s', '%s'." % (thisid, thistype))
    return _DocCell(thisid, thisdefault, thisdoc)

_TABLE_START = "+=" # String prefix that starts the table.
_CELL_END = "+-" # String prefix that ends the cell.
_CELL_CONTENTS = "|" # String prefix that continues the cell.

class _LambdaType(object):
    """Surrogate type defined by a lambda function (or similar)."""
    def __init__(self, func, typerepr):
        """Initialize with function and representation."""
        self.__typerepr = typerepr
        self.__func = func
    def __call__(self, val):
        return self.__func(val)
    def __repr__(self):
        return "<type '%s'>" % self.__typerepr
    def __str__(self):
        return repr(self)

def _getDocDict(docstring):
    """
    Returns a dictionary of options drawn from docstring.
    
    Keys are option names, and values are a tuple with (default value,
    text description).
    """
    # Strip header from documentation string.
    lineiter = itertools.dropwhile(lambda x : not x.startswith(_TABLE_START),
                                   docstring.split("\n"))
    try:
        next(lineiter)
    except StopIteration:
        raise ValueError("No table found!")
        
    # Loop through table cells.
    thiscell = []
    allcells = []
    for line in lineiter:
        if line.startswith(_CELL_END):
            allcells.append(_getDocCell(thiscell))
            thiscell = []
        elif line.startswith(_CELL_CONTENTS):
            thiscell.append(line)
        else:
            break
    
    # Now return the dictionary.
    return {c.id : (c.default, c.doc) for c in allcells}


def getSolverOptions(solver, display=True):
        """
        Returns a dictionary of solver-specific options.
        
        Dictionary keys are option names, and values are tuples with the
        default value of each option and a text description. Notice that
        default values are always given, not any values that you may have set.
        Also, in some cases, the default may be a tuple whose first entry is
        the value and whose second entry is a type.        
        
        If display is True, all options are also printed to the screen.
        """
        availablesolvers = listAvailableSolvers(asstring=False)
        if solver in availablesolvers["NLP"]:
            docstring = casadi.doc_nlpsol(solver)
        elif solver in availablesolvers["QP"]:
            docstring = casadi.doc_qpsol(solver)
        else:
            raise ValueError("Unknown solver: '%s'." % solver)
        options = _getDocDict(docstring)
        if display:
            print("Available options [default] for %s:\n" % solver)
            for k in sorted(options.keys()):
                print(k, "[%r]: %s\n" % options[k])
        return options    


def listAvailableSolvers(asstring=False, front="    ", categorize=True):
    """
    Returns available solvers as a string or a dictionary.
    
    If asstring is True, lists solvers as in two categories (QP and NLP) on
    separate lines with the front string at the beginning of each line. If
    asstring is false, returns a dictionary with list entries "QP" and "NLP"
    containing the available solvers of each type.
    """
    availablesolvers = getCasadiPlugins(["Nlpsol", "Qpsol", "Conic"])
    solvers = dict(NLP=[], QP=[])
    for (k, v) in availablesolvers.items():
        if v == "Nlpsol":
            solvers["NLP"].append(k)
        elif v == "Qpsol" or v == "Conic":
            solvers["QP"].append(k)
    if asstring:
        types = ["%s : %s" % (s, ", ".join(solvers[s])) for s in ["QP", "NLP"]]
        retval = front + ("\n" + front).join(types)
    else:
        if not categorize:
            solvers = flattenlist(list(solvers.values()))
        retval = solvers
    return retval


def safevertcat(x):
    """
    Safer wrapper for Casadi's vertcat.
    
    the input x is expected to be an iterable containing multiple things that
    should be concatenated together. This is in contrast to Casadi 3.0's new
    version of vertcat that accepts a variable number of arguments. We retain
    this (old, Casadi 2.4) behavior because it makes it easier to check types.    
    
    If a single SX or MX object is passed, then this doesn't do anything.
    Otherwise, if all elements are numpy ndarrays, then numpy's concatenate
    is called. If anything isn't an array, then casadi.vertcat is called.
    """
    symtypes = set(["SX", "MX"])
    xtype = getattr(x, "type_name", lambda : None)()
    if xtype in symtypes:
        val = x
    elif (not isinstance(x, np.ndarray) and
            all(isinstance(a, np.ndarray) for a in x)):
        val = np.concatenate(x)
    else:
        val = casadi.vertcat(*x)
    return val


def jacobianfunc(func, indep, dep=0, name=None):
    """
    Returns a Casadi Function to evaluate the Jacobian of func.
    
    func should be a casadi.Function object. indep and dep should be the index
    of the independent and dependent variables respectively. They can be
    (zero-based) integer indices, or names of variables as strings.
    """
    if name is None:
        name = "jac_" + func.name()
    jacname = ["jac"]
    for (i, arglist) in [(dep, func.name_out), (indep, func.name_in)]:
        if isinstance(i, int):
            i = arglist()[i]
        jacname.append(i)
    jacname = ":".join(jacname)
    return func.factory(name, func.name_in(), [jacname])


def runfile(file, scope=None):
    """
    Executes a file in the given scope and return the dict of variables.
    
    The default is a new scope. If an existing scope is given, it is modified
    in place.
    """
    if scope is None:
        scope = {}
    execfile(file, scope)
    return scope
