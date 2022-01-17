# Generates dictionaries of all solver-specific options.
import mpctools as mpc

# First, generate a dummy problem.
f = mpc.getCasadiFunc(lambda x, u : x, [1, 1], ["x", "u"])
l = mpc.getCasadiFunc(lambda x, u : x**2 + u**2, [1, 1], ["x", "u"])
N = {"x" : 1, "u" : 1, "t" : 1}
solver = mpc.nmpc(f, l, N)
solver.isQP = True

# Now loop through all solvers and get options.
allsolvers = mpc.solvers.listAvailableSolvers(asstring=False)
alloptions = {}
errors = {}
for s in allsolvers["QP"] + allsolvers["NLP"]:
    print("Getting options for %s." % s)
    solver.solver = s
    try:
        alloptions[s] = solver.getSolverOptions(display=False)    
    except ValueError as err:
        if err.message == "No table found!":
            print("  *** No documentation table for %s!" % s)
        else:
            errors[s] = err.message
            print("  *** Error for %s!" % s)
    except RuntimeError as err:
        errors[s] = err.message
        print("  *** Error for %s!" % s)
