"""
Runs all the example files in mpc-tools-casadi showing only success or failure.
Error messages are copied to a log file.

By default, pdf plots are created, although this can be prevented by running
the script with an 'n' option, e.g.

    python runall.py -n

Note that you will be unable to see any plots in this case.

Most but not all solver output is hidden. To only display the status messages,
redirect stdout to a dummy file, e.g.,

    python runall.py 1> /dev/null

on Linux.
"""
import sys, traceback
import matplotlib.pyplot as plt
import mpctools.solvers, mpctools.plots
from mpctools.util import stdout_redirected, strcolor, dummy_context, runfile
import casadi

# Turn off output
mpctools.solvers.setMaxVerbosity(0)
mpctools.plots.SHOW_FIGURE_WINDOWS = False
choice = "y" if len(sys.argv) < 2 else sys.argv[1]
mpctools.plots.SHOWANDSAVE_DEFAULT_CHOICE = choice

logfile = open("runall-python%d.log" % sys.version_info.major, "w")
def printstatus(message, end="\n"):
    """Prints a status message to stderr. Syntax identical to print."""
    sys.stderr.write(message + end)

# List of files. We hard-code these so that we explicitly pick everything.
examplefiles = []
if casadi.has_conic("qpoases"):
    examplefiles.append("mpcexampleclosedloop.py")
if casadi.has_nlpsol("bonmin"):
    examplefiles += ["fishing.py", "cargears.py"]
examplefiles += [
    "airplane.py",
    "ballmaze.py",
    "cstr.py",
    "cstr_startup.py",
    "cstr_nmpc_nmhe.py",
    "collocationexample.py",
    "comparison_casadi.py",
    "comparison_mtc.py",
    "econmpc.py",    
    "example2-8.py",
    "icyhill.py",
    "mheexample.py",
    "mpcmodelcomparison.py",
    "nmheexample.py",
    "nmpcexample.py",
    "periodicmpcexample.py",
    "predatorprey.py",
    "softconstraints.py",
    "sstargexample.py",
    "template.py",
    "vdposcillator.py",
]
showoutput = set(["fishing.py", "cargears.py"])

# Now loop through everybody.
plt.ioff()
printstatus(__doc__)
abort = False
failures = []
for f in examplefiles:
    printstatus(strcolor("%s ... " % f, "blue"), end="")   
    try:
        context = dummy_context if f in showoutput else stdout_redirected
        with context():
            runfile(f)
            status = strcolor("succeeded", "green", bold=True)
    except KeyboardInterrupt:
        status =  "\n\n%s\n" % (strcolor("*** USER INTERRUPT ***", "yellow"),) 
        abort = True
    except:
        err = sys.exc_info()
        logfile.write("*** Error running <%s>:\n" % f)    
        traceback.print_exc(file=logfile)
        status = strcolor("FAILED", "red", bold=True)
        failures.append(f)
    finally:
        printstatus(status)
        plt.close("all")
        if abort:
            break
plt.ion()
logfile.close()

# Print final status.
if len(failures) > 0:
    printstatus(strcolor("%d scripts failed!" % len(failures), "red",
                         bold=True))
    for (i, f) in enumerate(failures):
        printstatus("    %d. %s" % (i + 1, f))
    printstatus(strcolor("See <%s> for more details." % logfile.name, "red",
                         bold=True))
else:
    printstatus(strcolor("All examples successful.", "green", bold=True))
        