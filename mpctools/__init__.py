"""
Import common functions here to make things a bit easier for the users.

Also check the version of Casadi and make sure it is new enough but not too
new (due to backward incompatibility in Casadi).
"""

# Version for mpctools.
__version__ = "2.4.2"
def get_changeset_id():
    """Returns the mercurial changeset id for this version of MPCTools."""
    from . import hg; changeset_id = hg.get_changeset_id() #CHANGESET_ID
    return changeset_id


# Check Casadi version to make sure it's up to date.
_MIN_CASADI_VERSION = "3.0"
_MAX_CASADI_VERSION = "4.0"
def _getVersion(vstring, N=3):
    """Returns a tuple with version number."""
    parts = vstring.split(".")
    try:
        major_version = int(parts[0])
        minor_version = int(parts[1])
    except (IndexError, ValueError):
        raise ValueError("Invalid version string '{}'. Must have major and "
                         "minor version.".format(vstring))
    return (major_version, minor_version)
import casadi
if _getVersion(casadi.__version__) < _getVersion(_MIN_CASADI_VERSION):
    raise ImportError("casadi version %s is too old (must be >=%s)"
        % (casadi.__version__, _MIN_CASADI_VERSION))
elif _getVersion(casadi.__version__) > _getVersion(_MAX_CASADI_VERSION):
    raise ImportError("casadi version %s is too new (must be <=%s)"
        % (casadi.__version__, _MAX_CASADI_VERSION))

# Add modules and some specific functions.
from . import tools
from . import plots
from . import util
from . import colloc
from . import solvers
from .tools import nmpc, nmhe, sstarg, getCasadiFunc, DiscreteSimulator
from .util import safevertcat as vcat
from .util import keyboard, mtimes, ekf
from .util import sum1 as sum
from .solvers import callSolver
