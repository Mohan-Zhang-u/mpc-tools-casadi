# Note: this file must be compatible with *both* Python 2 and 3.
"""Tests a .zip distribution file."""
import sys
import os
import tempfile
import zipfile
import runpy
import shutil
import matplotlib
matplotlib.use("agg")

def main(zipname):
    """Unzips the archive and runs runall.py."""
    maindir = os.getcwd()
    tmpdir = tempfile.mkdtemp(prefix="testzip-", dir=".")
    with zipfile.ZipFile(zipname) as z:
        z.extractall(tmpdir)
    rundir = os.path.join(tmpdir, os.listdir(tmpdir)[0])
    log = "runall-python%d.log" % sys.version_info.major
    os.chdir(rundir)
    sys.path.insert(0, ".")
    import mpctools
    sys.stderr.write("Importing mpctools from <%s>\n"
                     % os.path.relpath(mpctools.__file__, maindir))
    runpy.run_path("runall.py", {})
    os.chdir(maindir)
    shutil.copyfile(os.path.join(rundir, log), log)
    shutil.rmtree(tmpdir)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise RuntimeError("usage: testzip.py [zipname]")
    main(sys.argv[1])

