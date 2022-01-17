import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import scipy.io as sio
import sys

"""
Contains all of the plotting functions for mpc-tools-casadi.
"""

SHOW_FIGURE_WINDOWS = True # Global setting about whether to show figures.
SAVE_FIGURE_PDFS = True # Global setting to save figure pdfs.
SHOWANDSAVE_DEFAULT_CHOICE = "y"

# Override some things if user passes an --ioff flag.
if "--ioff" in sys.argv:
    plt.ioff()
    SHOW_FIGURE_WINDOWS = False

def mpcplot(x, u, t, xsp=None, fig=None, xinds=None, uinds=None, tightness=0.5,
            title=None, timefirst=True, legend=True, returnAxes=False,
            xnames=None, unames=None):
    """
    Makes a plot of the state and control trajectories for an mpc problem.
    
    Inputs x and u should be n by N+1 and p by N numpy arrays. xsp if provided
    should be the same size as x. t should be a numpy N+1 vector.
    
    If given, fig is the matplotlib figure handle to plot everything. If not
    given, a new figure is used.
    
    xinds and uinds are optional lists of indices to plot. If not given, all
    indices of x and u are plotted.
    
    Returns the figure handle used for plotting.
    """
    # Transpose data if time is the first dimension; we need it second.
    if timefirst:
        x = x.T
        u = u.T
        if xsp is not None:
            xsp = xsp.T
    
    # Process arguments.
    if xinds is None:
        xinds = np.arange(x.shape[0])
    if uinds is None:
        uinds = np.arange(u.shape[0])
    if fig is None:
        fig = plt.figure()
    if xsp is None:
        xlspec = "-k"
        ulspec = "-k"
        plotxsp = False
    else:
        xlspec = "-g"
        ulspec = "-b"
        plotxsp = True
    if xnames is None:
        xnames = ["State %d" % (i + 1) for i in xinds]
    if unames is None:
        unames = ["Control %d" % (i + 1) for i in uinds]
    
    # Figure out how many plots to make.
    numrows = max(len(xinds),len(uinds))
    if numrows == 0: # No plots to make.
        return None
    numcols = 2
    
    # u plots.
    u = np.hstack((u,u[:,-1:])) # Repeat last element for stairstep plot.
    uax = []
    for i in range(len(uinds)):
        uind = uinds[i]
        a = fig.add_subplot(numrows,numcols,numcols*(i+1))
        a.step(t,np.squeeze(u[uind,:]),ulspec,where="post")
        a.set_xlabel("Time")
        a.set_ylabel(unames[uind])
        zoomaxis(a,yscale=1.05)
        prettyaxesbox(a)
        prettyaxesbox(a,facecolor="white",front=False)
        uax.append(a)
    
    # x plots.
    xax = []    
    for i in range(len(xinds)):
        xind = xinds[i]
        a = fig.add_subplot(numrows,numcols,numcols*(i+1) - 1)
        a.plot(t,np.squeeze(x[xind,:]),xlspec,label="System")
        if plotxsp:
            a.plot(t,np.squeeze(xsp[xind,:]),"--r",label="Setpoint")
            if legend:            
                plt.legend(loc="best")
        a.set_xlabel("Time")
        a.set_ylabel(xnames[xind])
        zoomaxis(a,yscale=1.05)
        prettyaxesbox(a)
        prettyaxesbox(a,facecolor="white",front=False)
        xax.append(a)
    
    # Layout tightness.
    if tightness is not None:
        fig.tight_layout(pad=tightness)
    if title is not None:
        fig.canvas.set_window_title(title)       
    
    # Decide what to return.
    if returnAxes:
        retVal = {"x" : xax, "u" : uax, "fig" : fig}
    else:
        retVal = fig
    
    return retVal


def zoomaxis(axes=None,xscale=None,yscale=None):
    """
    Zooms the axes by a specified amounts (positive multipliers).
    
    If axes is None, plt.gca() is used.
    """
    # Grab default axes if necessary.
    if axes is None:
        axes = plt.gca()
    
    # Make sure input is valid.
    if (xscale is not None and xscale <= 0) or (yscale is not None and yscale <= 0):
        raise ValueError("Scale values must be strictly positive.")
    
    # Adjust axes limits.
    for (scale,getter,setter) in [(xscale,axes.get_xlim,axes.set_xlim), (yscale,axes.get_ylim,axes.set_ylim)]:
        if scale is not None:
            # Subtract one from each because of how we will calculate things.            
            scale -= 1
   
            # Get limits and change them.
            (minlim,maxlim) = getter()
            offset = .5*scale*(maxlim - minlim)
            setter(minlim - offset, maxlim + offset)

def prettyaxesbox(ax=None,linewidth=None,edgecolor="k",facecolor="none",front=True):
    """
    Replaces the box around the axes with a fancybox.

    This makes it an actual box and not just four lines.
    
    If linewidth is None, uses the initialized linewidth.
    """
    # First, figure out what axes object to use.
    if ax is None:
        ax = plt.gca()
    
    # Get linewidth if necessary.
    if linewidth is None:
        linewidth = 1
    
    # Now we're going to make the box around the axes look better.
    ap = ax.patch
    zorders = [c.get_zorder() for c in ax.get_children()]    
    if front:
        z = max(zorders) + 1
    else:
        z = min(zorders) - 1
    prettybox = FancyBboxPatch(ap.get_xy(),ap.get_width(),ap.get_height(),
                               boxstyle="square,pad=0.",ec=edgecolor,
                               fc=facecolor,transform=ap.get_transform(),
                               lw=linewidth,zorder=z)

    # Make current box invisible and make our better one.    
    ap.set_edgecolor("none")
    ax.set_frame_on(False)
    ax.add_patch(prettybox)
    prettybox.set_clip_on(False)
    plt.draw()
    
    return prettybox            


def showandsave(fig,filename="fig.pdf",choice=None,**kwargs):
    """
    Shows a figure in the interactive window and prompts user to save.
    """
    if choice is None:
        choice = SHOWANDSAVE_DEFAULT_CHOICE
    if SHOW_FIGURE_WINDOWS and plt.isinteractive():    
        fig.show()
    if SAVE_FIGURE_PDFS:
        if choice == "prompt":
            choice = input("Save figure as '%s' [y/n]? " % (filename,))
        if choice == "y":
            fig.savefig(filename,**kwargs)


# =============================================
# Some functions for saving/reading .mat files.
# =============================================

# First grab readmat for convenience.
savemat = sio.savemat

# The following three functions from http://stackoverflow.com/questions/7008608
def loadmat(filename):
    """
    Loads a mat file with sensible  behavior for nested scalar structs.
    
    This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(d):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in d:
        if isinstance(d[key], sio.matlab.mio5_params.mat_struct):
            d[key] = _todict(d[key])
    return d        


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries.
    """
    d = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            d[strg] = _todict(elem)
        else:
            d[strg] = elem
    return d


# ========================
# Some printing functions.
# ========================

def numberformat(n,nsig=3,minval=-np.inf,mathmode=True):
    """
    Formats a number as a string to the specified number of sig figs.
    """
    
    # Check minimum value.
    isSmall =  n < minval
    if isSmall:
        n = minval
    
    s = ("%%.%dg" % nsig) % (n,)
    # Add trailing period for floats.        
    if round(n) != n and s.find(".") == -1 and n < 10**(nsig+1):
        s += "."
    
    # Check if there's scientific notation.
    e = s.find("e")
    if e >= 0:
        head = s[0:e]
    else:
        head = s    
    
    # Make sure we show enough sig figs.    
    if head.find(".") >= 0 and len(head) <= nsig:
        addstr = "0"*(nsig - len(head) + 1)
    else:
        addstr = ""
    
    if e >= 0: # Need to handle scientific notation.
        s = s.replace("e",addstr + r" \times 10^{")
        for [f,r] in [["{+","{"],["{0","{"],["{-0","{-"]]:
            for i in range(5):
                s = s.replace(f,r)
        if s.endswith("."):
            s = s[:-1]
        s = s + "}"
    else:
        s += addstr
    if isSmall:
        s = "<" + s
    if mathmode:
        s = "$" + s + "$"
    return s

    
def printmatrix(A,before="     ",nsig=3,latex=True):
    """
    Prints a matrix A to be pasted into LaTeX.
    """
    if latex:
        print(r"\begin{pmatrix}")
    for i in range(A.shape[0]):
        print((before + " & ".join([numberformat(a,nsig) for a in 
                                    np.array(A)[i,:].tolist()]) + r" \\")) 
    if latex:
        print(r"\end{pmatrix}")
            