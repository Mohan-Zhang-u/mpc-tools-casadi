# mpcsim is a graphical user interface for the mpc-tools-casadi package
#
# Tom Badgwell April 2017
# Michael Risbeck February 2016
#
# to do list:
#
# - implement file open/closed options (quit the current window)
# - disable the menu options depending on mode
# - implement a menu for the unmeasured disturbances - tun factors
# - implement plots for the unmeasured disturbance variables
# - implement option to show process diagram
# - refactor Trndplt.plotvals

import sys
import collections
import copy
if sys.version_info.major == 2:
    import Tkinter as tk
    import tkMessageBox as tkmsg
    from tkFileDialog import askopenfilename
    from tkSimpleDialog import askfloat, askinteger # analysis:ignore
else:
    import tkinter as tk
    import tkinter.messagebox as tkmsg
    from tkinter.filedialog import askopenfilename
    from tkinter.simpledialog import askfloat, askinteger # analysis:ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
from . import util

def makegui(simcon, **kwargs):
    """Build Tk window and plots; then start simulation."""
    # create main window
    root = tk.Tk()
    root.title('MPC-Sim')
    simcon.root = root

    # create the menus
    menubar = makemenus(root, simcon)

    # create the run panel on the menubar
    rpanel = RunPanel(menubar)

    # create the control panel on the menubar
    cpanel = ConPanel(menubar)
    
    # Add the step button and the reset button.
    stepbutton = tk.Button(menubar)
    stepbutton.configure(text="Single\nStep")
    stepbutton.pack(side=tk.LEFT)
    
    #TODO: finalize reset button appearence.
    resetbutton = tk.Button(menubar)
    resetbutton.configure(text="Reset\nSimulation")
    resetbutton.pack(side=tk.LEFT)
    
    # add the simulation name box
    makename(menubar, simcon.simname)    
    
    # fill in remaining space on the menubar
    fillspace(menubar)

    # create the trend plots
    Trndplt(root, simcon, rpanel, cpanel, stepbutton, resetbutton, **kwargs)

    # start the main loop
    root.mainloop()

def notdone():
    """Tk error message for features not yet available."""
    tkmsg.showerror('Not implemented', 'Not yet available')

def menu_add_command(menu, var, desc):
    """Wrapper to add a command to dropdown menus."""    
    menu.add_command(label='Set ' + desc, command=lambda : setvalue(var, desc),
                     underline=0)

def openfile(simcon):
    """Open another python file and run it."""
    f = askopenfilename()
    simcon.root.destroy()
    util.execfile(f)

def askbool(description, message, **kwargs):
    """
    Asks a yes/no question and returns a bool.
    
    kwargs ignored for compatibility.
    """
    return tkmsg.askyesnocancel(description, message)

# Generate dictionary of available options for setvalue.
# TODO: refactor so that this list is generated in the simulation file instead
#       of having a single global list.
def _get_setvalue_data():
    """Returns a dictionary of available setvalue options."""
    SetvalueTuple = collections.namedtuple("SetValue", ["field", "xmin",
                                                        "xmax", "chflag",
                                                        "askfunc"])
    def svo(field, xmin=None, xmax=None, chflag=False, askfunc=askfloat):
        """Wrapper for SetvalueTuple with default arguments."""
        return SetvalueTuple(field, xmin, xmax, chflag, askfunc)
    setvalue_options = util.ReadOnlyDict({
        "Value" : svo("value"),
        "SS Target" : svo("sstarg", chflag=True), 
        "SS R Weight" : svo("ssrval", xmin=0, chflag=True),
        "SS Q Weight" : svo("ssqval", xmin=0, chflag=True),
        "Target" : svo("target"),
        "Setpoint" : svo("setpoint"),
        "Q Weight" : svo("qvalue", xmin=0, chflag=True),
        "R Weight" : svo("rvalue", xmin=0, chflag=True),
        "S Weight" : svo("svalue", xmin=0, chflag=True),
        "Max Limit" : svo("maxlim", xmin="minlim", chflag=True),
        "Min Limit" : svo("minlim", xmax="maxlim", chflag=True),
        "ROC Limit" : svo("roclim", xmin=0, chflag=True),
        "Plot High Limit" : svo("pltmax", xmin="pltmin"),
        "Plot Low Limit" : svo("pltmin", xmax="pltmax"),
        "Process Noise" : svo("noise", xmin=0),
        "Model Noise" : svo("mnoise", xmin=1e-10, chflag=True),
        "Disturbance Model Noise" : svo("dnoise", xmin=1e-10, chflag=True),
        "Process Step Dist." : svo("dist"),
        "Refresh Int." : svo("refint", xmin=10, xmax=10000),
        "Noise Factor" : svo("value", xmin=0),
        "Open-Loop Predictions" : svo("value", askfunc=askbool),
        "A Value" : svo("value", xmin=-10, xmax=10, chflag=True),
        "Gain Mismatch Factor" : svo("gfac", xmin=0, chflag=True),
        "Disturbance Model" : svo("value", xmin=1, xmax=5, chflag=True),
        "Control Gain" : svo("value", xmin=-10, xmax=10, chflag=True),
        "Reset Time" : svo("value", xmin=0, xmax=10000, chflag=True),
        "Derivative Time" : svo("value", xmin=0, xmax=100, chflag=True),
        "Heat Of Reaction" : svo("value", xmin=-1e-6, xmax=-1e-6, chflag=True),
        "LB Slack Weight" : svo("lbslack", xmin=0, chflag=True),
        "UB Slack Weight" : svo("ubslack", xmin=0, chflag=True),
        "Fuel increment" : svo("value", xmin=0, xmax=1),
        "Linear Model" : svo("value", askfunc=askbool, chflag=True),
        "Nonlinear MPC" : svo("value", askfunc=askbool),
        "Nonlinear MHE" : svo("value", askfunc=askbool),
    })
        
    # Also create a reverse mapping.
    setvalue_names = {sv.field : name for (name, sv) in
                      setvalue_options.items()}
    setvalue_names["value"] = "Value"
    setvalue_names = util.ReadOnlyDict(setvalue_names)
    
    return (setvalue_options, setvalue_names)
(_SETVALUE_OPTIONS, _SETVALUE_NAMES) = _get_setvalue_data()

def setvalue(var, desc):
    """Sets a specific variable field using a dialog box."""
    vinfo = _SETVALUE_OPTIONS.get(desc, None)    
    if vinfo is not None:
        value = getattr(var, vinfo.field)
        askfunc = askfloat if vinfo.askfunc is None else vinfo.askfunc
        if askfunc is askbool:
            if value:
                entrytext = "Currently using %s. Keep using %s?" % (desc, desc)
            else:
                entrytext = ("Currently not using %s. Start using %s?"
                             % (desc, desc))
        else:
            entrytext = "%s currently %g, enter new value" % (desc, value)
        kwargs = {}
        for (k, lim) in [("minvalue", vinfo.xmin), ("maxvalue", vinfo.xmax)]:
            if lim is not None:
                if isinstance(lim, str):
                    val = getattr(var, lim)
                else:
                    val = lim
                kwargs[k] = val
        value = askfunc(var.name, entrytext, **kwargs)
        if value is not None:
            setattr(var, vinfo.field, value)
            if vinfo.chflag:
                var.chflag = 1
    else:
        notdone()

def showhelp():
    """Tk dialog box with simulation information."""
    tkmsg.showinfo('About MPC-Sim', 'MPC-Sim is a GUI for the '
                   'mpc-tools-casadi package (Tom Badgwell and Michael Risbeck)')

def makemenus(win, simcon):

    mvlist = simcon.mvlist
    dvlist = simcon.dvlist
    cvlist = simcon.cvlist
    xvlist = simcon.xvlist
    oplist = simcon.oplist

    menubar = tk.Frame(win)
    menubar.config(bd=2, relief=tk.GROOVE)
    menubar.pack(side=tk.TOP, fill=tk.X)

    # build the file menu

    fbutton = tk.Menubutton(menubar, text='File', underline=0)
    fbutton.pack(side=tk.LEFT)
    filemenu = tk.Menu(fbutton, tearoff=0)
#    filemenu.add_command(label='Open',  command=lambda: openfile(simcon),  underline=0)
#    filemenu.add_command(label='Close', command=notdone,  underline=0)
    filemenu.add_command(label='Exit',  command=win.quit, underline=0)
    fbutton.config(menu=filemenu)

    # Helper function.
    def addmenus(variable, menu):
        """Loops through keyandname, adding to menu if any keys are found."""
        for key in variable.menu:
            name = variable.menu_names.get(key, None)
            if name is None:
                name = _SETVALUE_NAMES.get(key, None)
            if name is None:
                name = "<UNKNOWN NAME: {}>".format(key)
            menu_add_command(menu, variable, name)

    # build the MV menu

    mbutton = tk.Menubutton(menubar, text='MVs', underline=0)
    mbutton.pack(side=tk.LEFT)
    mvsmenu = tk.Menu(mbutton, tearoff=0)
    mbutton.config(menu=mvsmenu)

    for mv in mvlist:
        mvmenu = tk.Menu(mvsmenu, tearoff=False)
        addmenus(mv, mvmenu)
        mvsmenu.add_cascade(label=mv.name, menu=mvmenu, underline=0)

    # build the DV menu if there are DVs

    if len(dvlist) > 0:
        dbutton = tk.Menubutton(menubar, text='DVs', underline=0)
        dbutton.pack(side=tk.LEFT)
        dvsmenu = tk.Menu(dbutton, tearoff=0)
        dbutton.config(menu=dvsmenu)

        for dv in dvlist:
            dvmenu = tk.Menu(dvsmenu, tearoff=False)
            addmenus(dv, dvmenu) 
            dvsmenu.add_cascade(label=dv.name, menu=dvmenu, underline = 0)

    # build the XV menu if there are XVs
    if len(xvlist) > 0:
        xbutton = tk.Menubutton(menubar, text='XVs', underline=0)
        xbutton.pack(side=tk.LEFT)
        xvsmenu = tk.Menu(xbutton, tearoff=0)
        xbutton.config(menu=xvsmenu)

        for xv in xvlist:
            xvmenu = tk.Menu(xvsmenu, tearoff=False)
            addmenus(xv, xvmenu)
            xvsmenu.add_cascade(label=xv.name, menu=xvmenu, underline=0)

    # build the CV menu
    cbutton = tk.Menubutton(menubar, text='CVs', underline=0)
    cbutton.pack(side=tk.LEFT)
    cvsmenu = tk.Menu(cbutton, tearoff=0)
    cbutton.config(menu=cvsmenu)

    for cv in cvlist:
        cvmenu = tk.Menu(cvsmenu, tearoff=False)
        addmenus(cv, cvmenu)
        cvsmenu.add_cascade(label=cv.name, menu=cvmenu, underline = 0)

    # build the options menu

    obutton = tk.Menubutton(menubar, text='Options', underline=0)
    obutton.pack(side=tk.LEFT)
    opsmenu = tk.Menu(obutton, tearoff=0)
    obutton.config(menu=opsmenu)

    for op in oplist:
        opmenu = tk.Menu(opsmenu, tearoff=False)
        menu_add_command(opmenu, op, op.desc) 
        opsmenu.add_cascade(label=op.name, menu=opmenu, underline = 0)

    # build the help menu
    hbutton = tk.Menubutton(menubar, text='Help', underline=0)
    hbutton.pack(side=tk.LEFT)
    helpmenu = tk.Menu(hbutton, tearoff=0)
    helpmenu.add_command(label='About', command=showhelp,  underline=0)
    hbutton.config(menu=helpmenu)

    return menubar

class Trndplt(object):
    """Strip chart using matplotlib."""
    def __init__(self, parent, simcon, runpause, opnclsd, stepbutton,
                 resetbutton, plotspacing=None):
        # store inputs
        self.simcon = simcon
        self.copysimcon()        
        self.runpause = runpause
        self.opnclsd  = opnclsd
        self.stepbutton = stepbutton
        self.resetbutton = resetbutton
        self.k = 0
        self.parent = parent # Save parent to reference later.
        self.pendingsim = None

        # determine the subplot dimensions
        self.nmvs      = len(self.mvlist)
        self.ndvs      = len(self.dvlist)
        self.ncvs      = len(self.cvlist)
        self.nxvs      = len(self.xvlist)
        self.ninputs   = self.nmvs + self.ndvs
        self.noutputs  = self.ncvs
        self.nrows     = max(self.ninputs, self.nxvs, self.noutputs)
        self.ncols = 3 if self.nxvs > 0 else 2
        self.submat = str(self.nrows) + str(self.ncols)
        if plotspacing is None:
            plotspacing = {}
        self.plotspacing = plotspacing

        # build the figure
        self.fig = plt.Figure()
        self.resetfigure()
        
        # attach figure to parent.
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side=tk.TOP, expand=tk.YES,
                                         fill=tk.BOTH)
        
        # Configure buttons.
        self.stepbutton.configure(command=self.simulate)
        self.stepbutton.focus_force()

        self.resetbutton.configure(command=self.reset)
        
        self.runpause.configure(lcommand=self.pausesim, rcommand=self.playsim)

    def playsim(self, repeat=True):
        """Loop to automatically run simulation."""
        if self.runpause.status.get() == 1:
            self.simulate()
        if repeat:
            self.pendingsim = self.parent.after(int(self.refint),
                                                self.playsim)
    def pausesim(self):
        """Cancels any pending simulation step."""
        if self.pendingsim is not None:
            self.parent.after_cancel(self.pendingsim)
    
    def simulate(self, i=0):
        """Run one simulation step."""
        # Run step.
        self.simcon.runsim(self.k, self.simcon, self.opnclsd)

        # update the trends
        self.pltvals()

        # increment the iteration count
        self.k += 1

    def copysimcon(self):
        """Copies attributes from self.simcon to self."""
        simcon = self.simcon        
        self.N = simcon.N
        self.deltat = simcon.deltat
        self.mvlist = simcon.mvlist
        self.dvlist = simcon.dvlist
        self.cvlist = simcon.cvlist
        self.xvlist = simcon.xvlist
        self.oplist = simcon.oplist
        self.refint = simcon.refint
        self.runsim = simcon.runsim
        
    def reset(self):
        """Resets simulation to initial values."""
        self.pausesim()
        self.k = 0
        self.simcon.usedefaults()
        self.copysimcon()
        self.resetfigure(clear=True)
        if self.runpause.status.get() == 1:
            self.playsim() # Restart automatic loop.

    def resetfigure(self, clear=False):
        """Removes all axes from the figure and rebuilds them."""
        self.fig.clear()
        self.initlines()
        self.axes = makeaxes(self.fig, self.nrows, self.ncols, sharex=True)
        self.mvaxes = self.addaxes(self.mvlist, col=1, xticks=False)
        self.dvaxes = self.addaxes(self.dvlist, startrow=self.nmvs + 1, col=1)
        self.xvaxes = self.addaxes(self.xvlist, col=2)
        self.cvaxes = self.addaxes(self.cvlist, col=self.ncols)
        self.initializeaxes(clear=clear)
        self.fig.subplots_adjust(**self.plotspacing)

    def initlines(self):
        """Sets all line attributes to empty lists."""
        #TODO: replace with a dictionary or class
        self.mvlines   = []
        self.mvmxlines = []
        self.mvmnlines = []

        self.dvlines   = []

        self.cvlines   = []
        self.cveslines = []
        self.cvsplines = []
        self.cvmxlines = []
        self.cvmnlines = []

        self.xvlines   = []
        self.xveslines = []
        self.xvsplines = []
        self.xvmxlines = []
        self.xvmnlines = []

        self.fomvlines  = []
        self.fcmvlines  = []
        self.fmvmxlines = []
        self.fmvmnlines = []

        self.fdvlines   = []

        self.focvlines  = []
        self.fccvlines  = []
        self.fcvsplines = []
        self.fcvmxlines = []
        self.fcvmnlines = []

        self.foxvlines  = []
        self.fcxvlines  = []
        self.fxvsplines = []
        self.fxvmxlines = []
        self.fxvmnlines = []

    def addaxes(self, varlist, col=1, startrow=1, xticks=True):
        """Adds an axis in the given column for each variable."""
        # Label and scale axes.
        axes = []
        for (i, var) in enumerate(varlist):       
            ax = self.axes[startrow - 1 + i, col - 1]
            ax.set_visible(True)
            ax.set_ylabel("%s %s" % (var.name, var.units))
            ax.set_title(var.desc)
            ax.set_ylim([var.pltmin, var.pltmax])
            ax.margins(x=0)
            axes.append(ax)
            
        # Add xticks for bottom plot.
        if len(axes) > 0 and xticks:
            for label in ax.get_xticklabels():
                label.set_visible(True)
        return axes

    def initplot(self, ax, var, vallines=None, maxlines=None, minlines=None,
                 forecast=False, openloop=False, valcolor="b", vlinex=None,
                 valfield="value"):
        """Initialize given axis with variable information and append lines."""
        # Get x vector for everything.
        if forecast:
            xvec = np.arange(0, var.Nf)
        else:
            xvec = np.arange(-self.N, 1)

        # Plot value of variable.
        if vallines is not None:
            yvec = getattr(var, valfield)*np.ones(xvec.shape)
            linestyle = "--" if openloop else "-"
            [line] = ax.plot(xvec, yvec, color=valcolor,
                             linestyle=linestyle)
            vallines.append(line)

        # Plot minimum and maximum values.
        if maxlines is not None:
            yvec = var.maxlim*np.ones(xvec.shape)
            [line] = ax.plot(xvec, yvec, color="r", linestyle="--")
            maxlines.append(line)
            
        if minlines is not None:
            yvec = var.minlim*np.ones(xvec.shape)
            [line] = ax.plot(xvec, yvec, color="r", linestyle="--")
            minlines.append(line)

        # Plot vertical line.
        if vlinex is not None:
            ax.axvline(vlinex, color="r")

    def initializeaxes(self, clear=False):
        """Plots initial lines on all of the subplots."""
        # Add grids after clearing axes.
        for varax in [self.mvaxes, self.dvaxes, self.xvaxes, self.cvaxes]:
            for ax in varax:
                if clear:
                    for line in ax.get_lines():
                        line.remove()
                ax.grid(True, "major")

        # Plot initial values
        for (mv, mvaxis) in zip(self.mvlist, self.mvaxes):
            # Value history with limits and vertical line at t = 0.
            self.initplot(mvaxis, mv, vallines=self.mvlines,
                          maxlines=self.mvmxlines, minlines=self.mvmnlines,
                          vlinex=0, valcolor="k")

            # Closed-loop forecast with limits.
            self.initplot(mvaxis, mv, vallines=self.fcmvlines,
                          maxlines=self.fmvmxlines, minlines=self.fmvmnlines,
                          forecast=True)
                          
            # Open-loop forecast.
            self.initplot(mvaxis, mv, vallines=self.fomvlines, forecast=True,
                          openloop=True)

        for (dv, dvaxis) in zip(self.dvlist, self.dvaxes):
            # Value history with vertical line at t = 0.
            self.initplot(dvaxis, dv, vallines=self.dvlines, vlinex=0,
                          valcolor="k")

            # Value forecast.
            self.initplot(dvaxis, dv, vallines=self.fdvlines, forecast=True)

        for (xv, xvaxis) in zip(self.xvlist, self.xvaxes):

            # Value history for states with vertical line at t = 0.
            self.initplot(xvaxis, xv, vallines=self.xvlines, vlinex=0,
                          valcolor="k")
        
            # Estimated value history for states.
            self.initplot(xvaxis, xv, vallines=self.xveslines, valfield="est")

            # Closed-loop forecast.
            self.initplot(xvaxis, xv, vallines=self.fcxvlines, forecast=True)

            # Open-loop forecast.
            self.initplot(xvaxis, xv, vallines=self.foxvlines, forecast=True,
                          openloop=True)            
        
        for (cv, cvaxis) in zip(self.cvlist, self.cvaxes):
            # Value history with limits and vertical line at t = 0.
            self.initplot(cvaxis, cv, vallines=self.cvlines,
                          maxlines=self.cvmxlines, minlines=self.cvmnlines,
                          vlinex=0, valcolor="k")

            # Closed-loop forecast with limits.
            self.initplot(cvaxis, cv, vallines=self.fccvlines,
                          maxlines=self.fcvmxlines, minlines=self.fcvmnlines,
                          forecast=True)

            # Open-loop forecast.
            self.initplot(cvaxis, cv, vallines=self.focvlines, forecast=True,
                          openloop=True)

            # Setpoint past and history.
            self.initplot(cvaxis, cv, vallines=self.cvsplines, valcolor="g",
                          valfield="setpoint")
            self.initplot(cvaxis, cv, vallines=self.fcvsplines, valcolor="g",
                          valfield="setpoint", forecast=True)

            # Estimated history.
            self.initplot(cvaxis, cv, vallines=self.cveslines, valfield="est")
            
        # Make sure everything is drawn.
        if self.fig.canvas is not None:
            self.fig.canvas.draw()

    def pltvals(self, updatefig=True):

        # update mv trends

        for mv in self.mvlist:

            mvndx = self.mvlist.index(mv)
            mvaxis = self.mvaxes[mvndx]

            mvaxis.set_ylim([mv.pltmin, mv.pltmax])
#            mvaxis.plot((0,0),(mv.pltmin,mv.pltmax), 'r')

            mvline = self.mvlines[mvndx]
            ydata  = mvline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.N] = mv.value
            mvline.set_ydata(ydata)

            mvmxline = self.mvmxlines[mvndx]
            ydata  = mvmxline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.N] = mv.maxlim
            mvmxline.set_ydata(ydata)

            mvmnline = self.mvmnlines[mvndx]
            ydata  = mvmnline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.N] = mv.minlim
            mvmnline.set_ydata(ydata)

            fomvline = self.fomvlines[mvndx]
            fomvline.set_ydata(mv.olpred)

            fcmvline = self.fcmvlines[mvndx]
            fcmvline.set_ydata(mv.clpred)

            fmvmxline  = self.fmvmxlines[mvndx]
            yvec       = mv.maxlim*np.ones((mv.Nf,1))
            fmvmxline.set_ydata(yvec)

            fmvmnline  = self.fmvmnlines[mvndx]
            yvec       = mv.minlim*np.ones((mv.Nf,1))
            fmvmnline.set_ydata(yvec)

        # update dv trends

        for dv in self.dvlist:

            dvndx = self.dvlist.index(dv)
            dvaxis = self.dvaxes[dvndx]

            dvaxis.set_ylim([dv.pltmin, dv.pltmax])
#            dvaxis.plot((0,0),(dv.pltmin,dv.pltmax), 'r')

            dvline = self.dvlines[dvndx]
            ydata  = dvline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.N] = dv.value
            dvline.set_ydata(ydata)

            fdvline = self.fdvlines[dvndx]
            yvec    = dv.value*np.ones((dv.Nf,1))
            fdvline.set_ydata(yvec)

        # update xv trends

        for xv in self.xvlist:

            xvndx = self.xvlist.index(xv)
            xvaxis = self.xvaxes[xvndx]

            xvaxis.set_ylim([xv.pltmin, xv.pltmax])
#            xvaxis.plot((0,0),(xv.pltmin,xv.pltmax), 'r')

            xvline = self.xvlines[xvndx]
            ydata  = xvline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.N] = xv.value
            xvline.set_ydata(ydata)

            xvesline = self.xveslines[xvndx]
            ydata  = xvesline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.N] = xv.est
            xvesline.set_ydata(ydata)

#            xvspline = self.xvsplines[xvndx]
#            ydata  = xvspline.get_ydata()
#            ydata  = np.roll(ydata,-1,0)
#            ydata[self.Nm1] = xv.setpoint
#            xvspline.set_ydata(ydata)

#            xvmxline = self.xvmxlines[xvndx]
#            ydata  = xvmxline.get_ydata()
#            ydata  = np.roll(ydata,-1,0)
#            ydata[self.Nm1] = xv.maxlim
#            xvmxline.set_ydata(ydata)

#            xvmnline = self.xvmnlines[xvndx]
#            ydata  = xvmnline.get_ydata()
#            ydata  = np.roll(ydata,-1,0)
#            ydata[self.Nm1] = xv.minlim
#            xvmnline.set_ydata(ydata)

            foxvline = self.foxvlines[xvndx]
            foxvline.set_ydata(xv.olpred)

            fcxvline = self.fcxvlines[xvndx]
            fcxvline.set_ydata(xv.clpred)

#            fxvspline  = self.fxvsplines[xvndx]
#            yvec       = xv.setpoint*np.ones((xv.Nf,1))
#            fxvspline.set_ydata(yvec)

#            fxvmxline  = self.fxvmxlines[xvndx]
#            yvec       = xv.maxlim*np.ones((xv.Nf,1))
#            fxvmxline.set_ydata(yvec)

#            fxvmnline  = self.fxvmnlines[xvndx]
#            yvec       = xv.minlim*np.ones((xv.Nf,1))
#            fxvmnline.set_ydata(yvec)

        # update cv trends

        for cv in self.cvlist:

            cvndx = self.cvlist.index(cv)
            cvaxis = self.cvaxes[cvndx]

            cvaxis.set_ylim([cv.pltmin, cv.pltmax])
#            cvaxis.plot((0,0),(cv.pltmin,cv.pltmax), 'r')

            cvline = self.cvlines[cvndx]
            ydata  = cvline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.N] = cv.value
            cvline.set_ydata(ydata)

            cvesline = self.cveslines[cvndx]
            ydata  = cvesline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.N] = cv.est
            cvesline.set_ydata(ydata)

            cvspline = self.cvsplines[cvndx]
            ydata  = cvspline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.N] = cv.setpoint
            cvspline.set_ydata(ydata)

            cvmxline = self.cvmxlines[cvndx]
            ydata  = cvmxline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.N] = cv.maxlim
            cvmxline.set_ydata(ydata)

            cvmnline = self.cvmnlines[cvndx]
            ydata  = cvmnline.get_ydata()
            ydata  = np.roll(ydata,-1,0)
            ydata[self.N] = cv.minlim
            cvmnline.set_ydata(ydata)

            focvline = self.focvlines[cvndx]
            focvline.set_ydata(cv.olpred)

            fccvline = self.fccvlines[cvndx]
            fccvline.set_ydata(cv.clpred)

            fcvspline  = self.fcvsplines[cvndx]
            yvec       = cv.setpoint*np.ones((cv.Nf,1))
            fcvspline.set_ydata(yvec)

            fcvmxline  = self.fcvmxlines[cvndx]
            yvec       = cv.maxlim*np.ones((cv.Nf,1))
            fcvmxline.set_ydata(yvec)

            fcvmnline  = self.fcvmnlines[cvndx]
            yvec       = cv.minlim*np.ones((cv.Nf,1))
            fcvmnline.set_ydata(yvec)
            
        # Update figure.
        if updatefig:
            self.fig.canvas.draw()

class RadioPanel(object):
    def __init__(self, parent, title="", lbutton="", rbutton=""):
        self.status = tk.IntVar()
        self.frame = tk.Frame(parent)
        self.frame.config(bd=2, relief=tk.GROOVE)
        self.frame.pack(side=tk.LEFT)
        msg = tk.Label(self.frame, text=title)
        msg.pack(side=tk.TOP)
        leftb = tk.Radiobutton(self.frame, text=lbutton,
                               command=self.leftcommand, variable=self.status,
                               value=0)
        leftb.pack(side=tk.LEFT)
        rightb = tk.Radiobutton(self.frame, text=rbutton,
                                command=self.rightcommand,
                                variable=self.status, value=1)
        rightb.pack(side=tk.LEFT)
        self.status.set(0)
        self.frame.config(bg='red')
        self.__rcommand = None
        self.__lcommand = None

    def configure(self, lcommand=None, rcommand=None):
        """Set commands for left and right buttons."""
        self.__lcommand = lcommand
        self.__rcommand = rcommand

    def leftcommand(self):
        self.setbg()
        if self.__lcommand is not None:
            self.__lcommand()

    def rightcommand(self):
        self.setbg()
        if self.__rcommand is not None:
            self.__rcommand()

    def setbg(self):
        if self.status.get() == 0:
            self.frame.config(bg='red')
        if self.status.get() == 1:
            self.frame.config(bg='green')

class RunPanel(RadioPanel):
    def __init__(self, parent):
        super(RunPanel, self).__init__(parent, title="Sim Status",
                                       lbutton="Pause", rbutton="Run")    
    @property
    def rframe(self):
        return self.frame

class ConPanel(RadioPanel):
    def __init__(self, parent):
        super(ConPanel, self).__init__(parent, title="Loop Status",
                                       lbutton="Open", rbutton="Closed")    
    @property
    def rframe(self):
        return self.frame

def makename(parent, simname):
    """Adds name frame to top of UI."""
    nameframe = tk.Frame(parent)
    nameframe.config(bd=2, relief=tk.GROOVE)
    nameframe.pack(side=tk.LEFT)
    padname = ' ' + simname + ' '
    namebox = tk.Label(nameframe, text=padname, font=('ariel', 12, 'bold'))
    namebox.pack(side=tk.LEFT)

def fillspace(parent):
    """Fills remaining horizontal space at top of UI."""
    fillframe = tk.Frame(parent)
    fillframe.config(bg='blue')
    fillframe.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)

class VarList(list):
    """Class for list of variable objects."""
    def __init__(self, iterable):
        super(VarList, self).__init__(iterable)
    
    def valbyname(self, name):
        """Returns the list element whose name is name (None if not found)."""
        i = self.indexbyname(name)
        if i is not None:
            i = self[i]
        return i
    
    def indexbyname(self, name):
        """Returns the list index whose name is name (None if not found)."""
        for (i, val) in enumerate(self):
            if val.name == name:
                break
        else:
            i = None
        return i
    
    def asvec(self, field="value"):
        """Returns Numpy vector with item.field for each item."""
        return np.array([getattr(item, field) for item in self])
        
    def vecassign(self, vec, field="value", index=None):
        """
        Assigns self[i].field = vec[i] for i in range(len(self)).
            
        If index is not None, does self[i].field[index] = vec[i].
        """
        vec = np.array(vec)
        if vec.shape != (len(self),):
            if vec.shape == () and len(self) == 1:
                vec = np.array([vec])
            else:
                raise ValueError("Incorrect size: vec is %r, must be (%d,)."
                                  % (vec.shape, len(self)))
        for (i, item) in enumerate(self):
            if index is None:
                setattr(item, field, vec[i])
            else:
                subitem = getattr(item, field)
                subitem[index] = vec[i]

class Updatable(object):
    """Object with _update method to to update attributes."""
    def _update(self, newobj, attributes=None):
        """
        Updates self with attributes from newobj.

        By default, attributes is all attributes of newobj that don't start
        with an underscore. Pass an iterable to only use certain attributes.        
        
        newobj must be an instance of type(self).        
        """
        if not isinstance(newobj, type(self)):
            raise TypeError("Incompatible type for newobj.")
        if attributes is None:
            attributes = [x for x in dir(newobj) if not x.startswith("_")]
        for a in attributes:
            setattr(self, a, getattr(newobj, a))

class Option(Updatable):
    """Struct for dropdown menu options."""
    def __init__(self, name=' ', desc=' ', value=0.0):
        self.name   = name
        self.desc   = desc
        self.value  = value
        self.chflag = 0

# TODO: make another subclass for MVobj, DVobj, and CVobj.
class MVobj(Updatable):
    """Structure for manipulated variables."""
    def __init__(self, name=' ', desc=' ', units= ' ',
                 value=0.0, sstarg=0.0, ssrval=0.01, target=0.0, rvalue=0.001,
                 svalue=0.001, maxlim=1.0e10, minlim=-1.0e10, roclim=1.0e10,
                 pltmax=100.0, pltmin=0.0, noise=0.0, dist=0.0, Nf=0,
                 menu=("value","sstarg","ssrval","target","rvalue","svalue",
                       "maxlim","minlim","roclim","pltmax","pltmin","noise",
                       "dist")):
        self.name   = name
        self.desc   = desc
        self.units  = units
        self.value  = value
        self.est    = value
        self.sstarg = sstarg
        self.ssrval = ssrval
        self.target = target
        self.rvalue = rvalue
        self.svalue = svalue
        self.maxlim = maxlim
        self.minlim = minlim
        self.roclim = roclim
        self.pltmax = pltmax
        self.pltmin = pltmin
        self.noise  = noise
        self.dist   = dist
        self.ref    = value
        self.chflag = 0
        self.Nf     = Nf
        self.olpred = value*np.ones((Nf,))
        self.clpred = value*np.ones((Nf,))
        self.menu   = list(menu)
        self.menu_names = dict(noise="Input Noise", dist="Step Disturbance")

class DVobj(Updatable):
    """Structure for disturbance variables."""
    def __init__(self, name=' ', desc=' ', units=' ', value=0.0, pltmax=100.0,
                 pltmin=0.0, noise=0.0, Nf=0, menu=("value","pltmax","pltmin",
                                                    "noise")):
        self.name   = name
        self.desc   = desc
        self.units  = units
        self.value  = value
        self.est    = value
        self.pltmax = pltmax
        self.pltmin = pltmin
        self.noise  = noise
        self.ref    = value
        self.chflag = 0
        self.Nf     = Nf
        self.olpred = value*np.ones((Nf,))
        self.clpred = value*np.ones((Nf,))
        self.menu   = list(menu)
        self.menu_names = dict()

class CVobj(Updatable):
    """Structure for controlled variables."""
    def __init__(self, name=' ', desc=' ', units=' ',  value=0.0, sstarg=0.0,
                 ssqval=1.0, setpoint=0.0, qvalue=1.0, maxlim=1.0e10,
                 minlim=-1.0e10, roclim=1.0e10, pltmax=100.0, pltmin=0.0,
                 noise=0.0, mnoise=0.000001, dist=0.0, Nf=0, bias=0.0,
                 lbslack=None, ubslack=None,
                 menu=("value","sstarg","ssqval","setpoint","qvalue",
                       "maxlim","minlim","pltmax","pltmin","mnoise",
                       "noise","dist")):
        self.name   = name
        self.desc   = desc
        self.units  = units
        self.value  = value
        self.est    = value
        self.sstarg = sstarg
        self.ssqval = ssqval
        self.setpoint = setpoint
        self.qvalue = qvalue
        self.maxlim = maxlim
        self.minlim = minlim
        self.pltmax = pltmax
        self.pltmin = pltmin
        self.noise  = noise
        self.mnoise = mnoise
        self.dist   = dist
        self.ref    = value
        self.chflag = 0
        self.Nf     = Nf
        self.bias   = bias
        self.lbslack = lbslack
        self.ubslack = ubslack
        self.olpred = value*np.ones((Nf,))
        self.clpred = value*np.ones((Nf,))
        self.menu   = list(menu)
        self.menu_names = dict()

class XVobj(Updatable):
    """Struct for estimated variables."""
    def __init__(self, name=' ', desc=' ', units=' ', 
                 value=0.0,
                 sstarg=0.0, ssqval=1.0,
                 setpoint=0.0, qvalue=1.0,
                 maxlim=1.0e10, minlim=-1.0e10,
                 pltmax=100.0, pltmin=0.0,
                 noise=0.0, mnoise=0.000001, dnoise=1, dist=0.0, Nf=0, bias=0.0,
                 menu=("value","sstarg","ssqval","setpoint","qvalue",
                       "maxlim","minlim","pltmax","pltmin","mnoise",
                       "noise","dist")):
        self.name   = name
        self.desc   = desc
        self.units  = units
        self.value  = value
        self.est    = value
        self.sstarg = sstarg
        self.ssqval = ssqval
        self.setpoint = setpoint
        self.qvalue = qvalue
        self.maxlim = maxlim
        self.minlim = minlim
        self.pltmax = pltmax
        self.pltmin = pltmin
        self.noise  = noise
        self.mnoise = mnoise
        self.dnoise = dnoise
        self.dist   = dist
        self.ref    = value
        self.chflag = 0
        self.Nf     = Nf
        self.bias   = bias
        self.olpred = value*np.ones((Nf,))
        self.clpred = value*np.ones((Nf,))
        self.menu   = list(menu)
        self.menu_names = dict()

class XIobj(Updatable):
    """Struct for invisible (not plotted) estimated states."""
    def __init__(self, name=' ', desc=' ', units=' ', 
                 value=0.0, Nf = 0,
                 menu=("value")):
        self.value  = value
        self.est    = value
        self.clpred = value*np.ones((Nf,))
        self.olpred = value*np.ones((Nf,))
        self.name   = name
        self.desc   = desc
        self.menu   = list(menu)
        self.menu_names = dict()

class SimCon(object):
    """Struct for simulation contents."""
    def __init__(self, simname="", mvlist=(), dvlist=(), cvlist=(), xvlist=(),
                 xilist=(), oplist=(), N=100, refint=100, runsim=False, deltat=1,
                 alg=None, proc=None, mod=None, F=None, l=None, Pf=None,
                 xmk=None, gain=None, ydata=(), udata=(), root=None,
                 savedefaults=True):
        self.simname = simname
        self.mvlist = VarList(mvlist)
        self.dvlist = VarList(dvlist)
        self.cvlist = VarList(cvlist)
        self.xvlist = VarList(xvlist)
        self.xilist = VarList(xilist)
        self.nmvs = len(self.mvlist)
        self.ndvs = len(self.dvlist)
        self.ncvs = len(self.cvlist)
        self.nxvs = len(self.xvlist)
        self.oplist = VarList(oplist)        
        self.N = N
        self.refint = refint
        self.runsim = runsim
        self.deltat = deltat
        self.alg = alg
        self.proc = proc
        self.mod = mod
        self.F = F
        self.l = l
        self.Pf = Pf
        self.xmk = xmk
        self.gain = gain
        self.ydata = list(ydata)
        self.udata = list(udata)
        self.root = root
        self.extra = {}
        self.__defaults = None
        self.__defaultfields = ["refint", "deltat", "ydata", "udata", "N"]
        self.__updatables = ["mvlist", "dvlist", "cvlist", "xvlist", "xilist", "oplist"]
        
        if savedefaults:
            self.savedefaults()
        
    def savedefaults(self):
        """
        Saves current settings as default values.
        
        Note that only the fields specified in __defaultfields are stored.        
        """
        kwargs = dict(savedefaults=False)
        for k in self.__defaultfields + self.__updatables:
            kwargs[k] = copy.deepcopy(getattr(self, k))
        self.__defaults = SimCon(**kwargs)
        
    def usedefaults(self):
        """
        Reverts settings to default values.
        
        Only settings in __defaultfields are reverted, and fields in
        self.__updatables have their _update methods called for each element in
        the list.
        """
        if self.__defaults is None:
            raise ValueError("No defaults have been saved!")
        for k in self.__defaultfields:
            val = copy.deepcopy(getattr(self.__defaults, k))
            setattr(self, k, val)
        for k in self.__updatables:
            varanddefault = zip(getattr(self, k), getattr(self.__defaults, k))
            for (var, defaultvar) in varanddefault:
                var._update(defaultvar)
        self.extra = {}
                
def makeaxes(fig, rows, cols, sharex=True):
    """
    Returns a numpy array of axes arranged in a grid.
    
    Note that we can't use plt.subplots because we clear the axes but not the
    figure.
    """
    axes = np.empty((rows, cols), dtype=object)
    for j in range(cols):
        xax = None
        for i in range(rows):
            ax = fig.add_subplot(rows, cols, i*cols + j + 1, sharex=xax)
            ax.set_visible(False)
            axes[i, j] = ax
            if sharex:
                xax = ax
    
    # Shut off ticks for all axes. Will be added back later.
    if sharex:
        for ax in axes.flat:
            for label in ax.get_xticklabels():
                label.set_visible(False)
    return axes
