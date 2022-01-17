# This is a siso heater pid example.

# imports

from mpctools import mpcsim as sim
import random as rn

# pid control class

class PID:

    def __init__(self,mv, cv, Kc, ti, td, dt, opnclsd):

        self.mv      = mv
        self.cv      = cv
        self.Kc      = Kc
        self.ti      = ti
        self.td      = td
        self.dt      = dt
        self.opnclsd = opnclsd
        self.ek      = 0.0
        self.ekm1    = 0.0
        self.ekm2    = 0.0

    def calc(self, yk, ukm1):

        # calculate cv error

        self.ek = float(self.cv.setpoint - yk)

        # initialize input

        uk = ukm1

        # calculate pid input adjustment if control is on

        if (self.opnclsd.status.get() == 1):

            uk   = ukm1 + self.Kc*((self.ek - self.ekm1) 
                   + (self.dt/self.ti)*self.ek
                   + (self.td/self.dt)*(self.ek - 2*self.ekm1 + self.ekm2))
            if uk > self.mv.maxlim:
                uk = self.mv.maxlim
            if uk < self.mv.minlim:
                uk = self.mv.minlim

        # shift the error history values

        self.ekm2  = self.ekm1
        self.ekm1  = self.ek

        # return mv

        return uk

# siso first order process class

class SISOFO:

    def __init__(self, mv, cv, aproc, bproc, cproc):

        self.mv    = mv
        self.cv    = cv
        self.aproc = aproc
        self.bproc = bproc
        self.cproc = cproc
        self.dxkm1 = 0.0

    def update(self,ukm1):

        dukm1 = ukm1 - self.mv.ref
        dxk   = self.aproc*self.dxkm1 + self.bproc*dukm1
        dyk   = self.cproc*dxk
        yk    = dyk + self.cv.ref

        # shift history

        self.dxkm1 = dxk

        # return output

        return yk

# define the simulation function

def runsim(k, simcon, opnclsd):

    print("runsim: iteration %d -----------------------------------" % k)

    # unpack stuff from simulation container

    mvlist = simcon.mvlist
    dvlist = simcon.dvlist
    cvlist = simcon.cvlist
    oplist = simcon.oplist
    deltat = simcon.deltat

    # get mv, cv and options

    mv = mvlist[0]
    cv = cvlist[0]
    nf = oplist[0]
    kc = oplist[1]
    ti = oplist[2]
    td = oplist[3]
    av = oplist[4]
    vrlist = [mv,cv,nf,kc,ti,td,av]

    # check for changes

    chsum = 0
    for var in vrlist:
        chsum += var.chflag
        var.chflag = 0

    # initialize values on first execution or when something changes

    if (k == 0 or chsum > 0):

        print("runsim: initialization")

        # define PID controller

        KC  = kc.value
        TI  = ti.value
        TD  = td.value
        dt  = simcon.deltat
        pid = PID(mv, cv, KC, TI, TD, dt, opnclsd)
        simcon.alg = pid

        # define SISO first order process

        aproc = av.value
        bproc = 1.0
        cproc = 1.0
        sisofo = SISOFO(mv, cv, aproc, bproc, cproc)
        simcon.proc = sisofo

    # get previous input

    ukm1 = mv.value
        
    # update process model to get current output

    yk = simcon.proc.update(ukm1)

    # add noise if desired

    if (nf.value > 0.0):

        yk += nf.value*rn.uniform(-cv.noise,cv.noise) 

    # do PID calculation to get current input

    uk = simcon.alg.calc(yk, ukm1)

    # load current values

    mv.value = uk
    cv.value = yk
    cv.est   = yk

# set up heater example

simname = 'Heater PID Example'

# define mv and cv variables

MVmenu=["value","maxlim","minlim","pltmin","pltmax"]
CVmenu=["setpoint","pltmin","pltmax","noise"]

FGFCSP = sim.MVobj(name='FGFCSP', desc='fuel gas SP          ', units='BPH   ', 
               pltmin=0, pltmax=100,
               value=50, minlim=5, maxlim=95, noise=0.1, menu=MVmenu)
COMBTO = sim.CVobj(name='COMBTO', desc='combined outlet temp ', units='DEGF  ', 
               pltmin=700, pltmax=800,
               value=750, setpoint=750, minlim=705, maxlim=795, noise=0.5,
               menu=CVmenu)

# define options

NF = sim.Option(name='NF', desc='Noise Factor', value=0.0)

KC = sim.Option(name='KC', desc='Control Gain', value=0.4)

TI = sim.Option(name='TI', desc='Reset Time', value=5.0)

TD = sim.Option(name='TD', desc='Derivative Time', value=0.1)

AV = sim.Option(name='AV', desc='A Value', value=0.5)

# create simulation container

MVlist = [FGFCSP]
DVlist = []
CVlist = [COMBTO]
OPlist = [NF,KC,TI,TD,AV]
DeltaT = 1.0
N = 120
refint = 100
simcon = sim.SimCon(simname=simname,
                    mvlist=MVlist, dvlist=DVlist, cvlist=CVlist, 
                    oplist=OPlist, N=N, refint=refint, runsim=runsim, deltat=DeltaT)

# build the GUI and start it up

sim.makegui(simcon)

