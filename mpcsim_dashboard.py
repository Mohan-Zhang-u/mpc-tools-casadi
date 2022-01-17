"""
Dashboard for mpcsim simulations.

Choose the system you wish to simulate. Upon closing the simulation, you will
be brought back to this main menu.
"""
import matplotlib
matplotlib.use("Agg")

import collections
import numpy as np
import cstr_lqg_mpcsim as cstr_linear
import cstr_nmpc_mpcsim as cstr
import htr_nmpc_mpcsim as htr
import hab_nmpc_mpcsim as hab

simulations = collections.OrderedDict()

simulations["Linear CSTR"] = cstr_linear
simulations["Nonlinear CSTR"] = cstr
simulations["Fired Heater"] = htr
simulations["Hot Air Balloon"] = hab

def mainloop():
    """Runs the main loop allowing user to choose the simulation."""
    print(__doc__)
    keepgoing = True
    Ndig = int(round(np.log10(len(simulations))))
    fmt = "    (%{:d}s) %s".format(Ndig)
    choices = collections.OrderedDict()
    for (i, k) in enumerate(simulations.keys()):
        choices[str(i + 1)] = k
    while keepgoing:
        try:
            print("Make a selection:\n")
            for item in choices.items():
                print(fmt % item)
            print("")
            print(fmt % ("Q", "Quit\n"))
            choice = raw_input("Choice: ").strip().lstrip("(").rstrip(")")
            
            if choice in set(["Q", "q"]):
                keepgoing = False
            elif choice in choices:
                sim = choices[choice]
                print("Starting %s" % sim)
                simulations[sim].dosimulation()
                print(__doc__)
            else:
                print("Invalid choice '%s'. Make another selection.\n" % choice)
        except KeyboardInterrupt:
            print("\n")
            keepgoing = False
    print("Quitting.")

if __name__ == "__main__":
    mainloop()
