# MPCTools: Nonlinear Model Predictive Control Tools for CasADi (Python Interface) #

Copyright (C) 2017

Michael J. Risbeck and James B. Rawlings.

MPCTools is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3, or (at your option) any later
version.

MPCTools is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the file
COPYING for more details.

## Availability ##

The most recent release of MPCTools is available for download from the
[Downloads][bbdownloads] section. Choose the appropriate version for Python 2
or 3. The development sources are hosted in a Mercurial repository on 
[Bitbucket][bitbucket].

## Installation ##

To use MPCTools, you will need a recent versions of

* Python 3.5+ or 2.7 (see below for Python 2.7 support)
* Numpy
* Scipy
* Matplotlib
* Tkinter (only needed for `*_mpcsim.py` examples)
* CasADi (Version >=3.0; [download here](http://files.casadi.org))

With these packages installed, MPCTools can be downloaded from the
[downloads][bbdownloads] section, and the `mpctools` folder can be manually 
placed in the user's Python path, or the provided setup script
`mpctoolssetup.py` can be used, e.g.,

    python3 mpctoolssetup.py install --user

to install for the current user only, or

    sudo python3 mpctoolssetup.py install

to install systemwide. Note that in these commands, you should use the
appropriate `python` command for the version you downloaded, i.e., `python3`
or `python2`.

Code is used by importing `mpctools` within python scripts. See sample
files for complete examples.

### Python 2.7 Support ###

In older versions of MPCTools, source files were written to be compatible with
Python 2.7, and Python 3 versions were automatically generated using Python's
`2to3` conversion utility. However, as of version 2.4, these roles are reversed.
That is, the source files for MPCTools and the example files are now written to
be compatible with Python 3.5+, and Python 2 versions are generated
automatically. The code has been written so as to require only a minimal set
of changes for Python 2.7 compatibility, but please report any bugs that you
find.

For normal users who use MPCTools fia the [downloads][bbdownloads] link, this
change should be completely transparent, and you can continue to update
MPCTools by re-downloading the approprate `.zip` file. However, for advanced
users who may be using MPCTools directly from a clone of the repository, you
will need to either start using Python 3, or you will have to switch to using
`.zip` files from the [downloads][bbdownloads] section.

Finally, note that [Python 2.7 end of life][py27eol] is Jaunary 1st, 2020.
After this point, Python 2.7 will no longer be supported by the Python
developers. At or before this date, we will stop releasing Python 2 versions
of MPCTools, so you should make plans to upgrade in the near future.

## Documentation ##

Documentation for MPCTools is included in each function. We also
provide a cheatsheet (`doc/cheatsheet.pdf`). See sample files for complete
examples.

## Citing MPCTools ##

Because MPCTools is primarily an interface to CasADi, you should cite CasADi as
described on its [website][casadipubs]. In addition, you can cite MPCTools as

- Risbeck, M.J., Rawlings, J.B., 2015. MPCTools: Nonlinear model predictive
  control tools for CasADi (Python interface).
  `https://bitbucket.org/rawlings-group/mpc-tools-casadi`.

## Bugs ##

Questions, comments, bug reports can be posted on the
[issue tracker][bbissues] on Bitbucket.

Robert D. Mcallister  
<rdmcallister@ucsb.edu>  
University of California-Santa Barbara
Department of Chemical Engineering

[bitbucket]: https://bitbucket.org/rawlings-group/mpc-tools-casadi
[bbissues]: https://bitbucket.org/rawlings-group/mpc-tools-casadi/issues
[bbdownloads]: https://bitbucket.org/rawlings-group/mpc-tools-casadi/downloads
[casadi]: https://casadi.org
[casadipubs]: https://github.com/casadi/casadi/wiki/Publications
[casadidownloads]: https://files.casadi.org
[py27eol]: https://www.python.org/dev/peps/pep-0373

