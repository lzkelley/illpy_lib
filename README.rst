illpy - Python Package for the Illustris data
=============================================

**illpy** is a package designed to access and analyze data produced from the Illustris Simulations
http://www.illustris-project.org/w/index.php/The_Simulations

The package contains two primary modules,  
    - **illcosmo** : simulation and snapshot cosmological parameters
        - Based on a table of cosmological parameters for each snapshot
        - Returns cosmological parameters (e.g. comoving distance) at or between snapshots
    - **illbh** : accessing blackhole specific files and data
        - Access BH 'mergers' information (from 'blackhole_mergers_*.txt' files)
        - Access BH 'details' information (from 'blackhole_details_*.txt' files)
        - This module produces intermediate data files for faster access.



Installation
------------

The full git repository can be cloned (with permission) from:
https://bitbucket.org/lzkelley/illpy

The base directory contains the setup script 'setup.py' which can be run as  

    $ python setup.py install

Or, it can be installed as a 'development' version for the user with the command

    $ python setup.py develop --user 

This is the recommended method of installation as it builds in-place, allowing modification without
reinstalling, and does not require administrative access to build in a global directory.  This
method of installation can be executed using the 'setup.sh' script, i.e.

    $ bash setup.py

The cosmology module *illcosmo* should work immediately.
The Blackhole modules in **illbh** ***MUST BE INITIALIZED FIRST***.  Technically, they should self-
initialize the first time they are attempted to be used... but this process can take a couple of
hours so it will likely be better to just set it up first.  The python script **BuildFiles.py**
included in the base directory will configure the BH modules, and test them (in a simple way) to
make sure they are working properly.

**Configure 'illbh' by running 'BuildFiles.py'**:

    $ python BuildFiles.py

**This will construct about 10 GB of intermediate files allowing fast access to blackhole data.**  
**This process can take up to a couple of hours**

Working on **odyssey**, Id recommend using an interactive session to run 'BuildFiles.py':

    $ srun -n 1 --pty --x11=first -p interact --mem=4096 -t 400 bash    
    $ python BuildFiles.py

**Building the intermediate files requires access to the raw Illustris Blackhole data files.**  
**Once the intermediate files are produced, they are the only files required for usage of 'illbh',
ie. once intermediate files are produced, they could be copied off of 'odyssey' for future usage.**



illcosmo - Cosmological parameters for the illustris snapshots
--------------------------------------------------------------

The **illcosmo** module is a class, **illcosmo.Cosmology** (illpy/illcosmo/Cosmology.py), 
and a data set, **illpy/illcosmo/data/illustris-snapshot-cosmology-data.npz**.  The data set is
a table of cosmological measures (e.g. redshift, scalefactor, comoving and lum distance, etc)
obtained by numerically integrating the FLRW equations with appropriate parameters (densities).
The **Cosmology** class can be used to access those parameters at the scale-factor (time)
corresponding to each snapshot, or to interpolate between those values to find parameter values
at arbitrary scale-factors.

*Time* in illustris is measures in terms of scale-factor, and thus this is the basic independent
variable of the **Cosmology** class.  The parameter *`Cosmology.num`* gives the number of snapshots
(i.e. 136), and the scale-factor corresponding to each can be accessed using either the standard
array accessor '[]' or the function *`snapshotTimes()`* both of which accept a positive or negative
(to reverse index) integer between 0 and 'Cosmology.num-1' (i.e. {0,135}).

See the documentation inside **Cosmology.py** for detailed usage information.
Each cosmological measure (e.g. comoving distance) has its own function (e.g. 'Cosmology.comDist')
which accepts either  

- an 'integer' argument which refers to a particular illustris snapshot number  
  e.g. 'comDist(100)' returns the comoving distance at snapshot 100  
- a 'float' argument which refers to a particular scale-factor of the universe to interpolate to  
  e.g. 'comDist(0.5)' returns the comoving distance when the scalefactor a(t) = 0.5  


- Examples:

    >> import illpy.illcosmo as illcosmo      # Import 'illcosmo' module
    >> cosmo = illcosmo.Cosmology()           # Initialize the 'Cosmology' object
    >> firstRedshift = cosmo.redshift(0)      # Get the redshift of the first snapshot, number '0'
    >> halfRedshift = cosmo.redshift(0.5)     # Get the redshift at a scalefactor of '0.5'; z = 1.0
    >> sf = 0.5*(cosmo[-2] + cosmo[-1])       # Get scalefactor halfway between last two snapshots
    >> dm = cosmo.distMod(sf)                 # Get the distance modulus at that scale-factor



illbh - Acessing blackhole data from the illustris simulations
--------------------------------------------------------------


Contents
--------

illpy  
|-- illpy  
|   |-- AuxFuncs.py  
|   |-- Constants.py                              : Physical and numerical constants  
|   |-- illbh  
|   |   |-- BHConstants.py  
|   |   |-- BHDetails.py                          : Access BH Details data  
|   |   |-- BHMergers.py                          : Access BH Mergers data  
|   |   |-- __init__.py
|   |   |-- MatchDetails.pyx                      : Perform quick searches in details entries
|   |
|   |-- illcosmo
|   |   |-- Cosmology.py                          : Contains 'Cosmology' class for parameter calcs
|   |   |-- data
|   |   |   |-- illustris-snapshot-cosmology-data.npz
|   |   |
|   |   |-- __init__.py
|   |
|   |-- __init__.py
|   |-- MANIFEST.in
|
|-- README.md
|-- setup.py                                      : setup script to install package
|-- setup.sh                                      : bash script to run setup.py w/ standard config


