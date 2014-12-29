illpy - Python Package for the Illustris data
=============================================
**illpy** is a package designed to access and analyze data produced from the
[Illustris Simulations](http://www.illustris-project.org/w/index.php/The_Simulations).

The package contains two primary modules,  
    - **illcosmo** : simulation and snapshot cosmological parameters
        - Based on a table of cosmological parameters for each snapshot
        - Returns cosmological parameters (e.g. comoving distance) at or between snapshots
    - **illbh** : accessing blackhole specific files and data
        - Access BH 'mergers' information (from 'blackhole_mergers_*.txt' files)
        - Access BH 'details' information (from 'blackhole_details_*.txt' files)
        - This module produces intermediate data files for faster access.



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
|   |   |-- Cosmology.py                          : Contains `Cosmology` class for parameter calcs
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
