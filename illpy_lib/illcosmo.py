"""

"""

from collections import OrderedDict
import os
import numpy as np

import cosmopy

from zcode.constants import KPC, MSOL, YR, MPC

# Get local path, and data directory
_DATA_PATH = "%s/data/" % os.path.dirname(os.path.abspath(__file__))
# Contains cosmological values for each snapshot
_TIMES_FILENAME_TOS = "illustris-snapshot-cosmology-data.npz"
_TIMES_FILENAME_TNG = "illustris-tng_snapshot-cosmology-data.npz"


class _Illustris_Cosmology(cosmopy.Cosmology):

    Omega0 = None
    OmegaLambda = None
    OmegaBaryon = None
    HPAR = None
    FNAME = None
    H0 = None

    BOX_LENGTH_COM_MPC = 75                          # [cMpc = Mpc/h]
    BOX_VOLUME_MPC3 = None
    BOX_VOLUME_CGS = None
    NUM_SNAPS = None
    _BAD_SNAPS = None

    _Z_GRID = [10.0, 4.0, 2.0, 1.0, 0.5, 0.1, 0.02]
    _INTERP_POINTS = 40

    def __init__(self, core=None, log=None, BOX_LENGTH_COM_MPC=None, **kwargs):
        for kk, vv in kwargs.items():
            if not hasattr(self, kk):
                raise ValueError("Additional `kwargs` must already be defined in class definition!")
            setattr(self, kk, vv)

        super().__init__()
        if log is None and core is not None:
            log = core.log

        if log is not None:
            log.debug("Initializing `Illustris_Cosmology`")

        if BOX_LENGTH_COM_MPC is not None:
            self.BOX_LENGTH_COM_MPC = BOX_LENGTH_COM_MPC

        # tng_flag = core.sets.TNG
        # fname = _TIMES_FILENAME_TNG if tng_flag else _TIMES_FILENAME
        fname = os.path.join(_DATA_PATH, self.FNAME)

        cosmo_data = np.load(fname)
        self.snapshot_scales = cosmo_data['scale']
        msg = "Loaded cosmology with {} snapshot scales".format(self.snapshot_scales.size)
        if log is None:
            print(msg)
        else:
            log.info(msg)

        class CONV_ILL_TO_CGS:
            """Convert from illustris units to physical [cgs] units (multiply).
            """
            MASS = 1.0e10*MSOL/self.HPAR          # Convert from e10 Msol to [g]
            MDOT = 10.22*MSOL/YR                  # Multiply by this to get [g/s]
            DENS = 6.77025e-22                    # (1e10 Msol/h)/(ckpc/h)^3 to g/cm^3 *COMOVING*
            DIST = KPC/self.HPAR                  # Convert from [ckpc/h] to [comoving cm]
            VEL  = 1.0e5                          # [km/s] to [cm/s]
            CS   = 1.0                            # ??????? FIX

        class CONV_CGS_TO_SOL:
            """Convert from cgs units to (standard) solar units, e.g. Msol, PC, etc, by multiplication
            """
            MASS  = 1.0/MSOL                       # [g] ==> Msol
            MDOT  = YR/MSOL                        # [g/s] ==> [Msol/yr]
            DENS  = np.power(KPC/1000.0, 3.0)/MSOL          # [g/cm^3] ==> [Msol/pc^3]
            NDENS = np.power(KPC/1000.0, 3.0)               # [1/cm^3] ==> [1/pc^3]
            DIST  = 1000.0/KPC                         # [cm] ==> [pc]
            VEL   = 1.0e-5                         # [cm/s] ==> [km/s]
            ENER  = 1.0e-10                        # [erg/g] ==> [(km/s)^2]

        self.CONV_ILL_TO_CGS = CONV_ILL_TO_CGS
        self.CONV_CGS_TO_SOL = CONV_CGS_TO_SOL
        self._CONV_ILL_TO_SOL = None

        self.BOX_VOLUME_COM_MPC3 = np.power(self.BOX_LENGTH_COM_MPC, 3.0)
        self.BOX_VOLUME_COM_CGS = np.power(self.BOX_LENGTH_COM_MPC*MPC, 3.0)
        self.BOX_VOLUME_MPC3 = np.power(self.BOX_LENGTH_COM_MPC/self.HPAR, 3.0)
        self.BOX_VOLUME_CGS = np.power(self.BOX_LENGTH_COM_MPC*MPC/self.HPAR, 3.0)

        return

    @property
    def CONV_ILL_TO_SOL(self):
        if self._CONV_ILL_TO_SOL is None:
            class CONV_ILL_TO_SOL:
                """Convert from illustris units to standard solar units (e.g. Msol, pc), by multiplication
                """
                MASS = self.CONV_ILL_TO_CGS.MASS * self.CONV_CGS_TO_SOL.MASS  # e10 Msol to [Msol]
                MDOT = self.CONV_ILL_TO_CGS.MDOT * self.CONV_CGS_TO_SOL.MDOT  # to [Msol/yr]
                DENS = self.CONV_ILL_TO_CGS.DENS * self.CONV_CGS_TO_SOL.DENS  # to [Msol/pc^3]
                DIST = self.CONV_ILL_TO_CGS.DIST * self.CONV_CGS_TO_SOL.DIST  # to comoving-pc

                VEL = 1.0

            self._CONV_ILL_TO_SOL = CONV_ILL_TO_SOL

        return self._CONV_ILL_TO_SOL

    def scales(self):
        return np.array(self.snapshot_scales)

    def GET_BAD_SNAPS(self, run):
        return self._BAD_SNAPS[run]


class Illustris_Cosmology_TOS(_Illustris_Cosmology):

    Omega0 = 0.2726
    OmegaLambda = 0.7274
    OmegaBaryon = 0.0456
    HPAR = 0.704
    H0 = HPAR * 100.0

    FNAME = _TIMES_FILENAME_TOS
    NUM_SNAPS = 136

    _BAD_SNAPS = {1: [53, 55],
                  2: [],
                  3: []}


class Illustris_Cosmology_TNG(_Illustris_Cosmology):

    Omega0 = 0.3089
    OmegaLambda = 0.6911
    OmegaBaryon = 0.0486
    HPAR = 0.6774
    H0 = HPAR * 100.0

    FNAME = _TIMES_FILENAME_TNG
    NUM_SNAPS = 100

    _BAD_SNAPS = {1: [],
                  2: [],
                  3: []}


# def sim_cosmo(sim_name, **kwargs):
def cosmo_from_name(sim_name, **kwargs):
    if os.path.sep in sim_name:
        sim_name = os.path.join(sim_name, '')
        sim_name = os.path.split(os.path.dirname(sim_name))[-1]

    import re
    matches = re.findall('^L([0-9]{2,3})n([0-9]{2,4})([a-zA-Z]+)$', sim_name)
    if len(matches) != 1 or len(matches[0]) != 3:
        err = "Failed to match sim_name '{}' with Illustris specification!".format(sim_name)
        raise ValueError(err)

    length, pnum, name = matches[0]
    length = int(length)
    pnum = int(pnum)

    if name == 'TNG':
        cosmo = Illustris_Cosmology_TNG
    elif name == 'FP':
        cosmo = Illustris_Cosmology_TOS
    else:
        raise ValueError("Failed to match sim specification '{}'!".format(name))

    return cosmo(BOX_LENGTH_COM_MPC=int(length), **kwargs)


def cosmo_from_path(sim_path):
    fname_params = os.path.join(sim_path, 'param.txt-usedvalues')
    params = _load_used_params(fname_params)

    kwargs = {}
    cosmo_pars = ['Omega0', 'OmegaLambda', 'OmegaBaryon', ]
    for cp in cosmo_pars:
        kwargs[cp] = params[cp]

    hpar = params['HubbleParam']
    kwargs['HPAR'] = hpar
    kwargs['H0'] = hpar * 100.0
    box_length_mpc = params['BoxSize'] / 1000.0
    cosmo = Illustris_Cosmology_TNG(BOX_LENGTH_COM_MPC=box_length_mpc, **kwargs)

    mass = params['UnitMass_in_g'] / hpar
    dist = params['UnitLength_in_cm'] / hpar
    vel = params['UnitVelocity_in_cm_per_s']
    dens = mass / np.power(dist, 3.0)
    time = dist / vel
    mdot = mass / time

    cosmo.CONV_ILL_TO_CGS.MASS = mass
    cosmo.CONV_ILL_TO_CGS.MDOT = mdot
    cosmo.CONV_ILL_TO_CGS.DIST = dist
    cosmo.CONV_ILL_TO_CGS.DENS = dens
    cosmo.CONV_ILL_TO_CGS.VEL = vel
    cosmo._usedparams = params

    return cosmo


def _load_used_params(fname):
    params = OrderedDict()
    with open(fname, 'r') as load:
        for line in load.readlines():
            line = line.strip()
            if len(line) == 0:
                continue

            kk, vv = line.split()
            for tt in [np.int, np.float]:
                try:
                    vv = tt(vv)
                except ValueError:
                    pass
                else:
                    break
            else:
                vv = str(vv)

            try:
                params[kk] = vv
            except:
                print(f"kk = {kk} ({type(kk)}), vv = {vv} ({type(vv)})")
                raise
        
    return params
