"""
"""
import os
import logging
from datetime import datetime

import numpy as np
import h5py


def load_hdf5_to_mem(fname):
    with h5py.File(fname, 'r') as data:
        # out = {kk: data[kk][()] if np.shape(data[kk]) == () else data[kk][:] for kk in data.keys()}
        # use `[()]` instead of `[:]` to handle scalar datasets
        out = {kk: data[kk][()] for kk in data.keys()}

    return out


def _distribute_snapshots(core, comm):
    """Evenly distribute snapshot numbers across multiple processors.
    """
    size = comm.size
    rank = comm.rank
    mySnaps = np.arange(core.sets.NUM_SNAPS)
    if size > 1:
        # Randomize which snapshots go to which processor for load-balancing
        mySnaps = np.random.permutation(mySnaps)
        # Make sure all ranks are synchronized on initial (randomized) list before splitting
        mySnaps = comm.bcast(mySnaps, root=0)
        mySnaps = np.array_split(mySnaps, size)[rank]

    return mySnaps


def _save_meta_to_hdf5(save_hdf5, sim_path, version, script_file):
    save_hdf5.attrs['created'] = str(datetime.now())
    save_hdf5.attrs['version'] = str(version)
    save_hdf5.attrs['script'] = str(os.path.abspath(script_file))
    save_hdf5.attrs['sim_path'] = str(sim_path)
    return


def _check_output_path(fname):
    fname = os.path.abspath(fname)
    path = os.path.dirname(fname)
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isdir(path):
        err = "ERROR: filename '{}' path '{}' is not a directory!".format(fname, path)
        logging.error(err)
        raise FileNotFoundError(err)

    return fname


# def git_hash():
#     import subprocess
#     cwd = os.path.abspath(illpy_lib.__file__)
#     hash = subprocess.check_output(["git", "describe", "--always"], cwd=cwd).strip().decode()
