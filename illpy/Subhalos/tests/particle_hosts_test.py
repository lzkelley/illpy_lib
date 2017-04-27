"""
"""

import os
from .. import particle_hosts
from nose.tools import assert_equal, assert_raises, assert_true


def test_filenames():
    print("particle_hosts_test.filenames_test()")
    for run in range(1, 4):
        print("run = '{}'".format(run))
        processed_dir = particle_hosts.GET_PROCESSED_DIR(run)
        print("processed_dir = '{}'".format(processed_dir))
        assert_true(os.path.exists(processed_dir))
        # Offset table
        print("loading path for: offset table")
        path = particle_hosts.FILENAME_OFFSET_TABLE(run, 135, version='1.0')
        print(path)
        # bh-hosts-snap table
        print("loading path for: bh-hosts-snap table")
        path = particle_hosts.FILENAME_BH_HOSTS_SNAP_TABLE(run, 135, version='1.0')
        print(path)
        # bh-hosts table
        print("loading path for: bh-hosts table")
        path = particle_hosts.FILENAME_BH_HOSTS_TABLE(run, 135, version='1.0')
        print(path)

    print("using 'run' 0 should fail")
    assert_raises(KeyError, particle_hosts.GET_PROCESSED_DIR, 0)

    return
