#!/bin/bash
#    run with `sbatch run.sh`

#SBATCH --mail-user=lzkelley@northwestern.edu
#SBATCH -n 1
#SBATCH -p hernquist,itc_cluster
#SBATCH --mem-per-cpu=40000
#SBATCH --time=20:00:00
#SBATCH -o snaps_out.%j
#SBATCH -e snaps_err.%j
#SBATCH -J snaps

python illpy_lib/illbh/details.py --RECREATE=True
# python illpy_lib/illbh/snapshots.py
