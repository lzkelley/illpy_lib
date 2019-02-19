#!/bin/bash
#    run with `sbatch run.sh`

#SBATCH --mail-user=lzkelley@northwestern.edu
#SBATCH -n 1
#SBATCH -p hernquist,itc_cluster
#SBATCH --mem-per-cpu=40000
#SBATCH --time=20:00:00
#SBATCH -o mrgs_out.%j
#SBATCH -e mrgs_err.%j
#SBATCH -J mrgs

# python illpy_lib/illbh/details.py --RECREATE=True
python illpy_lib/illbh/mergers.py --RECREATE=True
# python illpy_lib/illbh/snapshots.py
