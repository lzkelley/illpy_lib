#!/bin/bash
#    run with `sbatch run.sh`

#SBATCH --mail-user=lzkelley@northwestern.edu
#SBATCH -n 1
#SBATCH -p hernquist,itc_cluster
#SBATCH --mem-per-cpu=40000
#SBATCH --time=20:00:00
#SBATCH -o dets_out.%j
#SBATCH -e dets_err.%j
#SBATCH -J dets

python -m illpy_lib --RECREATE=True
