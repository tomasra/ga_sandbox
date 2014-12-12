#!/bin/sh
#SBATCH -p short
#SBATCH -C beta
#SBATCH -n100

# Arguments:
# $1 - result directory

srun true
mpirun ./all_runs.py $1
