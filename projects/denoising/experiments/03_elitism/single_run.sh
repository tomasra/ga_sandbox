#!/bin/sh
#SBATCH -p short
#SBATCH -C beta
#SBATCH -n30

# Arguments:
# $1 - result directory (should not be existing)

srun true
mpirun -np 30 ./all_runs.py $1
