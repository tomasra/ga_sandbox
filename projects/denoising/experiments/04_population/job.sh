#!/bin/sh
#SBATCH -p short
#SBATCH -C beta
#SBATCH -n100

# Arguments:
# $1 - result directory

srun true
mpirun ./runs.py $1
