#!/bin/sh
#SBATCH -p short
#SBATCH -C beta
#SBATCH -n30

# Arguments:
# $1 - result directory (should not be existing)

mpirun ./run.py $1
