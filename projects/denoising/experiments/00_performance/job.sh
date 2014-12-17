#!/bin/sh
#SBATCH -p short
#SBATCH -C beta

# Arguments:
# $1 - result dir

# --ntasks should be passed via sbatch command line args
srun true
mpirun ./run.py $1
