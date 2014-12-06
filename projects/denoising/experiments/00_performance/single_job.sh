#!/bin/sh
#SBATCH -p short
#SBATCH -C beta

# --ntasks should be passed via sbatch command line args
# The same number should also be passed for script itself
# Also need to set correct python environment:
srun true
mpirun -np $1 run.py
