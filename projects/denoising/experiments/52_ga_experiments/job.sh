#!/bin/sh
#SBATCH -p short
#SBATCH -n10
#SBATCH -C alpha
srun true
mpirun ./run.py $1 $2
