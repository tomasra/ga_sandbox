#!/bin/sh
#SBATCH -p short

# $1 - input directory with parameter sets
# $2 - output directory

srun true
mpirun ./run.py $1 $2
