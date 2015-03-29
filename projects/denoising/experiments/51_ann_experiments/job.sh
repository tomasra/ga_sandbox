#!/bin/sh
#SBATCH -p short

# $1 - noisy image dir
# $2 - clear image dir
# $3 - main result dir
# $4 - param set file (.json)

srun true
mpirun ./run.py $1 $2 $3 $4
