#!/bin/sh
#SBATCH -p short
#SBATCH -C beta

# Arguments:
# $1 - MPI process count
# $2 - filename for results

# --ntasks should be passed via sbatch command line args
# The same number should also be passed for script itself
# Also need to set correct python environment:
srun true
mpirun -np $1 ../experiment.py \
    --output=$2 \
    --population-size=100 \
    --elite-size=10 \
    --crossover-rate=0.8 \
    --mutation-rate=0.001 \
    --chromosome-length=30 \
    --fitness-threshold=1.0 \
    --max-iterations=10 \
    --rng-freeze=True \
    --noise-type=snp \
    --noise-param=0.2
