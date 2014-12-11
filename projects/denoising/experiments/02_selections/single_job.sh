#!/bin/sh
#SBATCH -p short
#SBATCH -C beta

# Arguments:
# $1 - MPI process count
# $2 - filename for results
# $3 - selection type
# $4 - tournament size, if applicable

# --ntasks should be passed via sbatch command line args
# The same number should also be passed for script itself
# Also need to set correct python environment:
srun true
mpirun -np $1 ../experiment.py \
    --output=$2 \
    --population-size=100 \
    --elite-size=10 \
    --selection=$3 \
    --tournament-size=$4 \
    --crossover-rate=0.8 \
    --mutation-rate=0.005 \
    --chromosome-length=30 \
    --fitness-threshold=0.98 \
    --max-iterations=1000 \
    --noise-type=snp \
    --noise-param=0.2 \
