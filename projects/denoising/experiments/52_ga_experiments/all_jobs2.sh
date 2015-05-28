#!/bin/bash

for run in  {1..10}
do
	sbatch job2.sh noisy-15-003-10 $run
done
sleep 3600

for run in  {1..10}
do
	sbatch job2.sh noisy-20-002-06 $run
done
sleep 3600

for run in  {1..10}
do
	sbatch job2.sh noisy-20-003-04 $run
done
sleep 3600

for run in  {1..10}
do
        sbatch job2.sh noisy-20-003-08 $run
done
sleep 3600

for run in  {1..10}
do
        sbatch job2.sh noisy-20-003-10 $run
done
