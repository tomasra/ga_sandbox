#!/bin/bash

for run in  {1..10}
do
	sbatch job.sh noisy-00-003-06 $run
done
sleep 3600

for run in  {1..10}
do
	sbatch job.sh noisy-05-002-10 $run
done
sleep 3600

for run in  {1..10}
do
	sbatch job.sh noisy-10-001-08 $run
done
sleep 3600

for run in  {1..10}
do
        sbatch job.sh noisy-10-004-04 $run
done
sleep 3600

for run in  {1..10}
do
        sbatch job.sh noisy-15-00-08 $run
done
