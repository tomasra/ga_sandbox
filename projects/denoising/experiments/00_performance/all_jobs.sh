#!/bin/bash

# Number of parallel tasks to begin with
TASK_COUNT_START=2

# Number of parallel tasks to end with
TASK_COUNT_FINISH=101

# How much time to wait (in seconds) before running squeue
# and so checking if the last job has finished
POLLING_PERIOD=5

# Move on to next job if last one does not end soon enough
MAX_POLLS=500

function start_job {
    TASK_COUNT=$1
    sbatch --ntasks=$TASK_COUNT single_job.sh $TASK_COUNT

    # ***TESTING***
    # JOB_ID=$[$TASK_COUNT+1000]
    # echo "$TASK_COUNT 99.999999" > "slurm-$JOB_ID.out"
    # echo "Submitted batch job $JOB_ID"
}

function job_running {
    JOB_ID=$1

    # If the job has finished running, squeue output will only contain
    # one line with column headers
    SQUEUE_WC=$(squeue -j $JOB_ID | wc -l)
    if [ $SQUEUE_WC -eq 1 ]; then
        echo 0
    else
        echo 1
    fi

    # ***TESTING***
    # echo 0
}

# Run job with each number of tasks:
for TASK_COUNT in `seq $TASK_COUNT_START $TASK_COUNT_FINISH`;
do
    # Create batch job and parse its ID
    # Output is supposed to be in such format:
    # "Submitted batch job 181057"
    JOB_ID=$(start_job $TASK_COUNT | awk '{ print $4 }')

    # Now wait until it disappears from squeue or poll count reaches maximum
    POLL_COUNT=0
    while [ $POLL_COUNT -le $MAX_POLLS ]; do
        sleep $POLLING_PERIOD
        let POLL_COUNT=POLL_COUNT+1
        JOB_RUNNING=$(job_running $JOB_ID)
        if [ $JOB_RUNNING -eq 0 ]; then
            break
        fi
    done
    # echo "$JOB_ID has finished"
done
