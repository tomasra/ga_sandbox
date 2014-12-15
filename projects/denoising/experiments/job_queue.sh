#!/bin/bash

# To be passed as parameter
JOB_SCRIPT=$1

RUN_START=1
RUN_END=10

RESULT_DIR=results
# Create if is not existing yet
mkdir -p $RESULT_DIR

# How much time to wait (in seconds) before running squeue
# and so checking if the last job has finished
POLLING_PERIOD=2

# Move on to next job if last one does not end soon enough
# Let's wait for two hours total (max time of 'short' job queue)
MAX_POLLS=3600

function start_job {
    sbatch $JOB_SCRIPT ./$RESULT_DIR/run$1

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
for RUN in `seq $RUN_START $RUN_END`;
do
    # Create batch job and parse its ID
    # Output is supposed to be in such format:
    # "Submitted batch job 181057"
    JOB_ID=$(start_job $RUN | awk '{ print $4 }')

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
done
