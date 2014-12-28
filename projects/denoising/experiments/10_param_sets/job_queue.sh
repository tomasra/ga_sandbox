#!/bin/bash

# --- Arguments:
# $1 - input directory with parameter sets
# $2 - output directory for results
# $3 - number of parallel batches
# $4 - number of processes per batch
INPUT_DIR=$1
OUTPUT_DIR=$2
BATCH_COUNT=$3
PROC_COUNT=$4
TAG_NAME=$5
CLUSTER_NAME=$6

# Make sure that result directory exists
mkdir -p $OUTPUT_DIR

# How much time to wait (in seconds) before running squeue
# and so checking if the last job has finished
POLLING_PERIOD=2

# Move on to next job if last one does not end soon enough
# Let's wait for two hours total (max time of 'short' job queue)
MAX_POLLS=3600

# Batches should be in separate directories
# numbered from 0 to N-1
BATCH_ID_CURRENT=0
BATCH_ID_END=$(ls -lA $INPUT_DIR | grep "^d" | wc -l)
BATCH_ID_END=$(($BATCH_ID_END-1))


BATCHES_RUNNING=0
while [ $BATCH_ID_CURRENT -le $BATCH_ID_END ]; do
    # Input directory with parameter sets for current batch
    BATCH_INPUT_DIR=${INPUT_DIR%%/}/${BATCH_ID_CURRENT}

    # Start the job
    sbatch --ntasks=$PROC_COUNT \
        --constraint=$CLUSTER_NAME \
        --job-name=$TAG_NAME \
        job.sh $BATCH_INPUT_DIR $OUTPUT_DIR
    BATCH_ID_CURRENT=$(($BATCH_ID_CURRENT+1))

    BATCHES_RUNNING=$(squeue | grep $TAG_NAME | wc -l)

    # When all jobs are up and running
    if [ $BATCHES_RUNNING -eq $BATCH_COUNT ]; then
        # Wait till one of the jobs finishes
        POLL_COUNT=0
        while [ $POLL_COUNT -le $MAX_POLLS ]; do
            sleep $POLLING_PERIOD
            let POLL_COUNT=POLL_COUNT+1

            # Running jobs according to squeue
            BATCHES_RUNNING=$(squeue | grep $TAG_NAME | wc -l)

            # A job has finished running?
            if [ $BATCHES_RUNNING -lt $BATCH_COUNT ]; then
                # Start over at the beginning of outer loop
                break
            fi
        done
    fi
done
