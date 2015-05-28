#!/bin/bash

# --- Arguments:
# $1 - noisy image dir
# $2 - clear image dir
# $3 - main result dir
# $4 - parameter set dir
# $5 - cluster name - alpha/beta

NOISY_IMAGE_DIR=$1
CLEAR_IMAGE_DIR=$2
RESULT_DIR=$3
PARAMS_DIR=$4
CLUSTER_NAME=$5
QUEUE_ID=$6

USERNAME="tora6799"

IMAGE_COUNT=$(ls -lA $NOISY_IMAGE_DIR | grep "png" | wc -l)
echo $IMAGE_COUNT

# How much time to wait (in seconds) before running squeue
# and so checking if the last job has finished
POLLING_PERIOD=2

# Move on to next job if last one does not end soon enough
# Let's wait for two hours total (max time of 'short' job queue)
MAX_POLLS=3600

# Make sure that result directory exists
mkdir -p $RESULT_DIR

for PARAM_FILE in $PARAMS_DIR*; do
    if [ -f "$PARAM_FILE" ]; then
        # Get param set name from its file path
        PARAM_SET_NAME=$(basename $PARAM_FILE)
        PARAM_SET_NAME=${PARAM_SET_NAME%.*}
        # echo $PARAM_SET_NAME
        # echo $PARAM_FILE

        PARAM_SET_NAME=$QUEUE_ID

        # Start the job
        sbatch --ntasks=$IMAGE_COUNT \
            --constraint=$CLUSTER_NAME \
            --job-name=$PARAM_SET_NAME \
            job.sh $NOISY_IMAGE_DIR $CLEAR_IMAGE_DIR $RESULT_DIR $PARAM_FILE

        # Wait for it to finish
        POLL_COUNT=0
        while [ $POLL_COUNT -le $MAX_POLLS ]; do
            sleep $POLLING_PERIOD
            let POLL_COUNT=POLL_COUNT+1

            # Check if it has finished
            JOB_RUNNING=$(squeue | grep $PARAM_SET_NAME | grep $USERNAME | wc -l)
            if [ $JOB_RUNNING -eq 0 ]; then
                break
            fi
        done
    fi
done
