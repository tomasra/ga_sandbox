#!/bin/bash

# GA parameters
TOURNAMENT_SIZE_MIN=2
TOURNAMENT_SIZE_MAX=10

# Parallel process count
PROC_COUNT=31

# How many times to run whole algorithm with each set of parameters
REPEAT_RUNS=20

# How much time to wait (in seconds) before running squeue
# and so checking if the last job has finished
POLLING_PERIOD=1

# Move on to next job if last one does not end soon enough
MAX_POLLS=1000

function start_job {
    TASK_COUNT=$1
    SELECTION_TYPE=$2
    TOURNAMENT_SIZE=$3
    
    # Submit slurm job
    sbatch --ntasks=$TASK_COUNT single_job.sh \
        $TASK_COUNT \
        output.json \
        $SELECTION_TYPE \
        $TOURNAMENT_SIZE


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


# Roulette wheel selection
for RUN in `seq 1 $REPEAT_RUNS`;
do
    # Create batch job and parse its ID
    # Output is supposed to be in such format:
    # "Submitted batch job 181057"
    JOB_ID=$(start_job $PROC_COUNT roulette 0 | awk '{ print $4 }')

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

    # Rename output file so its name has job ID
    OUTPUT_FILENAME="roulette-$RUN-$JOB_ID.json"
    mv output.json $OUTPUT_FILENAME
done

# Tournament selection
for RUN in `seq 1 $REPEAT_RUNS`;
do
    for TOURNAMENT_SIZE in `seq $TOURNAMENT_SIZE_MIN $TOURNAMENT_SIZE_MAX`;
    do
        # Create batch job and parse its ID
        # Output is supposed to be in such format:
        # "Submitted batch job 181057"
        JOB_ID=$(start_job $PROC_COUNT roulette 0 | awk '{ print $4 }')

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

        # Rename output file so its name has job ID
        OUTPUT_FILENAME="tournament$TOURNAMENT_SIZE-$RUN-$JOB_ID.json"
        mv output.json $OUTPUT_FILENAME
    done
done
