#!/usr/bin/env bash

# need to submit jobs to SLURM such that each job waits for the prior job to finish

echo "Running master script"

JOBID1=$(sbatch --parsable grk_rnn_train_1.sh)

JOBID2=$(sbatch --parsable --dependency=afterok:$JOBID1 grk_rnn_train_2.sh)

JOBID3=$(sbatch --parsable --dependency=afterok:$JOBID2 grk_rnn_train_3.sh)

JOBID4=$(sbatch --parsable --dependency=afterok:$JOBID3 grk_rnn_train_4.sh)
