#!/bin/bash
#SBATCH --job-name run_array       # Set a name for your job. This is especially useful if you have multiple jobs queued.
#SBATCH --partition short          # Slurm partition to use
#SBATCH --array=0-959              # How many tasks in the array
#SBATCH --time 0-00:45             # Wall time limit in D-HH:MM
#SBATCH --mem-per-cpu=4000         # Memory limit for each tasks (in MB)
#SBATCH -o experiments/out/run_array.out       # File to which STDOUT will be written
#SBATCH -e experiments/out/run_array.err       # File to which STDERR will be written
#SBATCH --mail-type=ALL            # Type of email notification- NONE,BEGIN,END,FAIL,ALL
#SBATCH --mail-user=ronakdm@uw.edu # Email to which notifications will be sent
 
EXP_NAME=multitarget_simulated_cond_5_incr
VENV=xai

module load Python/3.10.8-GCCcore-12.2.0
source ~/${VENV}/bin/activate
TQDM_DISABLE=1 python experiments/run_array.py $SLURM_ARRAY_TASK_ID $EXP_NAME