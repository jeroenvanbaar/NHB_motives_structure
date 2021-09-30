#!/bin/bash

# Job Name
#SBATCH -J simulate_model_study1

# Walltime requested
#SBATCH -t 24:00:00

# Provide index values (TASK IDs)
#SBATCH --array=201,
# SBATCH --array=201-353

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e Logs/Logs_study4_simulate/sub-%a_err.txt
#SBATCH -o Logs/Logs_study4_simulate/sub-%a_out.txt

# single core
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --account=carney-ofeldman-condo

# Messages to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jvb@brown.edu

# Use the $SLURM_ARRAY_TASK_ID variable to provide different inputs for each job
 
echo "Starting job for subject "$SLURM_ARRAY_TASK_ID

module load matlab/R2019a

matlab-threaded -nodisplay -nojvm -r "feature_learner_simulate($SLURM_ARRAY_TASK_ID, 1, 'FeatureRL','CoGrRiNa','point'); exit;"
