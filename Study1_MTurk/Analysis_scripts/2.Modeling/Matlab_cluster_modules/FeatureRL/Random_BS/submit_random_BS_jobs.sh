#!/bin/bash

# Job Name
#SBATCH -J random_BS

# Walltime requested
#SBATCH -t 36:00:00

# Provide index values (TASK IDs)
#SBATCH --array=151-200

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e Logs/random_BS_sub-%a.err
#SBATCH -o Logs/random_BS_sub-%a.out

# Resources requested
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH --account=bibs-ofeldman-condo

# Messages to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jvb@brown.edu

# Use the $SLURM_ARRAY_TASK_ID variable to provide different inputs for each job
 
echo "Starting job for subject "$SLURM_ARRAY_TASK_ID

module load matlab/R2019a

matlab-threaded -nodisplay -nojvm -r "feature_learner_fit_random_BS($SLURM_ARRAY_TASK_ID, 5000, 'basedOnTrueBases', 'true'); exit;"
