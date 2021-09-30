#!/bin/bash

# Job Name
#SBATCH -J fRL_st1_choice_CoGrRiEV

# Walltime requested
#SBATCH -t 24:00:00

# Provide index values (TASK IDs)
#SBATCH --array=1-150

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e Logs/fRL_st1_choice_CoGrRiEV/sub-%a_err.txt
#SBATCH -o Logs/fRL_st1_choice_CoGrRiEV/sub-%a_out.txt

# single core
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --account=carney-ofeldman-condo

# Messages to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jvb@brown.edu

# Use the $SLURM_ARRAY_TASK_ID variable to provide different inputs for each job

echo "Starting job for subject "$SLURM_ARRAY_TASK_ID

module load matlab/R2019a

# CoHgShSgPd, CoGrRiEV, CoST
# matlab-threaded -nodisplay -nojvm -r "feature_learner_fit_4($SLURM_ARRAY_TASK_ID, 10, 'fit_to','choice_only','features','CoGrRiEV','comb_ind','fullmodel','bounded_weights',0,'asymmetric_LR',0,'gaze',0); exit;"
matlab-threaded -nodisplay -nojvm -r "feature_learner_fit_4($SLURM_ARRAY_TASK_ID, 10, 'fit_to','choice_only','features','CoGrRiEV','bounded_weights',0,'asymmetric_LR',0,'gaze',0); exit;"