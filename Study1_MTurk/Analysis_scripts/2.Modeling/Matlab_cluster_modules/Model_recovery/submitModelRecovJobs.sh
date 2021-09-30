#!/bin/bash

# Job Name
#SBATCH -J mod_rec_5

# Walltime requested
#SBATCH -t 24:00:00

# Provide index values (TASK IDs)
#SBATCH --array=1-100

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e Logs/smi-5_sim_index-%a_err.txt
#SBATCH -o Logs/smi-5_sim_index-%a_out.txt

# single core
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH --account=carney-ofeldman-condo

# Messages to
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jvb@brown.edu

# Use the $SLURM_ARRAY_TASK_ID variable to provide different inputs for each job

SMI=5
echo "Starting job for subject "$SLURM_ARRAY_TASK_ID "with simulation model "$SMI

module load matlab/R2019a

matlab-threaded -nodisplay -nojvm -r "Step2_fit_simulated_data($SMI, $SLURM_ARRAY_TASK_ID, 10, false); exit;"