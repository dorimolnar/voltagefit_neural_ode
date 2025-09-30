#!/bin/bash
#SBATCH --array=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00


echo "Hello from task $SLURM_ARRAY_TASK_ID"

# Start timing
start_time=$(date +%s)


source ~/anaconda3/etc/profile.d/conda.sh
conda activate gpu_jaxley_env

python -u /weka/macke/mwe528/thesis/universal_diff_eq/voltage_fitting/nn_fitting_to_real/multi_fitting/fmte_parallel_base_script.py

# End timing
end_time=$(date +%s)
elapsed=$((end_time - start_time))

# Convert to readable format
hours=$((elapsed / 3600))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$((elapsed % 60))

echo "Job completed in $hours hours, $minutes minutes, and $seconds seconds."
