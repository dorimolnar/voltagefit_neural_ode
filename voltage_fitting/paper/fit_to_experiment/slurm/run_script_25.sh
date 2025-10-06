#!/bin/bash
#SBATCH --array=0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=8:00:00


echo "Hello from task $SLURM_ARRAY_TASK_ID"

python -u /weka/macke/mwe528/thesis/voltagefit_neural_ode/voltage_fitting/nn_fitting_to_real/fte_parallel_base_script.py