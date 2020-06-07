#!/bin/bash
#
#SBATCH --export=ALL
#SBATCH --time=10:00:00
#SBATCH --ntasks=4
#SBATCH --partition=clab
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#
cd $SLURM_SUBMIT_DIR
#
module load anaconda/anaconda3
module load gurobi
python run_SEM_configs_fuels.py "date_$DATE" "version_$VERSION" "run_sem" $EXTRA_ARGS
# end script
