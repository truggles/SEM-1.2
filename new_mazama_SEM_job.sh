#!/bin/bash
#
#SBATCH --export=ALL
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --partition=clab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=truggles@carnegiescience.edu
#
cd $SLURM_SUBMIT_DIR
#
module load anaconda/anaconda3
module load gurobi
python run_SEM_configs_fuels.py "date_$DATE" "version_$VERSION" "run_sem" $EXTRA_ARGS
# end script
