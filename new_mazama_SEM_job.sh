#!/bin/bash
#
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --partition=clab
#SBATCH --mail-type=ALL
#SBATCH --mail-user=truggles@carnegiescience.edu
#
cd $SLURM_SUBMIT_DIR
#
module load anaconda/anaconda3
module load gurobi752
python run_reliability_analysis.py "date_$DATE" "version_$VERSION" "reliability_$RELIABILITY" "wind_$WIND" "run_sem" $EXTRA_ARGS
# end script
