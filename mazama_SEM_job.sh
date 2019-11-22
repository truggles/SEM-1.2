#!/bin/tcsh
#PBS -N TestDefault
#PBS -l nodes=1:ppn=1
#PBS -q clab
#PBS -V
#PBS -m e
#PBS -M truggles@carnegiescience.edu
#PBS -e /data/cees/truggles/SEM_job_Default.err
#PBS -o /data/cees/truggles/SEM_job_Default.out
#
cd $PBS_O_WORKDIR
#
module load anaconda/anaconda3
module load gurobi752
python run_reliability_analysis.py "date_$DATE" "version_$VERSION" "reliability_$RELIABILITY" "wind_$WIND" "run_sem"
# "zero_storage" # <-- Can grab this for zero storage runs
# end script

