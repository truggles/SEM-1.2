#!/bin/tcsh
#PBS -N TestJobv2
#PBS -l nodes=1:ppn=1
#PBS -q clab
#PBS -V
#PBS -m e
#PBS -M truggles@carnegiescience.edu
#PBS -e /data/cees/truggles/SEM_job_Nv2.err
#PBS -o /data/cees/truggles/SEM_job_Nv2.out
#
cd $PBS_O_WORKDIR
#
module load anaconda/anaconda3
module load gurobi752
python Simple_Energy_Model.py ./$input_file
# end script

