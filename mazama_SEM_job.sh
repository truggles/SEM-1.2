#!/bin/bash
#
#SBATCH --job-name=reli_${DATE}_${VERSION}_${RELIABILITY}_${WIND}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/data/cees/truggles/SEM-1.2/Output_Data/mazama_job_${DATE}_${VERSION}_${WIND}.err
#SBATCH --error=/data/cees/truggles/SEM-1.2/Output_Data/mazama_job_${DATE}_${VERSION}_${WIND}.out

module load anaconda/anaconda3
module load gurobi752
srun python run_reliability_analysis.py "date_$DATE" "version_$VERSION" "reliability_$RELIABILITY" "wind_$WIND" "run_sem" $EXTRA_ARGS
# end script
#SBATCH --time=0-15 # 0days-15hours
