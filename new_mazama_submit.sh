#!/usr/bin/env bash


export DATE=20200311
export VERSION=v1
nJobs=7

        #"Case2_NuclearStorage" \
        #"Case3_WindStorage" \
        #"Case4_SolarStorage" \
        #"Case5_WindSolarStorage" \
        #"Case6_NuclearWindSolarStorage" \
        #"Case0_NuclearFlatDemand" \


for CASE in \
        "Case1_Nuclear" \
        ; do

    for JOB in {1..$nJobs}; do

        export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB}"
        export SBATCH_JOB_NAME=test_fuel_${DATE}_${VERSION}_${CASE}_${nJobs}_${JOB} 
        sbatch new_mazama_SEM_job.sh

    done

done
