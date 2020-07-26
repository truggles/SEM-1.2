#!/usr/bin/env bash


export DATE=20200608
export VERSION=v1
nJobs=25

        #"Case0_NuclearFlatDemand" \
        #"Case6_NuclearWindSolarStorage" \


for CASE in \
        "Case1_Nuclear" \
        "Case2_NuclearStorage" \
        "Case3_WindStorage" \
        "Case4_SolarStorage" \
        "Case5_WindSolarStorage" \
        ; do

    for (( JOB=1; JOB<=$nJobs; JOB++ )); do

        echo "${CASE} nJobs_${nJobs} jobNum_${JOB}"
        export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB} FULL_YEAR factor_1.1"
        export SBATCH_JOB_NAME=test_fuel_${DATE}_${VERSION}_${CASE}_${nJobs}_${JOB} 
        sbatch new_mazama_SEM_job.sh

    done

done
