#!/usr/bin/env bash


export DATE=20200725
export VERSION=v5
nJobs=103

        #"Case0_NuclearFlatDemand" \

        #"Case2_NuclearStorage" \
        #"Case3_WindStorage" \
        #"Case4_SolarStorage" \
        #"Case5_WindSolarStorage" \

for CASE in \
        "Case1_Nuclear" \
        "Case6_NuclearWindSolarStorage" \
        ; do

    for (( JOB=1; JOB<=$nJobs; JOB++ )); do

        echo "${CASE} nJobs_${nJobs} jobNum_${JOB}"
        export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB} FULL_YEAR"
        export SBATCH_JOB_NAME=test_fuel_${DATE}_${VERSION}_${CASE}_${nJobs}_${JOB} 
        sbatch new_mazama_SEM_job.sh

    done

done
