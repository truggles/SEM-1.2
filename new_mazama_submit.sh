#!/usr/bin/env bash


export DATE=20200805
export VERSION=v5
nJobs=103

        #"Case0_NuclearFlatDemand" \

        #"Case2_NuclearStorage" \
        #"Case3_WindStorage" \
        #"Case4_SolarStorage" \
        #"Case1_Nuclear" \
        #"Case6_NuclearWindSolarStorage" \
        #"Case8_NatGasCCSStorage" \ # not interesting by default b/c zero storage is built

for CASE in \
        "Case5_WindSolarStorage" \
        "Case7_NatGasCCS" \
        "Case9_NatGasCCSWindSolarStorage" \
        ; do

    for (( JOB=1; JOB<=$nJobs; JOB++ )); do

        echo "${CASE} nJobs_${nJobs} jobNum_${JOB}"
        export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB} FULL_YEAR H2_ONLY"
        #export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB} FULL_YEAR"
        export SBATCH_JOB_NAME=test_fuel_${DATE}_${VERSION}_${CASE}_${nJobs}_${JOB} 
        sbatch new_mazama_SEM_job.sh

    done

done
