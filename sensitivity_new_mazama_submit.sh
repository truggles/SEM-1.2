#!/usr/bin/env bash


export DATE=20200826
nJobs=103


#export CASE="Case7_NatGasCCS"
#for EL in 0.5; do
#    export VERSION="v2EL${EL}"
##for NG in 0.75; do
#    #export VERSION="v2NG${NG}"
##    export VERSION="v2"
#    for (( JOB=1; JOB<=$nJobs; JOB++ )); do
#        echo "${CASE} version_${VERSION} nJobs_${nJobs} jobNum_${JOB}"
#        #export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB} FULL_YEAR H2_ONLY"
#        #export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB} FULL_YEAR H2_ONLY FIXED_NATGASCCS_${NG}"
#        export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB} FULL_YEAR H2_ONLY FIXED_ELECTROLYZER_${EL}"
#        export SBATCH_JOB_NAME=test_fuel_${DATE}_${VERSION}_${CASE}_${nJobs}_${JOB} 
#        sbatch new_mazama_SEM_job.sh
#    done
#done



export CASE="Case5_WindSolarStorage"
#export VERSION="v3"
##for (( JOB=1; JOB<=$nJobs; JOB++ )); do
#for JOB in 64 98 101 103; do
#    echo "${CASE} version_${VERSION} nJobs_${nJobs} jobNum_${JOB}"
#    export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB} FULL_YEAR H2_ONLY"
#    export SBATCH_JOB_NAME=test_fuel_${DATE}_${VERSION}_${CASE}_${nJobs}_${JOB} 
#    sbatch new_mazama_SEM_job.sh
#done
#for EL in 0.75 0.5; do
#    export VERSION="v3EL${EL}"
#    for (( JOB=1; JOB<=$nJobs; JOB++ )); do
#
#        echo "${CASE} version_${VERSION} nJobs_${nJobs} jobNum_${JOB}"
#        export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB} FULL_YEAR H2_ONLY FIXED_ELECTROLYZER_${EL}"
#        export SBATCH_JOB_NAME=test_fuel_${DATE}_${VERSION}_${CASE}_${nJobs}_${JOB} 
#        sbatch new_mazama_SEM_job.sh
#    done
#done
#for SOL in 0.75; do
#    export VERSION="v3SOL${SOL}"
#    #for (( JOB=1; JOB<=$nJobs; JOB++ )); do
#    for JOB in 63 99 100 102 103; do
#
#        echo "${CASE} version_${VERSION} nJobs_${nJobs} jobNum_${JOB}"
#        export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB} FULL_YEAR H2_ONLY FIXED_SOLAR_${SOL}"
#        export SBATCH_JOB_NAME=test_fuel_${DATE}_${VERSION}_${CASE}_${nJobs}_${JOB} 
#        sbatch new_mazama_SEM_job.sh
#    done
#done
#for WIND in 0.75 0.5; do
#    export VERSION="v3WIND${WIND}"
#    for (( JOB=1; JOB<=$nJobs; JOB++ )); do
#
#        echo "${CASE} version_${VERSION} nJobs_${nJobs} jobNum_${JOB}"
#        #export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB} FULL_YEAR H2_ONLY FIXED_NATGASCCS_${NG}"
#        export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB} FULL_YEAR H2_ONLY FIXED_WIND_${WIND}"
#        export SBATCH_JOB_NAME=test_fuel_${DATE}_${VERSION}_${CASE}_${nJobs}_${JOB} 
#        sbatch new_mazama_SEM_job.sh
#    done
#done






export CASE="Case9_NatGasCCSWindSolarStorage"
export VERSION="v4"
#for (( JOB=1; JOB<=$nJobs; JOB++ )); do
for JOB in 17 19 25 26 27 28 29 30 31 33 34 36 37 38 39 40 41 42 43 44 45 46 47 48 49 51 52 53 54 57 58 59 60 62 63 66 69 72 76 77 87 89 91 97 98 103; do

    echo "${CASE} version_${VERSION} nJobs_${nJobs} jobNum_${JOB}"
    export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB} FULL_YEAR H2_ONLY"
    export SBATCH_JOB_NAME=test_fuel_${DATE}_${VERSION}_${CASE}_${nJobs}_${JOB} 
    sbatch new_mazama_SEM_job.sh
done
#sleep 600
#for EL in 0.75 0.5; do
#    export VERSION="v4EL${EL}"
#    for (( JOB=1; JOB<=$nJobs; JOB++ )); do
#
#        echo "${CASE} version_${VERSION} nJobs_${nJobs} jobNum_${JOB}"
#        export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB} FULL_YEAR H2_ONLY FIXED_ELECTROLYZER_${EL}"
#        export SBATCH_JOB_NAME=test_fuel_${DATE}_${VERSION}_${CASE}_${nJobs}_${JOB} 
#        sbatch new_mazama_SEM_job.sh
#    done
#done
#sleep 600
#for SOL in 0.75 0.5; do
#    export VERSION="v4SOL${SOL}"
#    for (( JOB=1; JOB<=$nJobs; JOB++ )); do
#
#        echo "${CASE} version_${VERSION} nJobs_${nJobs} jobNum_${JOB}"
#        export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB} FULL_YEAR H2_ONLY FIXED_SOLAR_${SOL}"
#        export SBATCH_JOB_NAME=test_fuel_${DATE}_${VERSION}_${CASE}_${nJobs}_${JOB} 
#        sbatch new_mazama_SEM_job.sh
#    done
#done
#sleep 600
#for WIND in 0.75 0.5; do
#    export VERSION="v4WIND${WIND}"
#    for (( JOB=1; JOB<=$nJobs; JOB++ )); do
#
#        echo "${CASE} version_${VERSION} nJobs_${nJobs} jobNum_${JOB}"
#        export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB} FULL_YEAR H2_ONLY FIXED_WIND_${WIND}"
#        export SBATCH_JOB_NAME=test_fuel_${DATE}_${VERSION}_${CASE}_${nJobs}_${JOB} 
#        sbatch new_mazama_SEM_job.sh
#    done
#done
#sleep 600
#for NG in 0.75 0.5; do
#    export VERSION="v4NG${NG}"
#    for (( JOB=1; JOB<=$nJobs; JOB++ )); do
#
#        echo "${CASE} version_${VERSION} nJobs_${nJobs} jobNum_${JOB}"
#        export EXTRA_ARGS="${CASE} nJobs_${nJobs} jobNum_${JOB} FULL_YEAR H2_ONLY FIXED_NATGASCCS_${NG}"
#        export SBATCH_JOB_NAME=test_fuel_${DATE}_${VERSION}_${CASE}_${nJobs}_${JOB} 
#        sbatch new_mazama_SEM_job.sh
#    done
#done
#
#
