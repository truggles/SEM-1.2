#!/usr/bin/env bash





export DATE=20191217
version=v4


renewables_scan=true
#renewables_scan=false
qmu_scan=true
qmu_scan=false



if $renewables_scan; then
    #for reliability in 1.0 0.9999 0.9997 0.999 0.99 0.9; do
    for reliability in 1.0 0.999; do
        export VERSION=${version}ZSTX${reliability}
        #export VERSION=${version}N${reliability}
        for wind in 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0; do
    
            export RELIABILITY=$reliability
            export WIND=$wind
            export EXTRA_ARGS="zero_storage TEXAS"
            qsub -V -N test_reli_${DATE}_${VERSION}_${WIND} mazama_SEM_job.sh
    
        done
    done
fi

if $qmu_scan; then
    wind=1.0
    solar=0.75
    for reliability in 0.999; do
        #export VERSION=${version}ZS${reliability}
        export VERSION=${version}N${reliability}
        for nuclear in 1.0 1.02 1.04 1.06 1.08 1.1 1.12 1.14 1.16 1.18 1.2 1.22 1.24 1.26 1.28 1.3; do
    
            export RELIABILITY=$reliability
            export WIND=${wind}
            export EXTRA_ARGS="solar_${solar} nuclear_SF_${nuclear} qmu_scan"
            echo "reliability_${RELIABILITY} wind_${WIND} ${EXTRA_ARGS}"
            qsub -V -N test_reli_${DATE}_${VERSION}_${WIND}_${solar}_${nuclear} mazama_SEM_job.sh
    
        done
    done
fi

