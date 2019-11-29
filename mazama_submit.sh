





export DATE=20191128
version=v1

for reliability in 1.0 0.9999 0.9997 0.999 0.99 0.9; do
    #export VERSION=${version}ZS${reliability}
    export VERSION=${version}N${reliability}
    for wind in 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.25 3.5 3.75 4.0 4.25 4.5 4.75 5.0; do

        export RELIABILITY=$reliability
        export WIND=$wind
        qsub -V -N test_reli_${DATE}_${VERSION}_${WIND} mazama_SEM_job.sh \
            -e /data/cees/truggles/SEM-1.2/Output_Data/mazama_job_${DATE}_${VERSION}_${WIND}.err \
            -o /data/cees/truggles/SEM-1.2/Output_Data/mazama_job_${DATE}_${VERSION}_${WIND}.out

    done
done

