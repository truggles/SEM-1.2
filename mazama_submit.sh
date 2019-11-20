





export DATE=20191120
export VERSION=v5

for reliability in 0.9997; do
    for wind in 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 2.25 2.5 2.75 3.0 3.5 4.0 4.5 5.0; do

        export RELIABILITY=$reliability
        export WIND=$wind
        qsub -V -N test_reli_${DATE}_${VERSION}_${WIND} mazama_SEM_job.sh \
            -e /data/cees/truggles/SEM-1.2/Output_Data/mazama_job_${DATE}_${VERSION}_${WIND}.err \
            -o /data/cees/truggles/SEM-1.2/Output_Data/mazama_job_${DATE}_${VERSION}_${WIND}.out

    done
done

