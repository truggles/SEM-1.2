





export DATE="20191119"
export VERSION="v13"

for reliability in 0.999; do
    for wind in 0.0; do

        export RELIABILITY=$reliability
        export WIND=$wind
        qsub -V -N test_reli_$DATE_$VERSION mazama_SEM_job.sh

    done
done

