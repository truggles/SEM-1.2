#!/usr/bin/env bash



Date="20191119"
version="v10"
ARGS="run_sem make_results_file plot_results"
ARGS="run_sem"
ARGS="plot_results"


for RELIABILITY in 0.999; do
    for WIND in 0.0; do

        ./run_reliability_analysis.py "date_$Date" "version_$version" "reliability_$RELIABILITY" "wind_$WIND" $ARGS

    done
done

