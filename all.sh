#!/usr/bin/env bash


Date="20191023"
version="v5"
ARGS="run_sem make_results_file make_plots"
#ARGS="run_sem make_results_file"
ARGS="make_plots"

        #"Case1_Nuclear" \
        #"Case2_NuclearStorage" \
        #"Case3_WindStorage" \
        #"Case4_SolarStorage" \
        #"Case5_WindSolarStorage" \
        #"Case6_NuclearWindSolarStorage" \

for CASE in \
        "Case0_NuclearFlatDemand" \
        ; do
     
    ./run_SEM_configs_fuels.py "date_$Date" $CASE "version_$version" $ARGS

done



