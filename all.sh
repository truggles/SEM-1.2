#!/usr/bin/env bash


Date="20191105"
version="Nov05v3"
#ARGS="run_sem make_results_file"
#ARGS="make_results_file"
ARGS="make_plots"
ARGS="run_sem make_results_file make_plots"

        #"Case0_NuclearFlatDemand" \

        #"Case1_Nuclear" \
        #"Case2_NuclearStorage" \
        #"Case3_WindStorage" \
        #"Case4_SolarStorage" \
        #"Case5_WindSolarStorage" \

for CASE in \
        "Case6_NuclearWindSolarStorage" \
        ; do
     
    ./run_SEM_configs_fuels.py "date_$Date" $CASE "version_$version" $ARGS

done



