#!/usr/bin/env bash


Date="20200311"
version="v8"
#ARGS="run_sem make_results_file"
#ARGS="make_results_file"
ARGS="run_sem make_results_file make_plots"
ARGS="make_plots"
ARGS="make_results_file"
ARGS="make_plots"


        #"Case0_NuclearFlatDemand" \
        #"Case3_WindStorage" \
        #"Case4_SolarStorage" \
        #"Case5_WindSolarStorage" \
        #"Case6_NuclearWindSolarStorage" \

for CASE in \
        "Case1_Nuclear" \
        "Case2_NuclearStorage" \
        ; do
     
    ./run_SEM_configs_fuels.py "date_$Date" $CASE "version_$version" $ARGS

done



