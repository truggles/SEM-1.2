#!/usr/bin/env bash


Date="20200228"
version="v1"
#ARGS="run_sem make_results_file"
#ARGS="make_results_file"
ARGS="make_results_file make_plots"
ARGS="run_sem make_results_file make_plots"
ARGS="make_plots"
ARGS="run_sem"


        #"Case0_NuclearFlatDemand" \
        #"Case2_NuclearStorage" \
        #"Case3_WindStorage" \
        #"Case4_SolarStorage" \
        #"Case5_WindSolarStorage" \
        #"Case6_NuclearWindSolarStorage" \

for CASE in \
        "Case1_Nuclear" \
        ; do
     
    ./run_SEM_configs_fuels.py "date_$Date" $CASE "version_$version" $ARGS

done



