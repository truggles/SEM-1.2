#!/usr/bin/env bash


Date="20200725" # Good nuclear
version="v5"
#Date="20200802" # Nuclear zero unmet demand
#version="v3"
Date="20200803" # NatGas+CCS
version="v2"
#ARGS="run_sem make_results_file"
#ARGS="make_results_file"
ARGS="run_sem make_results_file make_plots"
ARGS="make_plots"
ARGS="make_results_file"


        #"Case0_NuclearFlatDemand" \
        #"Case1_Nuclear" \
        #"Case2_NuclearStorage" \
        #"Case3_WindStorage" \
        #"Case4_SolarStorage" \
        #"Case5_WindSolarStorage" \
        #"Case6_NuclearWindSolarStorage" \


for CASE in \
        "Case7_NatGasCCS" \
        "Case8_NatGasCCSStorage" \
        "Case9_NatGasCCSWindSolarStorage" \
        ; do
     
    ./run_SEM_configs_fuels.py "date_$Date" $CASE "version_$version" $ARGS

done



