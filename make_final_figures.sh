#!/usr/bin/env bash


ARGS="make_plots"


#######################################################################
#### Used for final results figures for systems making electrofuels ###
#######################################################################
#Date="20200805" # NatGas+CCS
#version="v5"
#for CASE in \
#        "Case5_WindSolarStorage" \
#        "Case7_NatGasCCS" \
#        "Case9_NatGasCCSWindSolarStorage" \
#        ; do
#    ./run_SEM_configs_fuels.py "date_$Date" $CASE "version_$version" $ARGS
#done


#######################################
#### Used for final results figures ###
#######################################
Date="20201116"
version="v5fullReli"
./curtailment_figures.py "date_$Date" "Case_ALL" "version_$version" $ARGS H2_ONLY

Date="20201201"
version="v8pgp"
./pgp-style_curtailment_figures.py "date_$Date" "Case_ALL" "version_$version" $ARGS H2_ONLY INCLUDE_PGP

Date="20200805" # NatGas+CCS H2_ONLY
version="v2"
./curtailment_figures.py "date_$Date" "Case_ALL" "version_$version" $ARGS H2_ONLY
