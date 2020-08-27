#!/usr/bin/env bash


ARGS="make_plots"
ARGS="make_results_file"




Date="20200826" # NatGas+CCS H2_ONLY
version="v2"
for CASE in \
        "Case7_NatGasCCS" \
        ; do
    for APP in "EL0.5" "EL0.75" "NG1.0" "NG0.75" "NG0.5"; do
        ./run_SEM_configs_fuels.py "date_$Date" $CASE "version_${version}${APP}" $ARGS H2_ONLY
    done
done

version="v3"
for CASE in \
        "Case5_WindSolarStorage" \
        ; do
    for APP in "" "EL0.5" "EL0.75" "SOL0.75" "SOL0.5" "WIND0.75" "WIND0.5"; do
        ./run_SEM_configs_fuels.py "date_$Date" $CASE "version_${version}${APP}" $ARGS H2_ONLY
    done
done

version="v4"
for CASE in \
        "Case9_NatGasCCSWindSolarStorage" \
        ; do
    for APP in "" "EL0.5" "EL0.75" "SOL0.75" "SOL0.5" "WIND0.75" "WIND0.5" "NG0.75" "NG0.5"; do
        ./run_SEM_configs_fuels.py "date_$Date" $CASE "version_${version}${APP}" $ARGS H2_ONLY
    done
done

