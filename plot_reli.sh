DATE=20191128
VERSION=v1

#for REL in 1.0 0.9999 0.9997 0.999 0.99 0.9; do
#    for TYPE in ZS; do
#        ./run_reliability_analysis.py date_${DATE} version_${VERSION}${TYPE}${REL} reliability_${REL} plot_results post_mazama
#    done
#done
#
#DATE=20191128
#VERSION=v1
#
##for REL in 1.0 0.9999 0.9997 0.999; do


#for REL in 0.999; do
#    for TYPE in N; do
#        ./run_reliability_analysis.py date_${DATE} version_${VERSION}${TYPE}${REL} reliability_${REL} plot_results post_mazama
#    done
#done


##DATE=20191130
##VERSION=v2
##for REL in 0.99 0.9; do
##    for TYPE in N; do
##        ./run_reliability_analysis.py date_${DATE} version_${VERSION}${TYPE}${REL} reliability_${REL} plot_results post_mazama
##    done
##done

#./run_reliability_analysis.py date_20191120 version_v13 reliability_0.99 plot_results post_mazama
#./run_reliability_analysis.py date_20191120 version_v12 reliability_0.999 plot_results post_mazama
#./run_reliability_analysis.py date_20191120 version_v11 reliability_0.9997 plot_results post_mazama

#./run_reliability_analysis.py version_v2N0.99 reliability_0.99 date_20191202 plot_results post_mazama qmu_scan
#./run_reliability_analysis.py version_v2N0.999 reliability_0.999 date_20191202 plot_results post_mazama qmu_scan
#./run_reliability_analysis.py version_v2N1.0 reliability_1.0 date_20191202 plot_results post_mazama qmu_scan
#



#for version in v1 v2 v3 v4 v5; do
#    ./run_reliability_analysis.py version_${version}N0.999 reliability_0.999 date_20191203 plot_results post_mazama qmu_scan
#done
#
#./run_reliability_analysis.py version_v1N1.0 reliability_1.0 date_20191203 plot_results post_mazama qmu_scan

./run_reliability_analysis.py version_v9ZSTX1.0 reliability_1.0 date_20191218 plot_results post_mazama
./run_reliability_analysis.py version_v9ZSTX0.999 reliability_0.999 date_20191218 plot_results post_mazama
./run_reliability_analysis.py version_v10NTX0.999 reliability_0.999 date_20191218 plot_results post_mazama
