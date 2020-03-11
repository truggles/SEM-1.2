# To Remember:

plk files and plot directories all moved into `plots_reli`
Also see: `agu_poster` and `demand_plots`

# To run:
* Local demand - VRE analysis `./make_basic_scan_plot.py`
* MEM on Mazama:
   * based off of file `./run_reliability_analysis.py`
   * Submit using these 2 files (2 version for old and new system):
   * `mazama_SEM_job.sh`
   * `mazama_submit.sh`
   * `new_mazama_SEM_job.sh`
   * `new_mazama_submit.sh`
   * Waiting for Gurobi license to be updated to run again
   * For aggregating results on Mazama: `XXX`
   * Plotting: `plot_reli.sh`
