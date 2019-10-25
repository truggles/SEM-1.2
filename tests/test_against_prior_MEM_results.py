#!/usr/bin/env python3

import subprocess
import os


# Need to get MEM base directory to add path for importing Preprocess_Input
# so this can be run in a different directory
import sys
cwd = os.getcwd()
tmp = cwd.strip().split('/')
while True:
    if 'SEM-1.2' in tmp[-1]:
        break
    else:
        tmp.pop()
mem_base_dir = '/'.join(tmp)
#print(f"MEM base directory: {mem_base_dir}")
sys.path.append(mem_base_dir)


from Preprocess_Input import preprocess_input
from Save_Basic_Results import read_pickle_raw_results





# The INTEGRATION_TEST in the argv runs the test in
# the Simple_Energy_Model.py script
def test_against_prior_MEM_results():

    # Change to MEM base directory
    os.chdir(mem_base_dir)

    # Run the test cases for updated MEM
    subprocess.call(['python', 'Simple_Energy_Model.py', f'{mem_base_dir}/tests/case_input_test_190726.csv', 'INTEGRATION_TEST'])


    # From the input file name, get the dictionaries needed to easily open results file
    global_dic,case_dic_list = preprocess_input(f'{mem_base_dir}/tests/case_input_test_190726.csv')


    # Values from: https://github.com/ClabEnergyProject/SEM-1.2/blob/master/Output_Data/test_190726_reference/test_190726_20190727_165256.csv#L55
    prior_results_map = {
            # case name       :  system cost ($/kWh)
            'wind+wind2+unmet' : 0.17224779438447668,
            'solar+solar2+PGP' : 0.5618362601149164,
            'solar+solar2+storage+storage2_0.30' : 0.29850788978488885,
            'nuclear+CSP_0.25' : 0.18707972211774984,
            'natgas+unmet' : 0.10702726219181723,
            'natgasCCS+nuclear_0.5' : 0.05088786294328981,
            'EIAbase' : 0.05339458150656109
    }

    for case_in in case_dic_list:

        assert(case_in['CASE_NAME'] in prior_results_map.keys())

        #Try reading pickled results
        results = read_pickle_raw_results(global_dic, case_in)

        print(case_in['CASE_NAME'], results['SYSTEM_COST'])
        print(f"Comparison for {case_in['CASE_NAME']}: {round(prior_results_map[case_in['CASE_NAME']],10)} == {round(results['SYSTEM_COST'],10)}")
        assert(round(prior_results_map[case_in['CASE_NAME']],4) == round(results['SYSTEM_COST'],4)) 


if '__main__' in __name__:
    test_against_prior_MEM_results()
