#!/usr/bin/env python3

import subprocess

# The INTEGRATION_TEST in the argv runs the test in
# the Simple_Energy_Model.py script
def test_against_prior_results():

    subprocess.call(['python', 'Simple_Energy_Model.py', 'Output_Data/test_190726_reference/case_input_test_190726.csv', 'INTEGRATION_TEST'])

