import pandas as pd
import numpy as np
import os, sys, glob
import re
from scipy.interpolate import interp1d
# from fun_penumbra_investigation_relative_yield import *

# -----------------------------------------------------------------------------------------------------------------
# Neutron detection setting
# -----------------------------------------------------------------------------------------------------------------

# cell track data from the out file
detector_cell = 90  # detector cell

# -----------------------------------------------------------------------------------------------------------------
# Folder settings, MCNP must be finished, ptrac must be analyzed (df exist)
# -----------------------------------------------------------------------------------------------------------------

master_folder = '//fs03/LTH_Neutimag/hkromer/10_Experiments/02_MCNP/neutron_emitting_spot/2_case_files/_experiment_gauss_20180904/'
# master_folder ='D:/neutron_emitting_spotsize/'
# case which this run corresponds to and master folder where the case will be in is given by master script
# lst_cases = np.arange(0,24+1,1)  # be careful to update this in the excel file!
lst_cases = [0]
# load dataframes
# properties
df_properties = pd.read_csv('{}df_properties.csv'.format(master_folder), index_col=0)

# -----------------------------------------------------------------------------------------------------------------
# Import the df for each case
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# Get list of directories
# -----------------------------------------------------------------------------------------------------------------
# list of all the directories, not files
lst_runs = [filename for filename in os.listdir(master_folder) if os.path.isdir(os.path.join(master_folder,filename))]
lst_runs = ['{}{}/'.format(master_folder, this_folder) for this_folder in lst_runs]

# only case files
temp = []
for d in lst_runs:
	_ = re.findall(r'case_(\d+)',d)
	if len(_) > 0:
		if (int(_[0]) in lst_cases):	
			temp.append(d)
		
lst_runs = temp


# print(lst_runs)
df_main = pd.DataFrame()
for this_dir in lst_runs:
	print('*** Processing {}'.format(this_dir))
	_ = re.findall(r'case_(\d+)',this_dir)
	num = _[0]
	path_to_result = '{}case_{}_df_result.csv'.format(this_dir,num)
	# check if the result file exists
	if os.path.isfile(path_to_result):
		d_temp = pd.read_csv(path_to_result, index_col=0)
		df_main = pd.concat([df_main,d_temp], ignore_index=True)
		# load the additional parameters from the excel overview file
		# this_df_overview = df_overview[ (df_overview['case'] == int(num)) ]
		# df_overview_to_add = pd.concat([df_overview_to_add, this_df_overview], ignore_index=True)
		# df_main = pd.merge(df_main, this_df_overview, on='case')
	else:
		print('File {} does not exist.'.format(path_to_result))

df_main.to_csv('{}df_results.csv'.format(master_folder))