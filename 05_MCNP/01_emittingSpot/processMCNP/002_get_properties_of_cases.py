import pandas as pd
import numpy as np
import os, sys, glob
import re
from scipy.interpolate import interp1d

sys.path.append('//fs03/LTH_Neutimag/hkromer/02_Simulations/01_Python/MCNP_useful_functions/')
from fun_MCNP_read_tally_data import *

# -----------------------------------------------------------------------------------------------------------------
# Neutron detection setting
# -----------------------------------------------------------------------------------------------------------------

# cell track data from the out file
detector_cell = 90  # detector cell

# -----------------------------------------------------------------------------------------------------------------
# Folder settings, MCNP must be finished, ptrac must be analyzed (df exist)
# -----------------------------------------------------------------------------------------------------------------
# which folder to modify
# invs = ['05','06','07','08','09','10']


master_folder = '//fs03/LTH_Neutimag/hkromer/10_Experiments/02_MCNP/neutron_emitting_spot/2_case_files/_experiment_gauss_20180808/'
# master_folder ='D:/neutron_emitting_spotsize/'
# case which this run corresponds to and master folder where the case will be in is given by master script
# lst_cases = np.arange(0,24+1,1)  # be careful to update this in the excel file!
lst_cases = [0]




# -----------------------------------------------------------------------------------------------------------------
# Populate the df
# -----------------------------------------------------------------------------------------------------------------
df = pd.DataFrame(columns = ['case','W_det','dist_det','D_W', 'D_det'])


for case in lst_cases:
	print('Case {}'.format(case))
	# template input file that will be modified
	# input_file = '//fs03/LTH_Neutimag/hkromer/10_Experiments/02_MCNP/neutron_emitting_spot/0_inputFile/000_case_{}'.format(case)

	# case folder
	case_folder = master_folder + 'case_{}/'.format(case)


	# open properties file
	with open('{}case_{}_properties.txt'.format(case_folder, case), 'r') as file:
		for line in file:
			line = line.rstrip()

			# detector width
			_ = re.findall(r'WxHxD: (\d.\d)x', line)
			if len(_) > 0:
				this_W_det = _[0]
				# print(this_det_width)

			# detector depth
			_ = re.findall(r'WxHxD: .*x.*x(\d.*) cm$', line)
			if len(_) > 0:
				this_D_det = _[0]
				# print(this_det_depth)

			# detector distance
			_ = re.findall(r'Distance to source: (\d+) cm', line)
			if len(_) > 0:
				this_dist_det = _[0]
				# print(this_det_dist)


			# tungsten depth
			_ = re.findall(r'Depth: (.+) ', line)
			if len(_) > 0:
				this_D_W = _[0]
				# print(this_tungsten_depth)

		file.close()


	# ['case','W_det','D_det','dist_det','D_W']
	# with detector width
	s = pd.Series([case, this_W_det, this_D_det, this_dist_det, this_D_W], index = df.columns)
	df = df.append(s, ignore_index=True)
	print('***** Processing finished.')

df.to_csv('{}df_properties.csv'.format(master_folder))
