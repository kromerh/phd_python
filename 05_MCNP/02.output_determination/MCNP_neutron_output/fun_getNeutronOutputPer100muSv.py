import pandas as pd
import numpy as np
import os, glob
from shutil import copyfile
import re
import sys


# # copy the MCNP files
# lst_ID = np.arange(126,135,1)
# path = '//fs03//LTH_Neutimag//hkromer//10_Experiments//02_MCNP//'

# for ID in lst_ID:
# 	this_path = '{}CurrentTarget{}/CurrentTarget{}_normal/'.format(path, ID, ID)
# 	for fname in glob.glob('{}df_neutron_output_for_Edeut_*.csv'.format(this_path)):
# 		newfname = fname.replace('df_', 'df_ID{}_'.format(ID))
# 		_ = re.findall(r'(df.+)', newfname)
# 		if len(_)>0:
# 			newfname = _[0]
# 		newfname = '//fs03/LTH_Neutimag/hkromer/02_Simulations/01_Python/MCNP_neutron_output/MCNP_results_oldTarget/{}'.format(newfname)
# 		copyfile(fname, newfname)

def getNeutronOutputPer100muSv(HV=100, LB6411_distance=70, newTarget=1):
	"""
	Retrieves the neutron output per 100µSv/h as determined from MCNP. Only works for the new target. Returns that value
	HV: High voltage. This determines which MCNP run is taken to load the data. Default is -100 kV
	LB6411_distance: Distance between the source and LB6411 position. Default is 70 cm
	newTarget: if 1, then the files for the new target are used, 0 the files for the old target are used
	returns neutron output per 100 µSv/hr as read by the LB6411
	"""
	master_path = '/Users/hkromer/02_PhD/01.github/phd/05_MCNP/02.output_determination/MCNP_neutron_output/'
	if newTarget == 1:
		path_to_MCNP_OutputPer100muSv = '{}MCNP_results_newTarget/'.format(master_path)
	else:
		path_to_MCNP_OutputPer100muSv = '{}MCNP_results_oldTarget/'.format(master_path)


	# get which HVs have been simulated in MCNP
	lst_HV = []
	for fname in glob.glob('{}*.csv'.format(path_to_MCNP_OutputPer100muSv)):
		_ = re.findall(r'(\d+)\.csv', fname)
		lst_HV.append(int(_[0]))


	# list of the ID's for the respective MCNP simulation
	lst_ID = []
	for fname in glob.glob('{}*.csv'.format(path_to_MCNP_OutputPer100muSv)):
		_ = re.findall(r'ID(\d+)_', fname)
		lst_ID.append(int(_[0]))
	# print(lst_ID)
	# find index of the HV in the lst_HV
	try:
		idx = lst_HV.index(HV)
	except ValueError:
		idx = -1

	if idx == -1:
		print('--- Available high voltage settings: {}'.format(lst_HV))
		print('--- High voltage value of ' + str(HV) + ' is not in an MCNP run. sys.exit(). --- ')
		sys.exit()
	else:
		csv_name = '{}df_ID{}_neutron_output_for_Edeut_{}.csv'.format(path_to_MCNP_OutputPer100muSv, lst_ID[idx], lst_HV[idx])
		df = pd.read_csv(csv_name, header=0)

		distance = LB6411_distance

		neutronOutputPer100muSv = df.W[ df.distance == distance ].values
		# print(path_to_MCNP_OutputPer100muSv)

		return neutronOutputPer100muSv

# print(getNeutronOutputPer100muSv(HV=85, LB6411_distance=50, newTarget=0))