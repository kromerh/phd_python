import pandas as pd
pd.set_option("display.max_columns",101)
import re
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# import the source definition file
master_dir = 'D:/neutron_emitting_spotsize/check_source_gauss/'   # 2018-09-04
fname = '{}/df_res.csv'.format(master_dir)
df_source_check = pd.read_csv(fname)
# print(df_source_check)


# get the radius definition from the mcnp_filename as a new column
def get_radius(x):
	radius = re.findall(r'rad(0.\d\d\d\d\d)', x)[0]
	return radius

df_source_check['radius'] = df_source_check['mcnp_filename'].apply(lambda x: get_radius(x)).astype(float)
# print(df_source_check)

# take the FWHM as the mean between x and y direction
df_source_check['FWHM_source'] = (np.abs(df_source_check['FWHM_x'])+np.abs(df_source_check['FWHM_y']))/2.0
# convert FWHM_source to mm
df_source_check['FWHM_source'] = df_source_check['FWHM_source'] * 10

# drop unnecessary columns
df_source_check = df_source_check.drop(columns=['Unnamed: 0', 'mcnp_filename', 'FWHM_x', 'FWHM_y'])
# print(df_source_check)


# load the simulated FWHM 
master_folder = '//fs03/LTH_Neutimag/hkromer/10_Experiments/02_MCNP/neutron_emitting_spot/2_case_files/_experiment_gauss_20180904/'
fname = '{}/df_MCNP_FWHM.csv'.format(master_folder)
df_mcnp_sim = pd.read_csv(fname)

# rename the FWHM to FWHM_detector to distinguish between the source and detector 
df_mcnp_sim['FWHM_detector'] = df_mcnp_sim['FWHM']

# the radius is the identifying parameter
df_mcnp_sim['radius'] = df_mcnp_sim['diameter'] / 2.0
# convert back to cm
df_mcnp_sim['radius'] = df_mcnp_sim['radius'] / 10.0

# drop unnecessary columns
df_mcnp_sim = df_mcnp_sim.drop(columns=['Unnamed: 0', 'case', 'diameter', 'FWHM'])
# print(df_mcnp_sim)


# join the two dataframes
df = pd.merge(df_mcnp_sim, df_source_check, on='radius')
# convert radius to mm
df['radius'] = df['radius'] * 10
print(df)





# plot source definition FWHM vs detected FWHM
fig = plt.figure(figsize=(8,5))
ax1 = fig.add_subplot(1, 1, 1)

X = df['FWHM_detector']
Y = df['FWHM_source']
ax1.scatter(X, Y, marker='o', color='darkblue')
# X = df['FWHM_mean']
# Y = df['diameter']
# ax1.scatter(X, Y, marker='o', color='red')

ax1.tick_params('x', colors='black', labelsize=12)	
ax1.tick_params('y', colors='black', labelsize=12)	
# grid
ax1.grid(b=True, which='major', linestyle='-')#, color='gray')
ax1.grid(b=True, which='minor', linestyle='--')#, color='gray')

ax1.set_xlabel(r'FWHM of LSF in detector [mm]')
ax1.set_ylabel(r'FWHM of source definition [mm]')
# plt.title('Diameter vs FWHM for case {}'.format(case))
# ylims = ax1.get_ylim()
# ax1.set_ylim(0, 1e7)
# fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.18)
plt.xlim(0,3)
plt.ylim(0,7)
plt.tight_layout()
# plt.show()
plt.savefig('{}/FWHM_LookupTable.png'.format(master_folder), dpi=300)
plt.clf()
plt.close('all')

# add the larger radius from run from 2018-09-11 to the dataframe
larger_folder = '//fs03/LTH_Neutimag/hkromer/10_Experiments/02_MCNP/neutron_emitting_spot/2_case_files/_experiment_gauss_20180911/'
df_larger = pd.read_csv(f'{larger_folder}/df_LookupTable.csv', index_col=0)

df = df.append(df_larger)


# plot source definition FWHM vs detected FWHM
fig = plt.figure(figsize=(8,5))
ax1 = fig.add_subplot(1, 1, 1)

X = df['FWHM_detector']
Y = df['FWHM_source']
ax1.scatter(X, Y, marker='o', color='darkblue')
# X = df['FWHM_mean']
# Y = df['diameter']
# ax1.scatter(X, Y, marker='o', color='red')

ax1.tick_params('x', colors='black', labelsize=12)	
ax1.tick_params('y', colors='black', labelsize=12)	
# grid
ax1.grid(b=True, which='major', linestyle='-')#, color='gray')
ax1.grid(b=True, which='minor', linestyle='--')#, color='gray')

ax1.set_xlabel(r'FWHM of LSF in detector [mm]')
ax1.set_ylabel(r'FWHM of source definition [mm]')
# plt.title('Diameter vs FWHM for case {}'.format(case))
# ylims = ax1.get_ylim()
# ax1.set_ylim(0, 1e7)
# fig.subplots_adjust(left=0.15, right=0.97, top=0.88, bottom=0.18)
plt.xlim(0,3)
plt.ylim(0,7)
plt.tight_layout()
# plt.show()
plt.savefig('{}/FWHM_LookupTable_with_larger_diameter.png'.format(master_folder), dpi=300)
plt.clf()
plt.close('all')
print(df)