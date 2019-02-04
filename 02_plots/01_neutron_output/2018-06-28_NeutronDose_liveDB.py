import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import pymysql
import datetime
from scipy.interpolate import interp1d
import os
import re
import sys
from matplotlib.ticker import AutoMinorLocator
sys.path.append('//fs03/LTH_Neutimag/hkromer/02_Simulations/01_Python/MCNP_neutron_output/')


timeStart = "2018-07-11 11:00"
timeEnd = "2018-07-11 16:00"
dateStamp = re.findall(r'(\d\d\d\d-\d\d-\d\d)',timeStart)[0]


# Experiment from 2017-12-21
db = pymysql.connect(host="twofast-RPi3-0",  # your host
					 user="doseReader",  # username
					 passwd="heiko",  # password
					 db="NG_twofast_DB")  # name of the database
# Create a Cursor object to execute queries.
cur = db.cursor()

try:
	# HIGH VOLTAGE
	sql = "SELECT * FROM HBox_Uno WHERE HBox_Uno.time BETWEEN \"{}\" AND \"{}\" ".format(timeStart, timeEnd)
	# print(sql)
	cur.execute(sql)
	rows = cur.fetchall()
	df_HV = pd.DataFrame( [[ij for ij in i] for i in rows] )
	# voltage_dose, counts_WS, counts_BS, counts_DF
	df_HV.rename(columns={0: 'ID', 1: 'time', 2:'dose_voltage', 3: 'HV_current', 4: 'HV_voltage'}, inplace=True)
	# print(df_HV.columns)
	df_HV = df_HV.set_index(['time'])

	# DOSE
	sql = "SELECT * FROM HBox_Uno WHERE HBox_Uno.time BETWEEN \"{}\" AND \"{}\" ".format(timeStart, timeEnd)
	cur.execute(sql)
	rows = cur.fetchall()
	df_dose = pd.DataFrame( [[ij for ij in i] for i in rows] )
	# voltage_dose, counts_WS, counts_BS, counts_DF
	df_dose.rename(columns={0: 'ID', 1: 'time', 2:'dose_voltage', 3: 'HV_current', 4: 'HV_voltage'}, inplace=True)

	df_dose = df_dose.set_index(['time'])

except:
	cur.rollback()

cur.close()

"""
1.) Correct the read dose with the interpolation function 
"""

# correct the dose that the arduino reads. This is done using the dose_lookup_table which relates the pi dose with the displayed dose.
df_LT = pd.read_csv("//fs03/LTH_Neutimag/hkromer/02_Simulations/01_Python/MCNP_neutron_output/dose_lookup_table.txt", delimiter="\t")

# interpolation function
interp_dose = interp1d(df_LT['dose_pi'], df_LT['dose_display'], kind='cubic')
df_dose['dose'] = df_dose['dose_voltage'] * 3000 / 5.5
df_dose['dose_corr'] = interp_dose(df_dose['dose'])

"""
2.) Bring the dataframes onto the same timestep by averaging over 1 minute intervals
"""


# # average data over 1 minutes
df_m_HV = pd.DataFrame()
df_m_HV['HV_voltage'] = df_HV['HV_voltage'].resample('1Min').mean()
df_m_HV['HV_current'] = df_HV['HV_current'].resample('1Min').mean()

# compute beam power
# assumption on the leakage current:
leakage_current = 0.15
df_m_HV['beam_power'] = df_m_HV['HV_current'] * df_m_HV['HV_voltage']
df_m_HV['beam_power_leakage_current'] = (df_m_HV['HV_current']-leakage_current) * df_m_HV['HV_voltage']
# df_m_HV.to_csv('HV_m.csv')

df_m_dose = pd.DataFrame()
df_m_dose['dose_corr'] = df_dose['dose_corr'].resample('1Min').mean()
# df_m_dose.to_csv('dose_m.csv')

# combine dataframes
df = pd.merge(df_m_HV, df_m_dose, left_index=True, right_index=True)
# df.to_csv('df.csv')

"""
3.) Compute the neutron yield from the dose
"""

def getNeutronOutputPer100muSv(HV=100, LB6411_distance=70):
	"""
	Retrieves the neutron output per 100µSv/h as determined from MCNP. Only works for the new target. Returns that value
	HV: High voltage. This determines which MCNP run is taken to load the data. Default is -100 kV
	LB6411_distance: Distance between the source and LB6411 position. Default is 70 cm
	"""
	# list of the HV's simulated in MCNP
	lst_HV = [80,85,90,95,100,105,110]
	# list of the ID's for the respective MCNP simulation
	lst_ID = [230,231,232,233,234,235,236]
	# find index of the HV in the lst_HV
	try:
		idx = lst_HV.index(HV)
	except ValueError:
		idx = -1

	if idx == -1:
		print('--- High voltage value of ' + str(HV) + ' is not in an MCNP run. Exit SCRIPT. --- ')
		sys.exit()
	else:
		# TODO: change this into a common directory and select from the HV, maybe as an interpolation
		path_to_MCNP_OutputPer100muSv = '//fs03//LTH_Neutimag//hkromer//10_Experiments//02_MCNP//CurrentTarget%s/CurrentTarget%s_normal/' % (lst_ID[idx], lst_ID[idx])
		csv_name = 'df_neutron_output_for_Edeut_%s.csv' % HV
		df = pd.read_csv(path_to_MCNP_OutputPer100muSv + csv_name, header=0)
		
		distance = LB6411_distance

		neutronOutputPer100muSv = df.W[ df.distance == distance ].values
		
		return neutronOutputPer100muSv

# Neutron output per 100 µSv/h for some HV's
lst_HV = [80,85,90,95,100,105,110]
lst_output = [getNeutronOutputPer100muSv(val, 70)[0] for val in lst_HV]
# lst_output = [getNeutronOutputPer100muSv(80, 70)[0] for val in lst_HV]
# print(lst_output)
# interpolation function
interp_output = interp1d(lst_HV, lst_output, kind='cubic')

# compute the output
def compute_output(HV, dose):
	try:
		# neutronOutputPer100muSv = interp_output(HV)
		neutronOutputPer100muSv = interp_output(HV)
	except ValueError:
		# take 80 kV if the HV is not in the interpolation range
		neutronOutputPer100muSv = lst_output[0]

	return (dose/100) * neutronOutputPer100muSv 

df['output'] = np.vectorize(compute_output)(df['HV_voltage'], df['dose_corr'])


# df.to_csv('df.csv')

"""
4.) Compute maximum theoretical neutron yield and fraction
"""

# import the neutron yield dataframe from the thick target calculation

df_neutronYield = pd.read_csv('//fs03/LTH_Neutimag/hkromer/02_Simulations/01_Python/thick_target_yield_estimation/neutron_yield.csv')

# energy dEdx xs dYdE Y [total neutron yield per mA and per s]
interp_neutronYield = interp1d(df_neutronYield['energy'], df_neutronYield['Y'], kind='cubic')

def calculate_theoretical_output(HV):
	if HV < df_neutronYield['energy'].iloc[0]:
		HV = df_neutronYield['energy'].iloc[0]
	
	return interp_neutronYield(HV)

# output in n/s/mA
df['output_theoretical_per_mA'] = df['HV_voltage'].apply(calculate_theoretical_output)

# output in n/s using the current
df['output_theoretical'] = df['output_theoretical_per_mA'] * df['HV_current']
# assuming 0.2 mA leakage current
df['output_theoretical_leakage_current'] = df['output_theoretical_per_mA'] * (df['HV_current']-leakage_current)

# output over theoretical output
df['output_fraction'] = df['output'] / df['output_theoretical']
df['output_fraction_leakage_current'] = df['output'] / df['output_theoretical_leakage_current']

# df.to_csv('df.csv')






# -------------------------------------------------------------------
# plot output
# -------------------------------------------------------------------

plt.rc('text', usetex=True)
plt.rc('font', weight='bold')
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
matplotlib.rcParams['mathtext.it'] = 'Arial:italic'
matplotlib.rcParams['mathtext.bf'] = 'Arial:bold'
matplotlib.rcParams['mathtext.tt'] = 'Arial'
matplotlib.rcParams['mathtext.cal'] = 'Arial'
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

fig = plt.figure(figsize=(8*0.60,5*0.60))

# axis 1
ax1 = fig.add_subplot(1, 1, 1)
# Hide the right and top spines
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)

# df['mytime'] = df.index
# df['hours'] = df.mytime.dt.strftime('%H:%M')
df = df.reset_index()
# print(df)
# df1 = df[ df.index > 80 ] 
ax1.plot(df.index, df['HV_voltage'], 'o',color="darkorange", markersize=5,  alpha=0.65, markeredgewidth=1.5, markeredgecolor='darkorange')

# plt.ylim(0.05,0.25)
# ax1.yaxis.set_ticks(np.arange(0.05,0.25+0.05,0.05))
# minor ticks x
minor_locator = AutoMinorLocator(2)
ax1.xaxis.set_minor_locator(minor_locator)
# minor ticks y
minor_locator = AutoMinorLocator(2)
ax1.yaxis.set_minor_locator(minor_locator)
# tick font size
ax1.tick_params('x', colors='black', labelsize=12)	
ax1.tick_params('y', colors='black', labelsize=12)	

ax1.set_ylabel(r'\textbf{HV [-kV]}', fontsize=12)
ax1.set_xlabel(r'\textbf{Timestamp [minutes]}', fontsize=12, labelpad=2)



ax1.grid(b=True, which='major', linestyle='-')#, color='gray')
ax1.grid(b=True, which='minor', linestyle='--')#, color='gray')


plt.tight_layout()
filename =  '{}_HV_vs_timestamp'.format(dateStamp)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)

# plt.show()


# -------------------------------------------------------------------
# plot current
# -------------------------------------------------------------------

plt.rc('text', usetex=True)
plt.rc('font', weight='bold')
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
matplotlib.rcParams['mathtext.it'] = 'Arial:italic'
matplotlib.rcParams['mathtext.bf'] = 'Arial:bold'
matplotlib.rcParams['mathtext.tt'] = 'Arial'
matplotlib.rcParams['mathtext.cal'] = 'Arial'
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

fig = plt.figure(figsize=(8*0.60,5*0.60))

# axis 1
ax1 = fig.add_subplot(1, 1, 1)
# Hide the right and top spines
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)

# df['mytime'] = df.index
# df['hours'] = df.mytime.dt.strftime('%H:%M')
df = df.reset_index()
# print(df)
# df1 = df[ df.index > 80 ] 
ax1.plot(df.index, df['HV_current'], 'o',color="red", markersize=5,  alpha=0.65, markeredgewidth=1.5, markeredgecolor='red')

# plt.ylim(0.05,0.25)
# ax1.yaxis.set_ticks(np.arange(0.05,0.25+0.05,0.05))
# minor ticks x
minor_locator = AutoMinorLocator(2)
ax1.xaxis.set_minor_locator(minor_locator)
# minor ticks y
minor_locator = AutoMinorLocator(2)
ax1.yaxis.set_minor_locator(minor_locator)
# tick font size
ax1.tick_params('x', colors='black', labelsize=12)	
ax1.tick_params('y', colors='black', labelsize=12)	

ax1.set_ylabel(r'\textbf{HV [-kV]}', fontsize=12)
ax1.set_xlabel(r'\textbf{Timestamp [minutes]}', fontsize=12, labelpad=2)



ax1.grid(b=True, which='major', linestyle='-')#, color='gray')
ax1.grid(b=True, which='minor', linestyle='--')#, color='gray')


plt.tight_layout()
filename =  '{}_I_vs_timestamp'.format(dateStamp)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)

# plt.show()




# -------------------------------------------------------------------
# plot HV
# -------------------------------------------------------------------

plt.rc('text', usetex=True)
plt.rc('font', weight='bold')
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Arial'
matplotlib.rcParams['mathtext.it'] = 'Arial:italic'
matplotlib.rcParams['mathtext.bf'] = 'Arial:bold'
matplotlib.rcParams['mathtext.tt'] = 'Arial'
matplotlib.rcParams['mathtext.cal'] = 'Arial'
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

fig = plt.figure(figsize=(8*0.60,5*0.60))

# axis 1
ax1 = fig.add_subplot(1, 1, 1)
# Hide the right and top spines
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)

# df['mytime'] = df.index
# df['hours'] = df.mytime.dt.strftime('%H:%M')
df = df.reset_index()
# print(df)
# df1 = df[ df.index > 80 ] 
ax1.plot(df.index, df['output'], 'o',color="darkblue", markersize=5,  alpha=0.65, markeredgewidth=1.5, markeredgecolor='darkblue')

# plt.ylim(0.05,0.25)
# ax1.yaxis.set_ticks(np.arange(0.05,0.25+0.05,0.05))
# minor ticks x
minor_locator = AutoMinorLocator(2)
ax1.xaxis.set_minor_locator(minor_locator)
# minor ticks y
minor_locator = AutoMinorLocator(2)
ax1.yaxis.set_minor_locator(minor_locator)
# tick font size
ax1.tick_params('x', colors='black', labelsize=12)	
ax1.tick_params('y', colors='black', labelsize=12)	

ax1.set_ylabel(r'\textbf{Neutron yield [n/s]}', fontsize=12)
ax1.set_xlabel(r'\textbf{Timestamp [minutes]}', fontsize=12, labelpad=2)



ax1.grid(b=True, which='major', linestyle='-')#, color='gray')
ax1.grid(b=True, which='minor', linestyle='--')#, color='gray')


plt.tight_layout()
# filename =  'neutron_output_vs_timestamp'
filename =  '{}_neutron_output_vs_timestamp'.format(dateStamp)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)

# plt.show()






# -------------------------------------------------------------------
# plot measured output (n/s/W) vs beam power
# -------------------------------------------------------------------

fig = plt.figure(figsize=(8*0.60,5*0.60))

# axis 1
ax1 = fig.add_subplot(1, 1, 1)
# Hide the right and top spines
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)

# take only reasonable data
# df1 = df[ ((df.index >= "21.12.2017 14:03") & (df.index <= "21.12.2017 15:06"))]
# df2 = df[((df.index >= "21.12.2017 15:15") & (df.index <= "21.12.2017 16:14"))]

ax1.plot(df['beam_power'], (df['output'] / df['beam_power']) / 1e5,  'v',color="darkorange", markersize=5, label='80', alpha=0.65, markeredgewidth=1.5, markeredgecolor='darkorange')
ax1.plot(df['beam_power'], (df['output'] / df['beam_power']) / 1e5,  'o',color="darkblue", markersize=5, label = '90', alpha=0.65, markeredgewidth=1.5, markeredgecolor='darkblue')
plt.xlim(60,85)
plt.ylim(1.0,3.5)
# minor ticks x
minor_locator = AutoMinorLocator(2)
ax1.xaxis.set_minor_locator(minor_locator)
# minor ticks y
minor_locator = AutoMinorLocator(2)
ax1.yaxis.set_minor_locator(minor_locator)
# tick font size
ax1.tick_params('x', colors='black', labelsize=12)	
ax1.tick_params('y', colors='black', labelsize=12)	
ax1.set_ylabel(r'\textbf{Neutron yield [$\cdot 10^5$ n/s/W]}', fontsize=12)
ax1.set_xlabel(r'\textbf{Beam power [W]}', fontsize=12, labelpad=2)

# ax1.text(70, 1e7, r"Data averaged over 1 min"  "\n"  r"ca 250 rpm" "\n" r"0.4 - 1.0 mA (DF 1 - 60 \% )",  bbox={'facecolor':'white', 'alpha':0.9, 'pad':10})

ax1.grid(b=True, which='major', linestyle='-')#, color='gray')
ax1.grid(b=True, which='minor', linestyle='--')#, color='gray')
l1 = plt.legend(loc="best",  fontsize=10)
l1.set_title(r"High voltage [-kV]", prop = {'size': 10})

plt.tight_layout()

filename =  '{}act_nsW_vs_beam_power_noLeakageCurrent'.format(dateStamp)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)

# plt.show()