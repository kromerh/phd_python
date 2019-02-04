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


timeStart = "2018-07-08 12:00"
timeEnd = "2018-07-11 13:00"
timeThreshold = 0  # timestamps (minutes) to skip for the plot
dateStamp = re.findall(r'(\d\d\d\d-\d\d-\d\d)',timeStart)[0]


# Experiment from 2017-12-21
db = pymysql.connect(host="twofast-RPi3-0",  # your host
					 user="doseReader",  # username
					 passwd="heiko",  # password
					 db="NG_twofast_DB")  # name of the database
# Create a Cursor object to execute queries.
cur = db.cursor()

try:
	# pressure
	sql = "SELECT * FROM data_pressure WHERE data_pressure.time BETWEEN \"{}\" AND \"{}\" ".format(timeStart, timeEnd)
	# print(sql)
	cur.execute(sql)
	rows = cur.fetchall()
	df = pd.DataFrame( [[ij for ij in i] for i in rows] )
	# voltage_dose, counts_WS, counts_BS, counts_DF
	df.rename(columns={0: 'ID', 1: 'time', 2: 'pressure_IS', 3: 'pressure_VC', 4: 'voltage_IS', 5: 'voltage_VC', 6: 'pressure_IS_corrected'}, inplace=True)
	# print(df_HV.columns)
	df = df.set_index(['time'])

except:
	cur.rollback()

cur.close()

# # average data over 1 minutes
df_pr = pd.DataFrame()
df_pr['pressure_IS'] = df['pressure_IS'].resample('1Min').mean()
df_pr['pressure_VC'] = df['pressure_VC'].resample('1Min').mean()

print(df_pr)
# -------------------------------------------------------------------
# plot pressure
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
df_pr = df_pr.reset_index()
# print(df)
df1 = df_pr[ df_pr.index > timeThreshold ] 
ax1.plot(df1.index, df1['pressure_VC'], 'o',color="darkorange", markersize=5,  alpha=0.65, markeredgewidth=1.5, markeredgecolor='darkorange')

plt.ylim(0.000006,0.000009)
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

ax1.set_ylabel(r'\textbf{pressure VC [mbar]}', fontsize=12)
ax1.set_xlabel(r'\textbf{Timestamp [minutes]}', fontsize=12, labelpad=2)

ax1.grid(b=True, which='major', linestyle='-')#, color='gray')
ax1.grid(b=True, which='minor', linestyle='--')#, color='gray')



ax2 = ax1.twinx()
df1 = df_pr[ df_pr.index > timeThreshold ] 
ax2.plot(df1.index, df1['pressure_IS'], 'x',color="darkblue", markersize=5,  alpha=0.65, markeredgewidth=1.5, markeredgecolor='darkblue')
ax2.set_ylabel(r'\textbf{pressure IS [mbar]}', fontsize=12, color='darkblue')
ax2.tick_params('y', colors='darkblue', labelsize=12)	
ax2.set_ylim(0.045,0.055)
plt.tight_layout()
filename =  '{}_press_vs_timestamp'.format(dateStamp)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)

# plt.show()

