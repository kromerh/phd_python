#!/Users/hkromer/anaconda3/bin/python3

import numpy as np
import pandas as pd
import pymysql
import sqlalchemy as sql
import matplotlib.pyplot as plt
import matplotlib.dates as md
import seaborn as sns
sns.set()
import getopt
import sys
import datetime
from scipy.interpolate import interp1d
from NGLiveDataObject import NGLiveDataObject


#*******************************
# CONSTANTS
#*******************************
# connection to database
CREDENTIALS_FILE = '/Users/hkromer/02_PhD/01.github/dash_NG/credentials.pw'
DB = "NG_twofast_DB" # name of the database
HOST = "twofast-RPi3-0"  # database host

# LOOKUP TABLES
LUT_PRESSURE_ION_SOURCE = "/Users/hkromer/02_PhD/01.github/phd/01_neutron_generator_contol/LUT_pressure_ion_source.txt"

#*******************************
# Get day from the command line!
#*******************************
# day = ... # day to query
# t_plot_start = ... # start time when to plot HH:MM:SS
# t_plot_end = ... # end time when to plot HH:MM:SS
# output_path = ... # where to store the plots

def main(day, t_plot_start, t_plot_end, output_path):

	# read password and user to connect to database
	credentials = pd.read_csv(CREDENTIALS_FILE, header=0)
	user = credentials['username'].values[0]
	pw = credentials['password'].values[0]


	#*******************************
	# GET DATA
	#*******************************
	# connect to DB
	con = NGLiveDataObject(host=HOST, database=DB, user=user, password=pw)

	# get HV and dose
	query = "SELECT * FROM HBox_Uno"
	data_HV_full = con.get_from_database(query=query)
	# for plotting, scale the current
	data_HV_full['HV_current_x100'] = data_HV_full['HV_current'] * 100.0
	# convert dose voltage to dose
	data_HV_full['dose'] = data_HV_full['dose_voltage'] * 3000 / 5.5

	# get pressure
	query = "SELECT * FROM BBox"
	data_pressure_full = con.get_from_database(query=query)
	# correct pressure
	data_pressure_full = con.correct_pressure(LUT_PRESSURE_ION_SOURCE, data_pressure_full)

	print(f"Loaded {data_HV_full.shape} from HBox_Uno.")
	print(f"Loaded {data_pressure_full.shape} from BBox.")
	print(f"Saving plots to folder {output_path} for day {day} between {t_plot_start} and {t_plot_end}.")

	#*******************************
	# PLOT: pressure and high voltage
	#*******************************

	# select a subset of the day
	time_start = f'{day} {t_plot_start}'
	time_end = f'{day} {t_plot_end}'

	df_hv = data_HV_full.loc[time_start:time_end,:]

	df_pressure = data_pressure_full.loc[time_start:time_end,:]

	fig, ax = plt.subplots(figsize=(15,6))

	sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
	sns.lineplot(x=df_pressure.index, y='pressure_IS_corrected', data=df_pressure, color='darkgreen', ax=ax, label='pressure')
	plt.legend(loc='upper left')

	ax2 = ax.twinx()
	sns.lineplot(x=df_hv.index, y='HV_voltage', data=df_hv, ax=ax2, color='darkred', label='voltage [-kV]')
	sns.lineplot(x=df_hv.index, y='HV_current_x100', data=df_hv, ax=ax2, color='darkorange', label='current [x0.01 mA]')
	plt.legend(loc='upper right')

	ax.set(yscale="log")
	ax.set_ylabel('Pressure [mbar]')
	ax2.set_ylabel('HV')

	ax.set_xlabel(f'Time, {day}')
	ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))
	ax.set_ylim(1e-6, 1e-3)

	plt.savefig(f'{output_path}{day}_pressure_and_hv.png', dpi=600, format='png')
	plt.close()

	#*******************************
	# PLOT: dose and high voltage
	#*******************************

	# select a subset of the day
	time_start = f'{day} {t_plot_start}'
	time_end = f'{day} {t_plot_end}'

	df_hv = data_HV_full.loc[time_start:time_end,:]

	df_pressure = data_pressure_full.loc[time_start:time_end,:]

	fig, ax = plt.subplots(figsize=(15,6))

	sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
	sns.lineplot(x=df_hv.index, y='dose', data=df_hv, color='darkblue', ax=ax, label='dose')
	plt.legend(loc='upper left')

	ax2 = ax.twinx()
	sns.lineplot(x=df_hv.index, y='HV_voltage', data=df_hv, ax=ax2, color='darkred', label='voltage [-kV]')
	sns.lineplot(x=df_hv.index, y='HV_current_x100', data=df_hv, ax=ax2, color='darkorange', label='current [x0.01 mA]')
	plt.legend(loc='upper right')

	ax.set_ylabel('Dose [muSv/hr]')
	ax2.set_ylabel('HV')

	ax.set_xlabel(f'Time, {day}')
	ax.xaxis.set_major_formatter(md.DateFormatter('%H:%M:%S'))

	plt.savefig(f'{output_path}{day}_dose_and_hv.png', dpi=600, format='png')
	plt.close()


if __name__ == '__main__':
	# Get the arguments from the command-line except the filename
	argv = sys.argv[1:]

	try:
		if len(argv) == 3:
			# day = argv[0]
			day = datetime.datetime.now().date()
			t_plot_start = argv[0]
			t_plot_end = argv[1]
			output_path = argv[2]
			main(day, t_plot_start, t_plot_end, output_path)
		else:
			print('Error! usage: 001.ETL.from_live_table.py.py t_plot_start t_plot_end output_path')
			sys.exit(2)

	except getopt.GetoptError:
		# Print something useful
		print('usage: 001.ETL.from_live_table.py.py t_plot_start t_plot_end output_path')
		sys.exit(2)

