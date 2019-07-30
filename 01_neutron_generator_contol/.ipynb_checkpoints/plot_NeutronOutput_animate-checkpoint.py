import matplotlib.pyplot as plt
import numpy as np
import pymysql
import time
import pandas as pd
import datetime
from scipy.interpolate import interp1d
import matplotlib.animation as animation
from matplotlib.ticker import AutoMinorLocator

# compute the output
def compute_output(HV, dose):
	try:
		# neutronOutputPer100muSv = interp_output(HV)
		neutronOutputPer100muSv = interp_output(HV)
	except ValueError:
		# take 80 kV if the HV is not in the interpolation range
		neutronOutputPer100muSv = lst_output[0]

	return (dose/100) * neutronOutputPer100muSv 

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


# correct the dose that the arduino reads. This is done using the dose_lookup_table which relates the pi dose with the displayed dose.
df_LT = pd.read_csv("//fs03/LTH_Neutimag/hkromer/02_Simulations/01_Python/MCNP_neutron_output/dose_lookup_table.txt", delimiter="\t")

# interpolation function
interp_dose = interp1d(df_LT['dose_pi'], df_LT['dose_display'], kind='cubic')

# Neutron output per 100 µSv/h for some HV's
lst_HV = [80,85,90,95,100,105,110]
lst_output = [getNeutronOutputPer100muSv(val, 70)[0] for val in lst_HV]
# lst_output = [getNeutronOutputPer100muSv(80, 70)[0] for val in lst_HV]
# print(lst_output)
# interpolation function
interp_output = interp1d(lst_HV, lst_output, kind='cubic')

def fun_readDoseFromDB():
	t0 = time.time()
	db = pymysql.connect(host="twofast-RPi3-0",  # your host
						 user="doseReader",  # username
						 passwd="heiko",  # password
						 db="NG_twofast_DB")  # name of the database
	# Create a Cursor object to execute queries.
	cur = db.cursor()

	try:
		print('Starting Neutron Output plot...')
		# cur.execute("""INSERT INTO pressure (pressure_IS, pressure_VC) VALUES (%s, %s)""",(press_IS,press_VC))
		cur.execute("""SELECT * FROM HBox_Uno ORDER BY id DESC LIMIT 300""")
		# cur.execute("""SELECT * FROM pressure""")
		rows = cur.fetchall()
		df = pd.DataFrame( [[ij for ij in i] for i in rows] )
		# voltage_dose, counts_WS, counts_BS, counts_DF
		df.rename(columns={0: 'ID', 1: 'date', 2: 'dose_voltage', 3: 'HV_current', 4: 'HV_voltage'}, inplace=True)

		df = df.set_index(['ID'])
		df['dose'] = df['dose_voltage'] * 3000 / 5.5
	
		df['dose_corr'] = interp_dose(df['dose'])




		df['output'] = np.vectorize(compute_output)(df['HV_voltage'], df['dose_corr'])

		# normalize to 1e6
		# print(df.output)
		df['output'] = df['output'] 
		df.output[ df.dose_corr < 50 ] = 0

		t1 = time.time()

		total = t1 - t0


	except:
		cur.rollback()

	cur.close()

	return df




fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

# cnt=0
# dose = []



line,  = ax1.plot([0], [1e5], lw=1, color='red')  # real data
text = ax1.text(0.85, 0.80, "", transform=ax1.transAxes, ha="right", va="top", color='red')
text_time = ax1.text(0.92, 0.07, "", transform=ax1.transAxes, ha="right", va="top", color='black')

plt.ylim(1e5,1e8)
plt.xlim(0,300)
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, which='minor', linestyle='--')
minor_locator = AutoMinorLocator(2)
ax1.xaxis.set_minor_locator(minor_locator)
minor_locator = AutoMinorLocator(5)
ax1.yaxis.set_minor_locator(minor_locator)
# ax1.set_yticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
ax1.text(0.40, 0.80, "Neutron output:", transform=ax1.transAxes, ha="right", va="top", color='red', fontsize=20)
ax1.text(0.50, 0.07, "Time:", transform=ax1.transAxes, ha="right", va="top", color='black')
ax1.set_yscale("log")
plt.subplots_adjust(left=0.1, right=0.98, top=0.94)
plt.xlabel('Last 300 readings')
plt.ylabel('Neutron output [n/s]')
plt.title('Neutron output via LB6411 at 70 cm W')

def animateinit(): #tells our animator what artists will need re-drawing every time
	# return line,text
	return line, text, text_time
	# return line_IS, line_VC

def animate(i):
	df = fun_readDoseFromDB()
	# print(df.head())

	line.set_data(range( len(df['output']) ) , df['output'])
	txt = str(round(df['output'].iloc[0]/1e6)) + 'x1e6 n/s'
	text.set_text(txt)
	text.set_fontsize(20)
	text_time.set_text(df['date'].iloc[0])
	ax1.set_yscale("log")
	# print(df['output'])
	# print(text_IS, text_VC)
	return line, text, text_time #return the updated artists
	# return line_IS, line_VC #return the updated artists


#inform the animator what our init_func is and enable blitting
ani = animation.FuncAnimation(fig, animate, interval=1000,init_func=animateinit, blit=True)
plt.show()
