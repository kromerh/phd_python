import datetime
import sys
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import numpy as np
import pymysql
import time
import re
import matplotlib.animation as animation
from matplotlib.ticker import AutoMinorLocator
pd.set_option("display.max_columns", 100)


def fun_readDoseFromDB(start_date, end_date):
	t0 = time.time()
	db = pymysql.connect(host="twofast-RPi3-0",  # your host
						 user="doseReader",  # username
						 passwd="heiko",  # password
						 db="NG_twofast_DB")  # name of the database
	# Create a Cursor object to execute queries.
	cur = db.cursor()

	try:
		# cur.execute("""INSERT INTO pressure (pressure_IS, pressure_VC) VALUES (%s, %s)""",(press_IS,press_VC))
		cur.execute("SELECT * FROM data_dose WHERE time BETWEEN \"{}\" AND \"{}\" ORDER BY id ASC".format(start_date, end_date))
		# cur.execute("""SELECT * FROM pressure""")
		rows = cur.fetchall()
		df = pd.DataFrame( [[ij for ij in i] for i in rows] )
		# voltage_dose, counts_WS, counts_BS, counts_DF
		df.rename(columns={0: 'id', 1: 'time', 2: 'dose', 3: 'dose_voltage', 4: 'dose_corrected'}, inplace=True)

		df = df.set_index(['id'])

	except:
		cur.rollback()

	cur.close()

	return df

# dose from database

# start of measurement
start_date = "2018-07-25 16:45:00"
my_date = re.findall(r'(\d\d\d\d-\d\d-\d\d) ', start_date)[0]
# end of measurement
end_date = "2018-07-25 18:53:36"

df_dose = fun_readDoseFromDB(start_date, end_date)
df_dose['t'] = pd.to_datetime(df_dose['time']).dt.time
print(df_dose.head())


# edge position from file
edge_file = '//fs03/LTH_Neutimag/hkromer/08_Data/NG_control/2018-07-25/2018-07-25_edge_pos.txt'

df_edge = pd.read_csv(edge_file, delimiter=r'\t+')

# datetime object
df_edge['t'] = pd.to_datetime(df_edge['Time'], format='%H:%M:%S').dt.time


# detector count data
cnt_file = '//fs03/LTH_Neutimag/hkromer/10_Experiments/08_EmittingSpotSizeMeasurement/2018-07-25_findZero/2018-07-25_readout.csv'

df_cnt = pd.read_csv(cnt_file)
# datetime object
df_cnt['t'] = pd.to_datetime(df_cnt['time']).dt.time
# print(df_cnt.head(5))


# find for each edge position timestamp the next timestamp in count dataframe (when the edge was moved and counts can be trusted)
def getStartTimeCount(x, s_cnt_time):
	my_s = s_cnt_time[ s_cnt_time > x ].values[0]
	return my_s
df_edge['cnt_start'] = df_edge['t'].apply(lambda x: getStartTimeCount(x, df_cnt['t']))

# end time, do a shift forst
df_edge['t_shift'] = df_edge['t'].shift(-1)
def getEndTimeCount(x, s_cnt_time):
	try:
		my_s = s_cnt_time[ s_cnt_time < x ].values[-1]
	except:
		my_s = s_cnt_time.values[0]
	return my_s

df_edge['cnt_end'] = df_edge['t_shift'].apply(lambda x: getEndTimeCount(x, df_cnt['t']))

# main dataframe
df = pd.DataFrame()
df['edge_pos_mm'] = df_edge['edge_pos [mm]']
df['t_cnt_start'] = df_edge['cnt_start']
df['t_cnt_end'] = df_edge['cnt_end']

# time difference
a = df['t_cnt_start']
b = df['t_cnt_end']
c = (pd.to_timedelta(b.astype(str)) - pd.to_timedelta(a.astype(str))).dt.seconds

df['d_t'] = c

def sumDetectorCounts(t_start, t_end, df_cnt):
	my_df = df_cnt[ (df_cnt['t'] >= t_start) & (df_cnt['t'] <= t_end)]
	# print(t_start, t_end)
	# print(my_df)
	# print('*** newline ***')
	counts = my_df['value'].values.astype(float)
	my_sum = np.sum(counts)
	
	return my_sum

df['sum_cnts'] = df.apply(lambda x: sumDetectorCounts(x['t_cnt_start'], x['t_cnt_end'], df_cnt), axis=1)


# average dose
def getAverageDose(t_start, t_end, df_dose):
	my_df = df_dose[ (df_dose['t'] >= t_start) & (df_dose['t'] <= t_end)]
	# print(t_start, t_end)
	# print(my_df[['t','dose_corrected']])
	# print('*** newline ***')
	avg_dose = np.mean(my_df['dose_corrected'])
	# plt.plot(my_df.index, my_df['dose_corrected'].values)
	# plt.show()
	return avg_dose

df['avg_dose'] = df.apply(lambda x: getAverageDose(x['t_cnt_start'], x['t_cnt_end'], df_dose), axis=1)

# counts per second
df['cps'] = df['sum_cnts'] / df['d_t']

# normalized with avg_dose
df['norm_cps'] = df['cps'] / df['avg_dose']

# plot norm cps vs position
print(df)
f, ax = plt.subplots()
X=df['edge_pos_mm'].values[0:-1]
Y=df['norm_cps'].values[0:-1]
ax.scatter(X,Y)
ax.set_xlabel('Edge position [mm]')
ax.set_ylabel('Normalized counts')
ax.set_xticks(np.arange(0,20+0.5,0.5))
# minor ticks x
minor_locator = AutoMinorLocator(2)
ax.xaxis.set_minor_locator(minor_locator)
plt.title('Edge spread function for measurement from \n {}'.format(my_date))
plt.tight_layout()
plt.grid(True)
plt.show()


