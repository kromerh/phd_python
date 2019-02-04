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
start_date = "2018-07-26 09:50:00"
my_date = re.findall(r'(\d\d\d\d-\d\d-\d\d) ', start_date)[0]
# end of measurement
end_date = "2018-07-26 15:57:17"

# When the background measurement started and ended in a list
lst_bg_start = ['08:10:00']
lst_bg_end = ['08:14:00']

df_dose = fun_readDoseFromDB(start_date, end_date)
df_dose['t'] = pd.to_datetime(df_dose['time']).dt.time
# print(df_dose.head())


# edge position from file
edge_file = '//fs03/LTH_Neutimag/hkromer/10_Experiments/08_EmittingSpotSizeMeasurement/{}_measurement/{}_edge_position.txt'.format(my_date, my_date)

df_edge = pd.read_csv(edge_file, delimiter=r'\t+')

# datetime object
df_edge['t'] = pd.to_datetime(df_edge['Time'], format='%H:%M:%S').dt.time
print(df_edge.head(5))

# detector count data
cnt_file = '//fs03/LTH_Neutimag/hkromer/10_Experiments/08_EmittingSpotSizeMeasurement/{}_measurement/{}_readout.csv'.format(my_date, my_date)

df_cnt = pd.read_csv(cnt_file)
# datetime object
df_cnt['t'] = pd.to_datetime(df_cnt['time']).dt.time
print(df_cnt.head(5))


# background
# input start and end of background measurement in a list 
bg_cnts = np.zeros(len(lst_bg_start))  # background counts

ii = 0
for bg_start, bg_end in zip(lst_bg_start, lst_bg_end):

	bg_start = datetime.datetime.strptime(bg_start, '%H:%M:%S').time()
	bg_end = datetime.datetime.strptime(bg_end, '%H:%M:%S').time()
	this_df = df_cnt[ (df_cnt['t'] >= bg_start) & (df_cnt['t'] <= bg_end ) ]
	cnts = np.sum(this_df['value'].values.astype(float))
	t = this_df['t']
	# convert back to string and then to datetime object to compute differences
	t1 = t.values[0]
	t1 = datetime.time.strftime(t1, '%H:%M:%S')
	t2 = t.values[-1]
	t2 = datetime.time.strftime(t2, '%H:%M:%S')

	FMT = '%H:%M:%S'
	tdelta = (datetime.datetime.strptime(t2, FMT) - datetime.datetime.strptime(t1, FMT)).seconds
	cnts_norm = cnts/tdelta

	bg_cnts[ii] = cnts_norm

	ii = ii + 1

print('Background counts per second: {}'.format(bg_cnts))  # check this first, then decide to make average
bg_cnt = np.mean(bg_cnts)
print(f'Mean background counts per second: {bg_cnt} 1/s')

# pay special attention to each file, maybe make an ESF for each day!

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
# print(df_edge.head(20))


# divide into two blocks because measurement was interrupted

# main dataframe
df = pd.DataFrame()
df['edge_pos_mm'] = df_edge['edge_pos [mm]']
df['t_cnt_start'] = df_edge['cnt_start']
df['t_cnt_end'] = df_edge['cnt_end']
df['t'] = df_edge['t']

df1 = df[ df['t'] < datetime.datetime.strptime('11:09:30', FMT).time() ]
df2 = df[ df['t'] > datetime.datetime.strptime('11:28:26', FMT).time() ]
# print(df1.head(20))
# print(df2.tail(20))


def plotXY(X,Y,xlbl,ylbl,title,fname):
	f, ax = plt.subplots()
	plt.title(title)
	ax.plot(X,Y)
	ax.scatter(X,Y)
	ax.set_xlabel(xlbl)
	ax.set_ylabel(ylbl)
	f.autofmt_xdate()
	plt.grid(True)
	plt.tight_layout()
	filename =  f'//fs03/LTH_Neutimag/hkromer/10_Experiments/08_EmittingSpotSizeMeasurement/{my_date}_measurement/{fname}'
	plt.savefig(filename + '.png', dpi=600)
	plt.close('all')
	pass

# # time difference
df_result = pd.DataFrame()
for df in [df1, df2]:
	a = df['t_cnt_start']
	b = df['t_cnt_end']
	c = (pd.to_timedelta(b.astype(str)) - pd.to_timedelta(a.astype(str))).dt.seconds

	df['d_t'] = c

	def sumDetectorCounts(t_start, t_end, df_cnt):
		my_df = df_cnt[ (df_cnt['t'] >= t_start) & (df_cnt['t'] <= t_end)]
		str_start = datetime.time.strftime(t_start, '%H:%M:%S').replace(':','-')
		# plotXY(my_df['t'].values.astype(str), my_df['value'].values.astype(float), 'time', 'counts in detector', f'Counts in detector \n at starttime {t_start}', f'CountsDet_{str_start}')
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
		
		str_start = datetime.time.strftime(t_start, '%H:%M:%S').replace(':','-')
		# plotXY(my_df['t'].values.astype(str), my_df['dose_corrected'].values.astype(float), 'time', 'Dose corrected', f'Dose \n at starttime {t_start}', f'dose_{str_start}')
		# print(t_start, t_end)
		# print(my_df[['t','dose_corrected']])
		# print('*** newline ***')
		avg_dose = np.mean(my_df['dose_corrected'])
		# plt.plot(my_df.index, my_df['dose_corrected'].values)
		# plt.show()
		return avg_dose

	df['avg_dose'] = df.apply(lambda x: getAverageDose(x['t_cnt_start'], x['t_cnt_end'], df_dose), axis=1)



	# take background counts into account
	df['bg_cnts'] = df['d_t'] * bg_cnt

	df['sum_cnts_bg_corrected'] = df['sum_cnts'] - df['bg_cnts']

	# counts per second
	df['cps'] = df['sum_cnts'] / df['d_t']

	# normalized with avg_dose
	df['norm_cps'] = df['cps'] / df['avg_dose']
	
	df_result = df_result.append(df)
	# print(df_result.head())



df_result = df_result[df_result.norm_cps > 0]


# print(df_result.head(50))
# plot norm cps vs position
# print(df)
f, ax = plt.subplots(figsize=(12,5))
X=df_result['edge_pos_mm'].values[0:-1]
Y=df_result['norm_cps'].values[0:-1]
ax.scatter(X,Y)
ax.set_xlabel('Edge position [mm]')
ax.set_ylabel('Normalized, background corrected counts [1/s]')
ax.set_xticks(np.arange(7,20+0.5,0.5))
ax.set_xlim(6,21)
# minor ticks x
minor_locator = AutoMinorLocator(2)
ax.xaxis.set_minor_locator(minor_locator)
plt.grid(True)
plt.title('Edge spread function for measurement from \n {}'.format(my_date))
plt.tight_layout()
filename =  f'//fs03/LTH_Neutimag/hkromer/10_Experiments/08_EmittingSpotSizeMeasurement/{my_date}_measurement/{my_date}_ESF'
df_result.to_csv(f'//fs03/LTH_Neutimag/hkromer/10_Experiments/08_EmittingSpotSizeMeasurement/{my_date}_measurement/{my_date}_ESF.csv')
plt.savefig(filename + '.png', dpi=600)
plt.close('all')



