import matplotlib.pyplot as plt
import numpy as np
import pymysql
import time
import pandas as pd
import datetime
import matplotlib.animation as animation




def fun_readPressureFromDB():
	t0 = time.time()
	db = pymysql.connect(host="twofast-RPi3-0",  # your host
						 user="pressReader",  # username
						 passwd="heiko",  # password
						 db="NG_twofast_DB")  # name of the database
	# Create a Cursor object to execute queries.
	cur = db.cursor()

	try:
		print('Starting pressure plot...')
		# cur.execute("""INSERT INTO pressure (pressure_IS, pressure_VC) VALUES (%s, %s)""",(press_IS,press_VC))
		cur.execute("""SELECT * FROM BBox ORDER BY id DESC LIMIT 300""")
		# cur.execute("""SELECT * FROM pressure""")
		rows = cur.fetchall()
		df = pd.DataFrame( [[ij for ij in i] for i in rows] )
		df.rename(columns={0: 'ID', 1: 'date', 2: 'voltage_IS', 3: 'voltage_VC'}, inplace=True)
		df = df.set_index(['ID'])
		df['pressure_IS'] = 10**(1.667*df['voltage_IS']-11.33)
		df['pressure_VC'] = 10**(1.667*df['voltage_VC']-11.33)
		# print(df.head())
		t1 = time.time()

		total = t1 - t0
		# print('Time to query last 300 entries from DB', total, ' seconds')

		# n = 300  # last n entries to plot
		# fig = plt.figure(figsize=(10,5))
		# ax1 = fig.add_subplot(1, 1, 1)
		# plt.ylim(1e-8, 1)
		# # plt.xlim(0, 65)
		# ax1.set_yscale('log')
		# ax1.set_yticks([1e-8,1e-7,1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
		# # time = matplotlib.dates.date2num(time)
		# # plt_df = pd.DataFrame()
		# # plt_df['date'] = df['date'].dt.time
		# # plt_df['voltage_IS'] = df['voltage_IS']
		# # plt_df = plt_df.set_index(['date'])
		# # print(plt_df.head())
		# # p_IS = df['voltage_IS'].apply(lambda x: 10**(1.667*x - 11.33))
		# # p_VC = df['voltage_VC'].apply(lambda x: 10**(1.667*x - 11.33))
		# ax1.plot(df['date'].tail(n), df['pressure_IS'].tail(n))
		# ax1.plot(df['date'].tail(n), df['pressure_VC'].tail(n), color='blue')
		# # ax1.plot(df['date'].tail(100), p_VC, color='blue')
		# # plt.xticks(df['date'].dt.time, rotation=50)
		# fig.subplots_adjust(bottom=0.5)
		# # beautify the x-labels
		# plt.gcf().autofmt_xdate()
		# plt.show()

	except:
		cur.rollback()

	cur.close()

	return df




fig = plt.figure(figsize=(6,3))
ax1 = fig.add_subplot(1, 1, 1)

# cnt=0
pressure_IS = []
pressure_VC = []


line_IS,  = ax1.plot([0], [0], lw=1, color='blue')  # real data
line_VC,  = ax1.plot([0], [0], lw=1, color='red')  # averaged data
text_IS = ax1.text(0.97, 0.07, "", transform=ax1.transAxes, ha="right", va="top", color='blue')
text_VC = ax1.text(0.57, 0.07, "", transform=ax1.transAxes, ha="right", va="top", color='red')
text_time = ax1.text(0.92, 0.87, "", transform=ax1.transAxes, ha="right", va="top", color='black')

plt.ylim(1e-8,1)
plt.xlim(0,300)
ax1.set_yscale('log')
ax1.set_yticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
ax1.text(0.60, 0.12, "Vaccum chamber:", transform=ax1.transAxes, ha="right", va="top", color='red')
ax1.text(0.95, 0.12, "Ion source:", transform=ax1.transAxes, ha="right", va="top", color='blue')
plt.xlabel('Time in last 300 seconds [s]')
plt.ylabel('Pressure [mbar]')
plt.title('Pressure reading (BBox)')

def animateinit(): #tells our animator what artists will need re-drawing every time
	# return line,text
	return line_IS, line_VC, text_IS, text_VC, text_time
	# return line_IS, line_VC

def animate(i):
	df = fun_readPressureFromDB()
	# print(df.head())

	line_IS.set_data(range( len(df['pressure_IS']) ) , df['pressure_IS'])
	line_VC.set_data(range( len(df['pressure_VC']) ) , df['pressure_VC'])
	text_IS.set_text("{0:.2E} mbar".format(df['pressure_IS'].iloc[0]))
	text_VC.set_text("{0:.2E} mbar".format(df['pressure_VC'].iloc[0]))
	text_time.set_text(df['date'].iloc[0])
	# print(text_IS, text_VC)
	return line_IS, line_VC, text_IS, text_VC, text_time #return the updated artists
	# return line_IS, line_VC #return the updated artists


#inform the animator what our init_func is and enable blitting
ani = animation.FuncAnimation(fig, animate, interval=750,init_func=animateinit, blit=True)
plt.show()
