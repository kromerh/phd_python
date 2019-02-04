import matplotlib.pyplot as plt
import numpy as np
import pymysql
import time
import pandas as pd
import datetime
import matplotlib.animation as animation
from matplotlib.ticker import AutoMinorLocator



def fun_readDoseFromDB():
	t0 = time.time()
	db = pymysql.connect(host="twofast-RPi3-0",  # your host
						 user="doseReader",  # username
						 passwd="heiko",  # password
						 db="NG_twofast_DB")  # name of the database
	# Create a Cursor object to execute queries.
	cur = db.cursor()

	try:
		print('Starting dose plot...')
		# cur.execute("""INSERT INTO pressure (pressure_IS, pressure_VC) VALUES (%s, %s)""",(press_IS,press_VC))
		cur.execute("""SELECT * FROM HBox_Uno ORDER BY id DESC LIMIT 300""")
		# cur.execute("""SELECT * FROM pressure""")
		rows = cur.fetchall()
		df = pd.DataFrame( [[ij for ij in i] for i in rows] )
		# voltage_dose, counts_WS, counts_BS, counts_DF
		df.rename(columns={0: 'ID', 1: 'date', 2: 'dose_voltage', 3: 'HV_current', 4: 'HV_voltage'}, inplace=True)

		df = df.set_index(['ID'])
		df['dose'] = df['dose_voltage'] * 3000 / 5.5

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




fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

# cnt=0
dose = []



line,  = ax1.plot([0], [0], lw=1, color='red')  # real data
text = ax1.text(0.70, 0.18, "", transform=ax1.transAxes, ha="right", va="top", color='red')
text_time = ax1.text(0.92, 0.07, "", transform=ax1.transAxes, ha="right", va="top", color='black')

plt.ylim(0,2500)
plt.xlim(0,300)
ax1.grid(b=True, which='major', linestyle='-')
ax1.grid(b=True, which='minor', linestyle='--')
minor_locator = AutoMinorLocator(2)
ax1.xaxis.set_minor_locator(minor_locator)
minor_locator = AutoMinorLocator(5)
ax1.yaxis.set_minor_locator(minor_locator)
# ax1.set_yticks([1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
ax1.text(0.25, 0.17, "Dose:", transform=ax1.transAxes, ha="right", va="top", color='red', fontsize=20)
ax1.text(0.50, 0.07, "Time:", transform=ax1.transAxes, ha="right", va="top", color='black')
plt.subplots_adjust(left=0.1, right=0.98, top=0.94)
plt.xlabel('Last 300 readings')
plt.ylabel('Dose [µSv/h]')
plt.title('Dose read by LB6411 at 70 cm W')

def animateinit(): #tells our animator what artists will need re-drawing every time
	# return line,text
	return line, text, text_time
	# return line_IS, line_VC

def animate(i):
	df = fun_readDoseFromDB()
	# print(df.head())

	line.set_data(range( len(df['dose']) ) , df['dose'])
	txt = str(round(df['dose'].iloc[0])) + ' µSv/h'
	text.set_text(txt)
	text.set_fontsize(25)
	text_time.set_text(df['date'].iloc[0])
	# print(text_IS, text_VC)
	return line, text, text_time #return the updated artists
	# return line_IS, line_VC #return the updated artists


#inform the animator what our init_func is and enable blitting
ani = animation.FuncAnimation(fig, animate, interval=500,init_func=animateinit, blit=True)
plt.show()
