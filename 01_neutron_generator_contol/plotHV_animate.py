import matplotlib.pyplot as plt
import numpy as np
import pymysql
import time
import pandas as pd
import datetime
import matplotlib.animation as animation




def fun_readHVFromDB():
	t0 = time.time()
	db = pymysql.connect(host="twofast-RPi3-0",  # your host
						 user="pressReader",  # username
						 passwd="heiko",  # password
						 db="NG_twofast_DB")  # name of the database
	# Create a Cursor object to execute queries.
	cur = db.cursor()

	try:
		print('Starting HV plot...')
		# cur.execute("""INSERT INTO pressure (pressure_IS, pressure_VC) VALUES (%s, %s)""",(press_IS,press_VC))
		cur.execute("""SELECT * FROM HBox_Uno ORDER BY id DESC LIMIT 300""")
		# cur.execute("""SELECT * FROM pressure""")
		rows = cur.fetchall()
		df = pd.DataFrame( [[ij for ij in i] for i in rows] )
		# voltage_dose, counts_WS, counts_BS, counts_DF
		df.rename(columns={0: 'ID', 1: 'date', 2: 'dose_voltage', 3: 'HV_current', 4: 'HV_voltage'}, inplace=True)

		df = df.set_index(['ID'])
		df['HV_current'] = df['HV_current'] * 100
		df['HV_voltage'] = df['HV_voltage']
		# print(df.head())

	except:
		cur.rollback()

	cur.close()

	return df




fig = plt.figure(figsize=(6,4))
ax1 = fig.add_subplot(1, 1, 1)


line_HV_I,  = ax1.plot([0], [0], lw=1, color='blue')  # real data
line_HV_V,  = ax1.plot([0], [0], lw=1, color='red')  # averaged data
text_HV_I = ax1.text(0.97, 0.37, "", transform=ax1.transAxes, ha="right", va="top", color='blue')
# text_HV_I = ax1.text(500, 0.37, "", transform=ax1.transAxes, ha="right", va="top", color='blue')
text_HV_V = ax1.text(0.57, 0.37, "", transform=ax1.transAxes, ha="right", va="top", color='red')
text_time = ax1.text(0.92, 0.87, "", transform=ax1.transAxes, ha="right", va="top", color='black')
text_HV_I.set_fontsize(25)
text_HV_V.set_fontsize(25)


plt.ylim(0,150)
plt.xlim(0,300)
# ax1.text(0.95, 0.45, "HV current [mA]:", transform=ax1.transAxes, ha="right", va="top", color='b')
# ax1.text(-50, 60, "HV current [mA]:", transform=ax1.transAxes, ha="right", va="top", color='b')
# ax1.text(0.60, 0.45, "HV voltage [-kV]:", transform=ax1.transAxes, ha="right", va="top", color='r')
plt.subplots_adjust(left=0.1, right=0.98, top=0.94)
# plt.subplots_adjust(right=0.15)
# fig.tight_layout()
plt.xlabel('Time in last 300 seconds [s]')
plt.ylabel('HV [-kV], I [*0.01 mA]')
plt.title('HV Power Supply')

def animateinit(): #tells our animator what artists will need re-drawing every time
	# return line,text
	return line_HV_I, line_HV_V, text_HV_I, text_HV_V, text_time
	# return line_IS, line_VC

def animate(i):
	df = fun_readHVFromDB()
	# print(df.head())

	line_HV_I.set_data(range( len(df['HV_current']) ) , df['HV_current'])
	line_HV_V.set_data(range( len(df['HV_voltage']) ) , df['HV_voltage'])
	text_HV_I.set_text(str(round(df['HV_current'].iloc[0]/100,2)) + ' mA')
	text_HV_V.set_text('-' + str(round(df['HV_voltage'].iloc[0],1)) + ' kV')
	text_time.set_text(df['date'].iloc[0])
	# print(text_IS, text_VC)
	return line_HV_I, line_HV_V, text_HV_I, text_HV_V, text_time #return the updated artists
	# return line_IS, line_VC #return the updated artists


#inform the animator what our init_func is and enable blitting
ani = animation.FuncAnimation(fig, animate, interval=750,init_func=animateinit, blit=True)
plt.show()
