import serial
import serial.tools.list_ports
import re
import sys
import pymysql
from time import sleep
import time
import pandas as pd
import numpy as np

db = pymysql.connect(host="twofast-RPi3-0",  # your host
                     user="pressReader",  # username
                     passwd="heiko",  # password
                     db="NG_twofast_DB")  # name of the database

arduinoPort = '/dev/ttyACM0'
serialArduino = serial.Serial(port=arduinoPort, baudrate=9600)

def saveDB(volt_1, volt_2):
    # Create a Cursor object to execute queries.
    # voltage 1 is ion source
    # voltage 2 is vacuum chamber
    cur = db.cursor()
    try:
        cur.execute("""INSERT INTO BBox (voltage_IS, voltage_VC) VALUES (%s, %s)""", (volt_1, volt_2))
    except:
        cur.rollback()

    db.commit()
    cur.close()

def fun_read_serial_ports():
    return list(serial.tools.list_ports.comports())

def pi_flush(serial_port):
    serialArduino = serial.Serial(port=serial_port, baudrate=9600)
    serialArduino.flushInput()  #flush input buffer, discarding all its contents
    serialArduino.flushOutput() #flush output buffer, aborting current output and discard all that is in buffer

def pi_read():
    serialArduino = serial.Serial(port=arduinoPort, baudrate=9600)
    while (serialArduino.inWaiting() == 0):  # wait for incoming data
        pass
    valueRead = serialArduino.readline()
    # print(valueRead)
    try:
        valueRead = (valueRead.decode('utf-8')).strip()
        serialArduino.flushInput()  #flush input buffer, discarding all its contents
        serialArduino.flushOutput() #flush output buffer, aborting current output and discard all that is in buffer
       # print(valueRead)
    except UnicodeDecodeError:
        valueRead = '-1'
    return valueRead

def getHVdf():
    # connectes to the DB, reads the HV voltage. If the value is above 30 kV, sends via serial to the arduino a 1, else it sends a 0
    # the arduino then allows to flip on the extraction voltage depending on 1 or 0
    db = pymysql.connect(host="twofast-RPi3-0",  # your host
                     user="pressReader",  # username
                     passwd="heiko",  # password
                     db="NG_twofast_DB")  # name of the database
    cur = db.cursor()
    try:
        cur.execute("""SELECT * FROM HBox_Uno ORDER BY id DESC LIMIT 20""")
        rows = cur.fetchall()
        df = pd.DataFrame( [[ij for ij in i] for i in rows] )
        # voltage_dose, counts_WS, counts_BS, counts_DF
        df.rename(columns={0: 'ID', 1: 'date', 2: 'dose_voltage', 3: 'HV_current', 4: 'HV_voltage'}, inplace=True)

        df = df.set_index(['ID'])

        # print(df.index)

        return df
    except:
        cur.rollback()


    cur.close()

def controlExtractionOnOff(df):
    # connectes to the DB, reads the HV voltage. If the value is above 30 kV, sends via serial to the arduino a 1, else it sends a 0
    # the arduino then allows to flip on the extraction voltage depending on 1 or 0
    # serialArduino.write(b'0')
    # print('send 0')
    
    m_lastHV = np.mean(df['HV_voltage'].values)  # average of last 30 readings of HV

    if m_lastHV >= 20.0:
        # larger than 20 kV
        serialArduino.write(b'1')
        # print(m_lastHV, '1')
        # print(df['HV_voltage'].values)
    else:
        # not larger than 30 kV in the last 30 seconds --> no extraction on
        serialArduino.write(b'0')
        # print(m_lastHV, '0')


# arduinoPort = '/dev/ttyACM0'
# ports = fun_read_serial_ports()
# print(ports)
# for port in ports:
#     t = re.findall(r'(/dev/\S+).+Arduino', str(port))
#     if len(t) > 0:
#         arduinoPort = t[0]
#         print('Arduino port found: ', str(port))
#         break

# if arduinoPort == None:
#     print('No Arduino connected on serial port. Exiting.')
#     sys.exit(1)

counts_WS = 0
counts_BS = 0
# readout of the arduino
# pi_flush(arduinoPort)
while True:
    try:
        df_extr = getHVdf()
        # read HV to control the extraction voltage on/off
        controlExtractionOnOff(df_extr)
        # pi_flush(arduinoPort)

        # t0 = time.time()
        ardRead = pi_read()
        # pi_flush(arduinoPort)
        # print(ardRead)

        # print(time.time() - t0)
        s = ardRead.rstrip().split()
        # sys.stdout.write('...running BBox reader...')
        if len(s) == 5:  # V1 V2 extractionOn
            print(s)
            volt_1 = s[0]
            volt_2 = s[1]
            # start_time = time.time()

            saveDB(volt_1, volt_2)
            # print("%s seconds " % (time.time() - start_time))
        sleep(0.1)
        
    except KeyboardInterrupt:
        print('Ctrl + C. Exiting. Flushing serial connection.')
        pi_flush(arduinoPort)
        sys.exit(1)
    finally:
        pi_flush(arduinoPort)

