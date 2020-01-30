import serial
import serial.tools.list_ports
import re
import sys
import pymysql
from time import sleep

db = pymysql.connect(host="twofast-RPi3-0",  # your host
                     user="writer",  # username
                     passwd="heiko",  # password
                     db="NG_twofast_DB")  # name of the database

def saveDB(ard_time, counts_D1, counts_D2, counts_D3, counts_D4):
    # Create a Cursor object to execute queries.
    cur = db.cursor()
    try:
        cur.execute("""INSERT INTO referenceDetectors (ard_time, counts_D1, counts_D2, counts_D3, counts_D4) 
                        VALUES (%s, %s, %s, %s, %s)""",
                    (ard_time, counts_D1, counts_D2, counts_D3, counts_D4))
    except:
        cur.rollback()

    db.commit()
    cur.close()

def readDB():
    # Create a Cursor object to execute queries.
    cur = db.cursor()
    try:
        cur.execute("""SELECT * FROM dose;""")
        rows = cur.fetchall()
        for row in rows:
            print(row)
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

def serial_open(serial_port):
    serialArduino = serial.Serial(port=serial_port, baudrate=9600, dsrdtr=True)
    return serialArduino

def pi_read(serialArduino):
    # print('Reading arduino.')
    while (serialArduino.inWaiting() == 0):  # wait for incoming data
        pass
    valueRead = serialArduino.readline(500)
    try:
        valueRead = (valueRead.decode('utf-8')).strip()
        # print(valueRead)
    except UnicodeDecodeError:
        valueRead = '-1'
    return valueRead

def serial_close(ser):
    ser.close()
# readDB()

# arduinoPort = None
# ports = fun_read_serial_ports()
arduinoPort = '/dev/ttyACM0'

# for port in ports:
#     t = re.findall(r'(/dev/\S+).+Arduino', str(port))
#     if len(t) > 0:
#         arduinoPort = t[0]
#         print('Using Arduino found on port: ', str(port))
#         break
#
# if arduinoPort == None:
#     print('No Arduino connected on serial port. Exiting.')
#     sys.exit(1)


# readout of the arduino
pi_flush(arduinoPort)
while True:
    try:
        serialArduino = serial_open(arduinoPort)
        ardRead = pi_read(serialArduino)
        s = ardRead.rstrip().split()
        print(s)
        # ard_time, ct1, ct2, ct3, ct4
        # ard_time is the time read by the arduino, ctX are countrates in detector X until that time interval
        if len(s) == 7:
            ard_time = s[1]
            counts_D1 = s[3]
            counts_D2 = s[4]
            counts_D3 = s[5]
            counts_D4 = s[6]

            print(ard_time, counts_D1, counts_D2, counts_D3, counts_D4)
            if float(ard_time) >= 30000.0:
                serial_close(serialArduino)
                pi_flush(arduinoPort)
                serialArduino = serial_open(arduinoPort)
                saveDB(ard_time, counts_D1, counts_D2, counts_D3, counts_D4)
        sleep(0.1)
    except KeyboardInterrupt:
        print('Ctrl + C. Exiting. Flushing serial connection.')
        pi_flush(arduinoPort)
        sys.exit(1)
    #finally:
     #   pi_flush(arduinoPort)

