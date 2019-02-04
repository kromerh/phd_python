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

def saveDB(counts_WS, counts_BS):
    # Create a Cursor object to execute queries.
    cur = db.cursor()
    try:
        cur.execute("""INSERT INTO HBox_Due (counts_WS, counts_BS) VALUES (%s, %s)""", (counts_WS, counts_BS))
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

def pi_read(serial_port):
    serialArduino = serial.Serial(port=serial_port, baudrate=9600)
    while (serialArduino.inWaiting() == 0):  # wait for incoming data
        pass
    valueRead = serialArduino.readline(500)
    try:
        valueRead = (valueRead.decode('utf-8')).strip()
       # print(valueRead)
    except UnicodeDecodeError:
        valueRead = '-1'
    return valueRead

arduinoPort = None
ports = fun_read_serial_ports()
print(ports)
for port in ports:
    # print(port)
    t = re.findall(r'(/dev/\S+).+Arduino Due', str(port))
    if len(t) > 0:
        arduinoPort = t[0]
        print('Arduino Due port found: ', str(port))
        break

if arduinoPort == None:
    print('No Arduino connected on serial port. Exiting.')
    sys.exit(1)

counts_WS = 0
counts_BS = 0
# readout of the arduino
pi_flush(arduinoPort)
while True:
    try:
        ardRead = pi_read(arduinoPort)
        s = ardRead.rstrip().split()
        if len(s) == 2:  # WS  BS2
            print(s)
            counts_WS = float(s[0])
            counts_BS = float(s[1])
            saveDB(counts_WS, counts_BS)
        sleep(0.1)
    except KeyboardInterrupt:
        print('Ctrl + C. Exiting. Flushing serial connection.')
        pi_flush(arduinoPort)
        sys.exit(1)
    finally:
        pi_flush(arduinoPort)

