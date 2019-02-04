import serial
import serial.tools.list_ports
import re
import sys
from time import sleep
import pandas as pd
import datetime

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
print(ports[0])
# for port in ports:
    # print(port)
    # t = re.findall(r'(/dev/\S+).+Arduino Due', str(port))
    # if len(t) > 0:
    #     arduinoPort = t[0]
    #     print('Arduino Due port found: ', str(port))
    #     break
arduinoPort = 'COM39'
# if arduinoPort == None:
#     print('No Arduino connected on serial port. Exiting.')
#     sys.exit(1)

# readout of the arduino
pi_flush(arduinoPort)
ardRead = pi_read(arduinoPort)
# print(ardRead)
ii = 1
df = pd.DataFrame(columns=['time', 'readtime', 'value']) 
while True:
    try:
        ardRead = pi_read(arduinoPort)      
        s = ardRead.rstrip().split()
        
        if len(s) == 3:
            
            time = str(datetime.datetime.now())
            
            readtime = s[1]
            val = s[2]
            print(time, val)
            df_read = pd.DataFrame([[time, readtime, val]], columns=df.columns)
            df = df.append(df_read, ignore_index=True)
            ii = ii + 1

        if ii%4 == 0:  # write df every 2 minutes
            date = datetime.date.today()
            df.to_csv('{}_readout.csv'.format(date))
            print('saved to csv {}_readout.csv'.format(date))
        sleep(0.1)
    except KeyboardInterrupt:
        print('Ctrl + C. Exiting. Flushing serial connection.')
        pi_flush(arduinoPort)
        sys.exit(1)
    finally:
        pi_flush(arduinoPort)

