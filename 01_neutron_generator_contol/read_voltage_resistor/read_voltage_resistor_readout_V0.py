from time import sleep
import serial
import sys
import pandas as pd
import re


arduinoPort = '/dev/ttyACM0'  # might need to be changed if another arduino is plugged in or other serial

ser = serial.Serial(arduinoPort, 9600)
print('Serial connected at ' + str(arduinoPort))
sleep(1)
# val = 0.5 # Below 32 everything in ASCII is gibberish
while True:
	try:
		valueRead = ser.readline(500) # read voltage

		print('Raw reading from Arduino :' + str(valueRead)) # Read the newest output from the Arduino
		sleep(0.5) # Delay
		ser.flushInput()  #flush input buffer, discarding all its contents
		ser.flushOutput() #flush output buffer, aborting current output and discard all that is in buffer
	except KeyboardInterrupt:
		print('Ctrl + C. Exiting. Flushing serial connection.')
		ser.flushInput()  #flush input buffer, discarding all its contents
		ser.flushOutput() #flush output buffer, aborting current output and discard all that is in buffer
		sys.exit(1)