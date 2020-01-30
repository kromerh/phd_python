import serial
import sys

def pi_read(serial_port):
	serialArduino = serial.Serial(port=serial_port, baudrate=9600)
	while (serialArduino.inWaiting()==0): # wait for incoming data
		pass
	valueRead = serialArduino.readline(500)
	try:
		valueRead = (valueRead.decode('utf-8')).strip()
	except UnicodeDecodeError:
		valueRead = '-1'
	print(valueRead)
	
serial_port = sys.argv[1]
pi_read(serial_port)