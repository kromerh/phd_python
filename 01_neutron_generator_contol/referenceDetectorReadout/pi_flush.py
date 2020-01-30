import serial
import sys

def pi_flush(serial_port):
	serialArduino = serial.Serial(port=serial_port, baudrate=9600)
	serialArduino.flushInput()  #flush input buffer, discarding all its contents
	serialArduino.flushOutput() #flush output buffer, aborting current output and discard all that is in buffer  

serial_port = sys.argv[1]
pi_flush(serial_port)
