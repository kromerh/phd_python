import serial
import sys

def pi_write(serial_port,serial_order):
	serialArduino = serial.Serial(port=serial_port, baudrate=9600)
	serialArduino.write(bytes(serial_order,'UTF-8'))
	serialArduino.close()

serial_port = sys.argv[1]
serial_order = sys.argv[2]
pi_write(serial_port,serial_order)