import paramiko
import sys
host="twofast-RPi3-6"
user="pi"
pwd="axfbj1122!"
paramiko.util.log_to_file('ssh.log') # sets up logging
try:
    print('Starting sensirion temp humid readout readout...')
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.load_system_host_keys()
    client.connect(host, username=user, password=pwd)
    stdin, stdout, stderr = client.exec_command('~/run_sensirion_readout.sh', get_pty=True)
    print(stdout.readlines())
    print(stderr.read())
    client.close()
except KeyboardInterrupt:
    print('Ctrl + C. Exiting. Flushing serial connection.')
    client.close()
    sys.exit(1)
finally:
    client.close()

