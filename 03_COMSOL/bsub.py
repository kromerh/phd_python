import os
from random import *
import re


folder_with_files = '//Users/hkromer/02_PhD/02_Data/01_COMSOL/\
01_IonOptics/2018-10-23_comsol/log/'
fname = 'bjobs.txt'

files = f'{folder_with_files}/{fname}'

jobs = []
with open(files, 'r') as file:
	for line in file:
		line = line.rstrip()
		# only take mph files
		if line.endswith('.mph'):
			jobs.append(line)
	file.close()

print(f'Processing {len(jobs)} COMSOL files.')

rx = randint(1, 10000)
with open(f'{folder_with_files}{rx}.bsub', 'w') as file:
	for run in jobs:
		fname = re.findall(r'(.+).mph', run)[0]
		bsub = "bsub -n 4 -W 24:00 -B -N -u kromerh@student.ethz.ch -R\ \"rusage[mem=4000, scratch=5000]\" \"comsol batch -tmpdir \\$TMPDIR\ -configuration \\$TMPDIR -data \\$TMPDIR -autosave off -np 4\ -inputfile {}.mph -outputfile {}_out.mph\"".format(fname, fname)
		file.write('{} \n'.format(bsub))
		file.write('\n')
file.close()

print('Wrote in {}.bsub'.format(rx))
