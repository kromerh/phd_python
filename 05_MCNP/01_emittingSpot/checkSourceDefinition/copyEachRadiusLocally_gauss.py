import os
import re 
import sys
import random
from shutil import copyfile

# directory where the MCNP run files are saved
dir_runfiles = '//fs03/LTH_Neutimag/hkromer/10_Experiments/02_MCNP/neutron_emitting_spot/2_case_files/_experiment_gauss_20180911/case_0'

# directory where the MCNP run files shall be put into
dir_out = 'D:/neutron_emitting_spotsize/check_source_gauss_larger/'

# get a list of radii in the run
lst_dirs = (os.listdir(dir_runfiles))
lst_dirs = [f for f in lst_dirs if os.path.isdir(f'{dir_runfiles}/{f}')]  # select only directories
# print(lst_dirs)
lst_radii = [re.findall(r'rad(\d.\d+)_',f) for f in lst_dirs]
lst_radii = [f[0] for f in lst_radii if len(f) > 0]
lst_radii = list(set(lst_radii))  # unique list

# for each radius take a random case and copy that MCNP file
for radius in lst_radii:	
	print(f'Doing radius {radius}')
	lst_files = [f for f in lst_dirs if len(re.findall(r'rad' + radius + r'_x',f)) > 0]  # select all the files
	el = random.choice(lst_files) # select random element
	directory = f'{dir_out}/{el}'
	# print(el)#
	# sys.exit()
	if not os.path.exists(directory):
		os.makedirs(directory)

	src = f'{dir_runfiles}/{el}/{el}.input'
	dst = f'{directory}/{el}.input'
	
	copyfile(src, dst)  # copy MCNP file


	# open the MCNP file and change nps as well as ptrac line
	c = []
	with open(dst, 'r') as file:
		for line in file:
			# line = line.rstrip()
			if 'nps 1' in line:
				line = 'nps 1e4 \n'
			if 'PTRAC' in line:
				line = 'PTRAC file=asc write=all max=1e9'
			c.append(line)
		file.close()

	with open(dst, 'w') as file:
		for l in c:
			# l = ' '.join(l)
			file.write(l)
		file.close()

	src = f'{dir_runfiles}/{el}/{el}.bat'
	dst = f'{directory}/{el}.bat'

	copyfile(src, dst)  # copy bat file

	# open bat file and replace the cd line
	c = []
	with open(dst, 'r') as file:
		for line in file:
			# line = line.rstrip()
			t0 = re.findall(r'net use Y:(.+)', line)
			if len(t0) > 0:
				directory = directory.replace('/','\\')
				line = f'cd {directory} \n'

			t1 = re.findall(r'cd /d', line)
			if len(t1) > 0:
				line = '\n'

			t2 = re.findall(r'net use /delete', line)
			if len(t2) > 0:
				line = '\n'

			c.append(line)
		file.close()

	with open(dst, 'w') as file:
		for l in c:
			# l = ' '.join(l)
			file.write(l)
		file.close()

	