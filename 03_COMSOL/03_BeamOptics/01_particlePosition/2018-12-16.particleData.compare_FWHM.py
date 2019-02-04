import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d

# PSI computer
# remote_path = '//fs03/LTH_Neutimag/hkromer/'

# local computer
remote_path = '/Users/hkromer/02_PhD/02_Data/01_COMSOL/01_IonOptics/\
03.new_chamber/07.mesh_refinement.right_alignment/meshRefinement/'

# files_along_x = '{}/02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/\
# IGUN_geometry\2018-09-19_comsolGeometry\mesh_refinement'.format(remote_path)

files_along_x = f'{remote_path}'

# Do not forget to comment out the reference df!

files = os.listdir(files_along_x)
# files = [f for f in files if f.endswith('.csv')]
files = [f for f in files if f.endswith('.csv')]
