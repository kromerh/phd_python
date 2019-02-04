import pandas as pd
import numpy as np
import os
import re 
import matplotlib.pyplot as plt
import sys


remote_path = '//fs03/LTH_Neutimag/hkromer/'

# particle_data_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/IGUN_geometry/mesh_refinement/total_electric_energy'.format(remote_path)
tot_el_en_path = '{}02_Simulations/06_COMSOL/03_BeamOptics/01_OldTarget/IGUN_geometry/mesh_refinement/2018-07-11/ES_MR'.format(remote_path)


file = '{}/total_electric_energy.txt'.format(tot_el_en_path)

df = pd.read_csv(file, delimiter=r'\t+')

print(df)

# total electric energy global

# relative differences
X = df['n_dof'].values[1:]
ref_W = df['tot_W'].values[-1]

W = 100* ( np.abs(df['tot_W'].values[1:]-ref_W)/ref_W)

f, ax = plt.subplots()

ax.plot(X,W )
ax.scatter(X,W)

plt.grid(True)
plt.ylabel('Relative difference in total electric energy [%]')
plt.xlabel('d.o.f.')
locs, labels = plt.xticks() 

plt.xticks([500000, 1000000, 1500000, 2000000, 2500000, 3000000], ['500 000', '1 000 000', '1 500 000', '2 000 000', '2 500 000', '3 000 000'])
plt.tight_layout()
# plt.show()
filename =  '{}/global_mr_totElEn_vs_dof'.format(tot_el_en_path)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')


# total electric energy accelerator column 1

metric = 'tot_W_accCol1'

# relative differences
X = df['n_dof'].values[1:]
ref_W = df[metric].values[-1]

W = 100* ( np.abs(df[metric].values[1:]-ref_W)/ref_W)

f, ax = plt.subplots()

ax.plot(X,W )
ax.scatter(X,W)

plt.grid(True)
plt.ylabel('Relative difference in total electric energy [%]')
plt.xlabel('d.o.f.')
locs, labels = plt.xticks() 

plt.xticks([500000, 1000000, 1500000, 2000000, 2500000, 3000000], ['500 000', '1 000 000', '1 500 000', '2 000 000', '2 500 000', '3 000 000'])
plt.tight_layout()
# plt.show()
filename =  '{}/{}_mr_totElEn_vs_dof'.format(tot_el_en_path, metric)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')


# total electric energy accelerator column 2

metric = 'tot_W_accCol2'

# relative differences
X = df['n_dof'].values[1:]
ref_W = df[metric].values[-1]

W = 100* ( np.abs(df[metric].values[1:]-ref_W)/ref_W)

f, ax = plt.subplots()

ax.plot(X,W )
ax.scatter(X,W)

plt.grid(True)
plt.ylabel('Relative difference in total electric energy [%]')
plt.xlabel('d.o.f.')
locs, labels = plt.xticks() 

plt.xticks([500000, 1000000, 1500000, 2000000, 2500000, 3000000], ['500 000', '1 000 000', '1 500 000', '2 000 000', '2 500 000', '3 000 000'])
plt.tight_layout()
# plt.show()
filename =  '{}/{}_mr_totElEn_vs_dof'.format(tot_el_en_path, metric)
# plt.savefig(filename + '.eps', dpi=1200)
# plt.savefig(filename + '.svg', dpi=1200)
plt.savefig(filename + '.png', dpi=600)
plt.close('all')