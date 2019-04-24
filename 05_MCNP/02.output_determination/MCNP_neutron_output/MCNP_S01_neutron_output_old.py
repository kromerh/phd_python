from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker
"""
1.) Get the fraction of direct neutrons in the LB6411 response. This is loaded from a previous file.

2.) Use the zerodensity run to get the tally responses for all tallies. Check for each energy the fraction of tally
response over total tally response. Use the interpolation from the LB6411 manual to get response to neutron
fluence R_phi(E).

3.) Use 2 and 1 to get the flux from direct neutrons at that position:
phi = cps_DIR * SUM( 1/R_phi(E) * ( F4(E)/F4_tot ) )

4.) Interpolate the results to have flux vs position
"""

import numpy as np

"""
1.) Get the fraction of direct neutrons in the LB6411 response. This is loaded from a previous file.
"""
inputFilePrefix = 'CurrentTarget' # insert the prefix of the MCNP input file
pathOutputDirectory = '//fs03//LTH_Neutimag//hkromer//10_Experiments//02_MCNP//'

IDrun = 52 # insert the ID of the MCNP run

lstDirection = ['North', 'South', 'West', 'East']  # directions where the tallies/detectors are in
pathLoad = pathOutputDirectory + inputFilePrefix + str(IDrun) + '/' # path to the result files

lstFracDirect = [[] for l in lstDirection]  # direct fraction, empty list for each direction
lstPos = [[] for l in lstDirection]         # position of the tally, empty list for each direction

for direction in lstDirection:  # load the fraction of direct counts in LB6411 response
    idx = lstDirection.index(direction)
    thisPathLoad = pathLoad + direction + '_results/'

    fname = direction + '_fracDirCps_vs_total.txt'
    with open(thisPathLoad + fname, 'r') as file:
        fileContent = file.read().rstrip()
        values = fileContent.split()
        if idx == 1:  # south
            for val in values[2:]:
                lstFracDirect[idx].append(val)
        else:
            for val in values[1:]:
                lstFracDirect[idx].append(val)
        file.close()

    fname = direction + '_tally_position.txt'
    with open(thisPathLoad + fname, 'r') as file:
        fileContent = file.read().rstrip()
        values = fileContent.split()
        for val in values:
            lstPos[idx].append(val)
        file.close()

"""
2.) Use the zerodensity run to get the tally responses for all tallies. Check for each energy the fraction of tally
response over total tally response. Use the interpolation from the LB6411 manual to get response to neutron
fluence R_phi(E).
"""

lstEnergy = [[] for l in lstDirection]  # energy for direct tally fraction, empty list for each direction
# direct flux, empty list for:  0: each direction, 1: each tally
lstTallyDirectFlux = [[[] for ll in lstFracDirect[ii]] for ii in range(0, len(lstFracDirect))]

lstTallyDirectNonZeroFlux = [[[] for ll in lstFracDirect[ii]] for ii in range(0, len(lstFracDirect))]
                                # direct flux, empty list for:  0: each direction, 1: each tally. contains energy, flux

for direction in lstDirection:  # load the fraction of direct counts in LB6411 response
    idx = lstDirection.index(direction)
    thisPathLoad = pathLoad + direction + '_results/'

    # import energy
    fname = direction + '_cps_direct.txt'
    with open(thisPathLoad + fname, 'r') as file:
        for line in file:
            sLine = line.rstrip().split()
            lstEnergy[idx].append(sLine[0])
            # print(sLine[0])
        file.close()

    # load direct flux
    fname = direction + '_zerodensityDirectFlux.txt'
    with open(thisPathLoad + fname, 'r') as file:
        for line in file:
            sLine = line.rstrip().split()
            if idx == 1:  # south direction
                for ii in range(2, len(sLine)):  #
                    lstTallyDirectFlux[idx][ii - 2].append(sLine[ii])
            else:
                for ii in range(1, len(sLine)): #
                    lstTallyDirectFlux[idx][ii-1].append(sLine[ii])
        file.close()

# make [energy, tally response] for nonzero tally responses
for idx_dir in range(0, len(lstTallyDirectFlux)): # each direction
    lstTally = lstTallyDirectFlux[idx_dir]

    for idx_tally in range(0, len(lstTally)):  # each tally
        lstFlux = lstTally[idx_tally]
        for val in lstFlux:  # each value
            idx_val = lstFlux.index(val)
            if float(val) > 0:
                energy = lstEnergy[idx_dir][idx_val]
                lstTallyDirectNonZeroFlux[idx_dir][idx_tally].append([energy, val]) # energy, value

                # print(idx_dir, idx_tally, val, energy)

# print(lstTallyDirectNonZeroFlux[1][-1])

# fractions of the tally response(s)
for idx_dir in range(0, len(lstTallyDirectNonZeroFlux)): # each direction
    lstTally = lstTallyDirectNonZeroFlux[idx_dir]

    for idx_tally in range(0, len(lstTally)):  # each tally
        lstFlux = lstTally[idx_tally]
        # for val in lstFlux:  # each value
        totFlux = 0.0
        for vals in lstFlux:
            totFlux = float(vals[1]) + totFlux
        for vals in lstFlux:
            val = float(vals[1]) / totFlux
            vals[1] = str(val)
        # print(idx_dir, idx_tally, lstFlux, totFlux)

# compute LB6411 response with the interpolation from the manual and the relative direct fluxes from the tallies

# Response = SUM( Rphi times F4(E)/F4_tot ) for each tally, empty list for:  0: each direction, 1: each tally
lstResponse = [[0.0 for ll in lstFracDirect[ii]] for ii in range(0, len(lstFracDirect))]

# Import LB6411 calibration energy R_phi
# raw data from LB6411 manual
# first column is neutron energy (MeV)
# second column is "R_phi", which is response to neutron fluence in cm2 (counts / neutron fluence)
lstLB6411_E = []  # float
lstLB6411_Rphi = []  # float
fhandle = open('LB6411_energy_Rphi.txt', 'r')
for line in fhandle:
    lstLine = line.split()
    lstLB6411_E.append(float(lstLine[0]))
    lstLB6411_Rphi.append(float(lstLine[1]))
fhandle.close()

# test interpolation
# energy = 2.125e+00
# inter = np.interp([energy], lstLB6411_E, lstLB6411_Rphi)[0]
# print(inter)
# energy = 2.175e+00
# inter = np.interp([energy], lstLB6411_E, lstLB6411_Rphi)[0]
# print(inter)

# fig = plt.figure(figsize=(6.4 * 1.2, 4.8 * 1.2))
# figsize = fig.get_size_inches()
# ax = fig.add_subplot(111)
# ax.scatter(lstLB6411_E,lstLB6411_Rphi)
# ax.scatter(energy,inter, color='red')
# plt.show()

# response function from interpolation
for idx_dir in range(0, len(lstTallyDirectNonZeroFlux)): # each direction
    lstTally = lstTallyDirectNonZeroFlux[idx_dir]

    for idx_tally in range(0, len(lstTally)):  # each tally
        lstFlux = lstTally[idx_tally]
        for vals in lstFlux:
            energy = float(vals[0])
            frac = float(vals[1])  # F4(E) / F4_tot
            inter = np.interp([energy], lstLB6411_E, lstLB6411_Rphi)[0]
            lstResponse[idx_dir][idx_tally] = lstResponse[idx_dir][idx_tally] + ( float(inter) * frac )
            # print(idx_dir, idx_tally, lstFlux, frac, lstResponse[idx_dir][idx_tally] )

# for item in lstResponse:
#     print(item)
#
# for itm in lstFracDirect:
#     print(itm)
"""
3.) Use 2 and 1 to get the flux from direct neutrons at that position:
phi = cps_DIR * SUM( 1/R_phi(E) * ( F4(E)/F4_tot ) )
phi(tally) = cps_DIR(tally) * lstResponse(tally)
 
tally can be seen as the position at which the tally resides
"""

# µSv/hr to cps according to the manual
dose_to_cps = 0.79  # per µSv/hr

# assume 100 µSv/hr
cps = 100.0 * dose_to_cps
lstFlux = [[0.0 for ll in lstFracDirect[ii]] for ii in range(0, len(lstFracDirect))]  # local flux at the tally position
lstOutput = [[0.0 for ll in lstFracDirect[ii]] for ii in range(0, len(lstFracDirect))] # total output if 100 µSv/hr at the tally position

for idx_dir in range(0, len(lstResponse)): # each direction
    thisResponse = lstResponse[idx_dir]    # list of response values for idx_dir (direction) each is one tally
    thisFracDirect = lstFracDirect[idx_dir] # list of fraction of direct counts in LB6411 response for idx_dir (direction) each is one tally
    for idx_tally in range(0, len(thisResponse)):
        response = thisResponse[idx_tally]
        fracDirect = thisFracDirect[idx_tally]
        distance = lstPos[idx_dir][idx_tally]  # distance source to tally center in cm
        flux = float(cps) * (float(fracDirect)) * (1.0 / float(response))   # flux in n/cm2/s at the tally position
        output = flux * (4.0 * np.pi * (float(distance)**2))

        lstFlux[idx_dir][idx_tally] = flux
        lstOutput[idx_dir][idx_tally] = output



# save output to file
for idx_dir in range(0, len(lstOutput)):  # each direction
    # save output to file
    fname = lstDirection[idx_dir] + '_Output.txt'
    thisPathSave = pathLoad + lstDirection[idx_dir] + '_results/'
    with open(thisPathSave + fname, 'w') as file:
        file.write('')
        file.close()
    with open(thisPathSave + fname, 'a') as file:
        for idx_dir in range(0, len(lstOutput)):  # each direction
            file.write(lstDirection[idx_dir] + ' tally position: \n')
            file.write(' '.join(lstPos[idx_dir]) + '\n')
            lstWrite = []  # string list of output list
            file.write(lstDirection[idx_dir] + ' source output per 100 muSv/hr: \n')
            for item in lstOutput[idx_dir]:
                lstWrite.append(str(item))
            file.write(' '.join(lstWrite) + '\n')
        file.close()

# for value in lstFlux:
#     print(value)
for value in lstOutput:
    idx = lstOutput.index(value)
    print(lstDirection[idx])
    print('output', value)
    print('direct fraction in LB6411 counts', list(map('{0:.8e}'.format, lstFracDirect[idx])))
    print('distance in cm', list(map(float,lstPos[idx])))
    print('flux', lstFlux[idx])
    print('R_phi', lstResponse[idx])


fig = plt.figure(figsize=(6.8 * 1.2, 4.8 * 1.2))
figsize = fig.get_size_inches()
ax = fig.add_subplot(111)
minor_locator = AutoMinorLocator(2)
ax.xaxis.set_minor_locator(minor_locator)
minor_locator = AutoMinorLocator(5)
ax.yaxis.set_minor_locator(minor_locator)
colors = ['b', 'r', 'g', 'k', 'r']
markers = ['x','o', 'v', 's', 'p']
for idx_dir in range(0, len(lstOutput)):
    ax.plot(lstPos[idx_dir], lstOutput[idx_dir], color=colors[idx_dir], marker=markers[idx_dir], label=lstDirection[idx_dir])
fig.subplots_adjust(top=0.90)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5e7))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
plt.suptitle('Neutron output', fontweight='bold')
ax.set_title('per 100 µSv/hr', fontsize=9, fontweight='bold')
plt.xlabel('Distance from source [cm]', fontweight='bold')
plt.ylabel('Neutron output [n/s] per 100 µSv/hr', fontweight='bold')
ax.grid(b=True, which='major', linestyle='-')
ax.grid(b=True, which='minor', linestyle='--')
# plt.ylim([0, 50])
plt.xlim([0, 210])
plt.legend()
plt.show()
# plt.savefig(pathOutputDirectory + inputFilePrefix + str(IDrun[0]) + '/relDiffCutoffEnergy.png', dpi=100)
# plt.savefig(pathOutputDirectory + inputFilePrefix + str(IDrun[1]) + '/relDiffCutoffEnergy.png', dpi=100)
# plt.close(fig)



# check
# for item in lstEnergy:
#     print(item)
# for item in lstTallyDirectFlux[2]:
#     for val in item:
#         if float(val) > 0:
#             print(val, lstTallyDirectFlux[2].index(item), lstEnergy[2][item.index(val)])




