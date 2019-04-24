import re
import numpy as np
import pandas as pd
import glob
from matplotlib import pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Helvetica'
matplotlib.rcParams['mathtext.it'] = 'Helvetica:italic'
matplotlib.rcParams['mathtext.bf'] = 'Helvetica:bold'
"""
1.) Load the tally outputs. This is flux per source particle flux(E). Take the center bin energy as the energy for that bin.

2.) Use LB6411 manual to get the response in the LB6411 for that energy, R(E) 
Get M(E) = R(E) * flux(E)
M(E) is the counts in LB6411 per source particle for each energy

3.)  Get M as the total of M(E) which is the total counts in LB6411 per source particle

4.) Assume 1 µSv/hr --> 0.79 counts in the LB6411 --> counts / M yields the number of source particles 
"""


"""
1.) Load the tally outputs. This is flux per source particle flux(E). Take the center bin energy as the energy for that bin.
"""
inputFilePrefix = 'CurrentTarget' # insert the prefix of the MCNP input file
pathOutputDirectory = '//fs03//LTH_Neutimag//hkromer//10_Experiments//02_MCNP//'

IDrun = 99 # insert the ID of the MCNP run
IDruns = np.arange(216, 217, 1) # insert the ID of the MCNP run

for IDrun in IDruns:
    run = inputFilePrefix + str(IDrun)

    thisMode = '_normal'

    # read tally files in the directory
    path_MCNP_inputFolder = pathOutputDirectory + run + '//' + run + thisMode + '//'
    print('Processing: ', str(path_MCNP_inputFolder))

    df_energy = []  # energy column
    energyMade = False
    df = pd.DataFrame()     # contains the flux
    df_err = pd.DataFrame() # contains the error in the flux (fraction)
    # create dataframe structure
    for filename in glob.glob(path_MCNP_inputFolder + "tallyOutputInCell*"):
        cell = filename[-2:]
        # print(filename, cell)
        fhandle = open(filename, 'r')
        lstFlux = []
        lst_d_Flux = []
        df_col = []  # column of the dataframe
        ll = 0  # line index
        for line in fhandle:
            if ll == 0:  # check which direction the tally is
                lLine = line.split()  # split the line into list
                x = float(lLine[0])
                y = float(lLine[1])
                z = float(lLine[2])
                if x > 0.0:  # west direction
                    df_col.append('W ' + '%.0f' % x)
                elif x < 0.0:  # east direction
                    x = abs(x)
                    df_col.append('E ' + '%.0f' % x)
                elif z > 0.0:  # north direction
                    df_col.append('N ' + '%.0f' % z)
                elif z < 0.0:  # south direction
                    z = abs(z)
                    df_col.append('S ' + '%.0f' % z)
            if ll > 0:
                line = line.rstrip().split()
                flux = line[1]
                d_flux = line[2]
                lstFlux.append(float(flux))
                lst_d_Flux.append(float(d_flux))
            if ll > 0 and energyMade == False:
                energy = line[0]
                df_energy.append(float(energy))
            ll = ll + 1
        df[df_col[0]] = lstFlux
        df_err[df_col[0]] = lst_d_Flux
        energyMade = True
        fhandle.close()


    # print(df_cols)
    df_energy = np.array(df_energy)
    df_energy_orig = np.copy(df_energy)
    e_diff = np.diff(df_energy)
    for ii in range(0, len(df_energy_orig)):
        if ii == 0:  # first element
            df_energy[ii] = -1
        else:
            this = df_energy_orig[ii]
            prev = df_energy_orig[ii - 1]
            res = prev + abs(this - prev)/2
            df_energy[ii] = res


    df['energy'] = df_energy
    df_err['energy'] = df_energy



    """
    2.) Use LB6411 manual to get the response in the LB6411 for that energy, R(E) 
    Get M(E) = R(E) * flux(E)
    M(E) is the counts in LB6411 per source particle for each energy
    """
    # LB6411 manual
    lstLB6411_E = []  # float
    lstLB6411_Rphi = []  # float
    fhandle = open('LB6411_energy_Rphi.txt', 'r')
    for line in fhandle:
        lstLine = line.split()
        lstLB6411_E.append(float(lstLine[0]))
        lstLB6411_Rphi.append(float(lstLine[1]))
    fhandle.close()

    # inter = np.interp([2.5], lstLB6411_E, lstLB6411_Rphi)[0]

    # response from the manual
    R = df['energy'].apply(lambda x: np.interp([x], lstLB6411_E, lstLB6411_Rphi)[0])

    df['R'] = R
    df = df.set_index('energy')




    # plot flux for one position
    x = df.index
    y = df['N 20'].values

    y_err = (df_err['N 20'].values) * y

    plt.rc('text', usetex=True)
    plt.rc('font', weight='bold')
    rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
    #plt.rc('font', family='serif')
    fig = plt.figure(figsize=(6,4))

    ax1 = fig.add_subplot(1, 1, 1)

    fig.subplots_adjust(left=0.32, right=0.87, top=0.85, bottom=0.25)
    ax1.errorbar(x, y, yerr=y_err, color='black', fmt="-o", markersize=5, linewidth=2, ecolor='red', elinewidth=2)
    fig.subplots_adjust(left=0.14)
    plt.xlabel(r'Energy [MeV]', fontsize=18)
    ax1.set_ylabel(r'\textbf{F4 tally} ($\phi$)', color='black', rotation=0, fontsize=18)
    ax1.yaxis.set_label_coords(0,1.05)
    # pyplot.xscale('log')
    plt.yscale('log')
    ax1.yaxis.set_ticks([1e-6, 1e-4])
    ax1.tick_params('y', labelsize=18)
    ax1.tick_params('x', labelsize=18)
    ax1.spines['top'].set_visible(False)

    # ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1e7))
    # ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))

    # other axis
    ax2 = ax1.twinx()
    ax2.plot(lstLB6411_E, lstLB6411_Rphi, color='orange', ls="-", marker='s', markersize=5, linewidth=2)

    ax2.set_ylabel(r'\textbf{Probe response (R)}', color='orange', rotation=0, fontsize=18)
    ax2.yaxis.set_label_coords(0.91,1.185)
    ax2.tick_params('y', colors='orange', labelsize=18)
    ax2.yaxis.set_ticks([0, 1.4])
    plt.xlim(-0.2, 3.2)
    
    ax2.spines['top'].set_visible(False)
    # for tick in ax1.xaxis.get_major_ticks():
    #    tick.label.set_fontsize(14)
    # for tick in ax1.yaxis.get_major_ticks():
    #    tick.label.set_fontsize(14)
    # for tick in ax2.yaxis.get_major_ticks():
    #    tick.label.set_fontsize(14)
    # ticks
    x_ticks = [0, 3.0]
    plt.xticks(x_ticks, x_ticks)


    # plt.legend(loc='best', title=r"\textbf{Legend}")
    plt.savefig(path_MCNP_inputFolder + '/flux_tally_N20_pres.png', dpi=1600)
    plt.savefig('H://hkromer//06_Presentation//2017-11-24 LabKo//flux_tally_N20_pres.svg', dpi=1200)
    
    # plt.show()
    plt.close()


print('Done with all runs ', IDruns)