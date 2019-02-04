import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob
import re
from scipy.interpolate import interp1d
from scipy.integrate import simps
from numpy import trapz

sys.path.append('//fs03/LTH_Neutimag/hkromer/02_Simulations/01_Python/MCNP_useful_functions/')
# sys.path.append('/home/hippo/Documents/01_PhD/01_Python/MCNP_useful_functions/')
from fun_MCNP_read_tally_data import *


def fun_fraction_of_emissions_in_angle(deutEnergy, # deuterium ion energy in keV
    emission_angle  # MCNP emission angle (full cone angle, not the half one!)
    ):

    # Deuterium ion energy
    # deutEnergy = 100  #keV

    # datafile for the absolute angular emission angle
    xsDatafile = '//fs03/LTH_Neutimag/hkromer/02_Simulations/01_Python/MCNP_geometry//pyMCNPscript_V000//NeutronSourceDefinition//relative_yield_DD.csv'
    # xsDatafile = '/home/hippo/Documents/01_PhD/01_Python/MCNP_geometry/pyMCNPscript_V000/NeutronSourceDefinition/relative_yield_DD.csv'
    energy = deutEnergy / 1000 # convert deuterium energy to MeV

    # read the csv-File from Benoit.
    # discretization for the source
    angle_start = 0.0
    angle_stop = 180.0  # +1 is added when the function is called
    angle_step = 0.5

    # ----------------------
    # ---------------------- Yield vs emission angle in lab System from Benoits csv (Lisiki)
    # ----------------------
    def fun_intPol(x_target, x1, x2, y1, y2):
        myData = 0  # desired y value
        m = 0
        c = 0
        # y1 = m*x1+c
        # y2 = m*x2+c
        # y2-y1 = m(x2-x1)
        m = (y2 - y1) / (x2 - x1)
        c = y2 - m * x2
        myData = m * x_target + c
        return myData

    def fun_getAngularYield(xsDatafile, E_target, angle):


        # returns x,y : x is the xvals (angle in radian), y is the xs data
        #       angle is the angle in emission radian, intp_Y_av is the neutron energy in MeV for MCNP
        # E_target is the deuterium ion beam energy in MeV

        fhandle = open(xsDatafile, 'r')
        lst_xsData = []  # list that contains the d_sig / d_omega data
        lst_xsAngle = []  # list that contains the angle in radian
        lst_xsEnergy = []  # list that contains the energy in MeV
        i = 0  # line number
        for line in fhandle:
            lst_line = line.split(',')
            lst_line = list(map(float, lst_line))  # convert scientific string to float
            if i is 0:
                lst_xsEnergy = lst_line
            lst_xsAngle.append(lst_line[0])
            lst_xsData.append(lst_line[1:])

            i = i + 1
        fhandle.close()
        lst_xsEnergy = lst_xsEnergy[1:7]  # remove the first element because that is zero, go only till 1.0 MeV
        lst_xsAngle = lst_xsAngle[1:]  # remove the first element because that is zero
        lst_xsData = lst_xsData[1:]  # remove first line because that is the energy, go only till 1.0 MeV

        # transpose list
        lst_xsData = list(map(list, zip(*lst_xsData)))
        # go only till 1.0 MeV
        lst_xsData = lst_xsData[:6]

        # print(lst_xsEnergy)
        # print(lst_xsAngle)
        # print(lst_xsData)

        # Interpolation
        lst_xsDataInterp = []  # interpolated data
        xvals = np.linspace(0.0, np.pi, 1000)
        for line in lst_xsData:
            j = lst_xsData.index(line)
            X = lst_xsAngle
            Y = line
            # interpolate values
            yinterp = np.interp(xvals, X, Y)
            lst_xsDataInterp.append(yinterp)

        # # plot
        # leg = []
        # for line in lst_xsData:
        #     j = lst_xsData.index(line)
        #     X = lst_xsAngle
        #     Y = line
        #     leg.append(str(lst_xsEnergy[j]) + ' MeV')
        #     # plot datapoints
        #     plt.plot(X,Y, 'o', label=leg[j] )
        #     plt.plot(xvals, lst_xsDataInterp[j], '--')
        #
        # plt.ylabel('d_sig / d_omega [MeV/strrad] ')
        # plt.xlabel('Emission angle [rad]')
        # plt.legend(loc='upper right')
        # plt.grid(True)
        # plt.show()

        def fun_getIntpolXSdata(E_target, E_data, intpolXSdata):
            myData = []
            # E_target is the energy in MeV that is wanted
            # E_data is the energy in MeV that is in the data
            # intpolXSdata is the interpolated xs data that is available
            if E_target > 1.0:
                print('No data available for energies beyond 1.0 MeV')
            else:
                # check if the energy is exactly the one in the original definition
                for e0 in E_data:
                    if e0 == E_target:
                        j = E_data.index(e0)
                        myData = intpolXSdata[j]
                        return myData
                if E_data[0] > E_target:
                    # below 0.00035 MeV
                    j = 0
                    for item in intpolXSdata[j]:
                        myData.append(fun_intPol(E_target, 0.0, E_data[j], 0.0, item))
                    return myData
                else:
                    # in  between two energies
                    for i in range(0, len(E_data) - 1):
                        # print(E_data[i])
                        # print(E_target)
                        # print((E_data[i] < E_target))
                        if (E_data[i + 1] > E_target) and (E_data[i] < E_target):
                            # in between two energies
                            k = 0
                            for item in intpolXSdata[i]:
                                myData.append(fun_intPol(E_target, E_data[i], E_data[i + 1], item, intpolXSdata[i + 1][k]))
                                k = k + 1
                            # print(E_data[i])
                            # print(E_data[i + 1])
                            return myData

        x = xvals
        y = fun_getIntpolXSdata(E_target, lst_xsEnergy, lst_xsDataInterp)

        # lookup the xs data for each emission angle --> interpolate between the two closest values
        def find_nearest(array, value):
            idx = (np.abs(array - value)).argmin()
            return idx, array[idx]

        # find where in the x data (emission angle) the angle that is desired for the MCNP file are located
        idxAngle = []
        for val in angle:
            idx, nearest = find_nearest(x, val)
            idxAngle.append(idx)  # append nearest value where that angle is located in the datafile

        # interpolate between the angles: take the interval between two values
        idxAngleCenter = []
        for ii in range(0, len(idxAngle) - 1):
            v2 = idxAngle[ii + 1]
            v1 = idxAngle[ii]
            v3 = round((v1 + v2) / 2, 0)
            idxAngleCenter.append(v3)

        # interpolate between the angles: intervals where to evaluate the angles
        idxAngleRange = []
        for ii in range(0, len(idxAngle)):
            if ii is 0:
                idxAngleRange.append([0, idxAngleCenter[ii]])
            elif ii < len(idxAngle) - 1:
                idxAngleRange.append([idxAngleCenter[ii - 1], idxAngleCenter[ii]])
            else:
                idxAngleRange.append([idxAngleCenter[ii - 1], idxAngle[-1]])

        # print(idxAngle)
        # print(idxAngleCenter)
        # print(idxAngleRange)

        # interpolate between the angles! for example 0 - 10 degree
        intp_Y_av = []  # [-1.0] * len(angle)# interpolated yield values as the average between the two angles (absolute, but not normalized!)
        for pair in idxAngleRange:
            # ii = np.where(idxAngleRange == pair)[0][0]
            val1 = int(pair[0])
            val2 = int(pair[1])
            res = np.mean(y[val1:val2])
            intp_Y_av.append(res)


        

        # # Check the interpolation
        # E_target = 0.1
        # a = fun_getIntpolXSdata(E_target, lst_xsEnergy, lst_xsDataInterp)
        # # plot
        # leg = []
        # for line in lst_xsData[:-1]:
        #     j = lst_xsData.index(line)
        #     X = lst_xsAngle
        #     Y = line
        #     leg.append(str(lst_xsEnergy[j]) + ' MeV')
        #     # plot datapoints
        #     plt.plot(X,Y, 'o', label=leg[j] )
        
        # plt.plot(xvals, a, '--', label=str(E_target)+ ' MeV' )
        # plt.ylabel('d_sig / d_omega [MeV/strrad] ')
        # plt.xlabel('Emission angle [rad]')
        # plt.legend(loc='upper right')
        # plt.grid(True)
        # plt.show()

        return x, y, angle, intp_Y_av

    angle_rad = list(map(np.cos, np.arange(angle_start, angle_stop + angle_step, angle_step)))
    angle = np.arange(angle_start, angle_stop + angle_step, angle_step)
    # convert to radian
    angle = (angle / 360.0) * 2 * np.pi
    mu = np.cos(angle)  # this is the discretization chosen for the MCNP source probability

    x, y, angle, intp_Y_av = fun_getAngularYield(xsDatafile, energy,angle)  # x is interpolated emission angle values in radian, y is xs data (d_sig / d_omega per sterrad)



    # interpolate
    interp_Y_av = interp1d(angle, intp_Y_av, kind='cubic')  # angle is in radian

    # for numpoints in [100,1e3,1e4,1e6,5e6]:
    # interpolate with 1e6 points
    X = np.linspace(0,np.pi, 1e6)
    Y = interp_Y_av(X)

    # f is the ratio of intp_Y_av(0->1°) / intp_Y_av(0->180°). 1° because the total angle of emission is 2°, so 1° in the half circle
    # divide total neutron emission angle by 2 and convert to radians
    # emission_angle = 2
    this_angle = np.radians(emission_angle/2)
    idx = (np.abs(X-this_angle)).argmin()  # find closest this angle
    # print(X[idx], this_angle)

    area_cone = np.trapz(Y[0:idx+1], x=X[0:idx+1] )
    area_tot =  np.trapz(Y, x=X)
    f =  area_cone / area_tot # f by trapz

    # # print(f'f is {f}, Cone area is {area_cone}, total area is {area_tot}')
    # # # plot datapoints
    # plt.plot(X,Y, 'o', alpha=0.25)
    # plt.scatter(X[0:idx+1], Y[0:idx+1], color='red')
    # plt.ylabel('d_sig / d_omega [MeV/strrad] ')
    # plt.xlabel('Emission angle [rad]')
    # plt.grid(True)
    # plt.show()

    return f 

fun_fraction_of_emissions_in_angle(100, 2)