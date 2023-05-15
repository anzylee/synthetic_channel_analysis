import numpy as np
import os
import sys
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy import stats
#import arcpy
import seaborn as sns
from scipy.interpolate import interp1d

import warnings
warnings.filterwarnings("ignore")

##############################
case_names = ['sfe_322', 'sfe_95', 'sfe_82', 'sfe_4523',
              'sfe_81', 'sfe_24', 'sfe_25', 'sfe_316',
              'sfe_2248', 'sfe_221', 'sfe_209']
#case_names = ['sfe_209']
versions = ['TS', 'n0', 's0', 's1', 's2']
type_of_cases = ['TS', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB']
var_periods = ['dsp', 'vsp', 'dfr', 'vfr', 'dju', 'vju']

channel_types = [1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7]
lines = ['-', ':', '--', '-.', '-','--','-']
width = [3, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
font_size_ticks = 15
font_size_label = 17
font_size_legend = 13

Q_of_interest = [13] # 1 = 5% bfQ, ..., 13 = 100 % bfQ
Q_calculate_error = [13]

Site_name, Version, Variable, Type = [], [], [], []
#xinterp1, xinterp2 = [], []
xint = []

# Line style
# https://matplotlib.org/2.1.2/api/_as_gen/matplotlib.pyplot.plot.html
##############################

per_num = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100]
ind_case = 0
n = Q_of_interest.__len__()
#colors0 = pl.cm.YlGn(np.linspace(0.4, 1, 13))
#colors = [colors0[-1], colors0[3], colors0[3], colors0[3], colors0[3], colors0[-1], colors0[-1]]
# instead of YlGn - Oranges, PuOr, ...
# https://chrisalbon.com/python/basics/set_the_color_of_a_matplotlib/

for case_name in case_names:
    channel_type = channel_types[case_names.index(case_name)]
    for var_period in var_periods:
        ind = 0
        for version in versions:
            case_v_name = case_name + '_' + version
            print(case_v_name)

            if version[0] == 'T':
                path_d = 'F:/tuflow_runs/' + case_name + '_tuflow/results/clipped/depth_dist.xlsx'
                path_v = 'F:/tuflow_runs/' + case_name + '_tuflow/results/clipped/velocity_dist.xlsx'
            else:
                path_d = 'F:/tuflow-RB/' + case_v_name + '/results/clipped/depth_dist.xlsx'
                path_v = 'F:/tuflow-RB/' + case_v_name + '/results/clipped/velocity_dist.xlsx'
                #path_d = 'F:/tuflow-RB/backup_wo_runway/' + case_v_name + '/results/clipped/depth_dist.xlsx'
                #path_v = 'F:/tuflow-RB/backup_wo_runway/' + case_v_name + '/results/clipped/velocity_dist.xlsx'

            excel_d = pd.read_excel(path_d)
            excel_v = pd.read_excel(path_v)


            if var_period[0]=='d':
                excel = excel_d
                ylabel_txt = 'Depth (m)'
                fignum = ind_case*2

                if var_period[1] == 's':
                    yt1 = 0.49
                    yt2 = 3.30
                elif var_period[1] == 'f':
                    yt1 = 0.27
                    yt2 = 2.11
                else:
                    yt1 = 0.45
                    yt2 = 2.79
            else:
                excel = excel_v
                ylabel_txt = 'Velocity (m/s)'
                fignum = ind_case*2+1

                if var_period[1] == 's':
                    yt1 = 0.8
                    yt2 = 3.66
                elif var_period[1] == 'f':
                    yt1 = 0
                    yt2 = 0.71
                else:
                    yt1 = 0.04
                    yt2 = 2.01

            allcols = list(excel.columns)
            qcols = [k for k in allcols if k[0] == 'd']
            dx = excel.bin[1] - excel.bin[0]
            numsampdist = excel.sum()
            qpropcols = []

            for j in range(len(qcols)):
                newcol0 = qcols[j] + 'prop'
                excel[newcol0] = excel[qcols[j]] / numsampdist[qcols[j]] / dx
                qpropcols.append(newcol0)

                newcol1 = qcols[j] + 'prop_sum'
                excel[newcol1] = np.cumsum(excel[newcol0])*dx
                qpropcols.append(newcol1)

            ind0 = 0

            for jj in Q_of_interest:

                if version[0] == 'T':
                    c_num = ""
                else:
                    c_num = ', ' + case_v_name.split('_')[2]

                xx = eval("excel.d"+str(jj)+"prop_sum")
                yy = excel.bin

                xt1 = np.interp(yt1, yy, xx)
                xt2 = np.interp(yt2, yy, xx)

                Site_name.append(case_name)
                Version.append(version)
                Variable.append(var_period)
                Type.append(channel_type)
                #xinterp1.append(xt1)
                #xinterp2.append(xt2)
                xint = np.append(xint, abs(xt2-xt1))
                ind0 += 1

            ind += 1

    #plt.close('all')
    ind_case += 1

datasets = [Site_name, Version, Variable, xint]
datasets = np.array(datasets).transpose()
xint_CDF = pd.DataFrame(data = datasets,
                        columns = ['Site_name', 'Version', 'Variable', 'xint'])

