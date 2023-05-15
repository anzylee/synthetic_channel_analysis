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
case_names = ['sfe_25']
versions = ['TS', 'n0', 's0', 's1', 's2']
type_of_cases = ['TS', 'RB', 'RB', 'RB', 'RB', 'RB', 'RB']
channel_types = [1, 1, 2, 2, 3, 4, 4, 4, 5, 6, 7]
lines = ['-', ':', '--', '-.', '-','--','-']
width = [3, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
font_size_ticks = 15
font_size_label = 17
font_size_legend = 13

Q_of_interest = [13] # 1 = 5% bfQ, ..., 13 = 100 % bfQ
Q_calculate_error = [13]

Site_name, Version, Variable, Type = [], [], [], []
RMSE_PDF, NRMSD_PDF, R2_PDF, Max_dist_PDF = [], [], [], []
RMSE_CDF, NRMSD_CDF, R2_CDF, Max_dist_CDF = [], [], [], []

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
    for var in ['Depth', 'Velocity']:
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


            if var[0]=='D':
                excel = excel_d
                ylabel_txt = 'Depth (m)'
                fignum = ind_case*2
                colors = np.array(
                    [[0, 0, 0, 255], [162, 235, 127, 255], [119, 205, 134, 255], [37, 141, 65, 255],  [17, 80, 41, 255]]) / 255
            else:
                excel = excel_v
                ylabel_txt = 'Velocity (m/s)'
                fignum = ind_case*2+1
                colors = np.array(
                    [[90, 0, 3, 255], [255, 204, 123, 255], [230, 163, 70, 255],  [180, 90, 40, 255], [125, 30, 11, 255]]) / 255

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

                plt.figure(fignum)
                ax1 = plt.subplot(121)
                x1 = eval("excel.d"+str(jj)+"prop")
                y1 = excel.bin

                ax1.plot(x1, y1, label=type_of_cases[ind]+c_num+str(per_num[jj-1])+"% bfd",
                         color=colors[ind], linestyle=lines[ind], linewidth=width[ind])
                plt.ylabel(ylabel_txt, fontsize=font_size_label)
                plt.xlabel('PDF', fontsize=font_size_label)
                plt.yticks(fontsize=font_size_ticks)
                plt.xticks(fontsize=font_size_ticks)

                if ind == versions.__len__()-1:
                    plt.gca().invert_xaxis()
                    if var == 'Depth':
                        plt.gca().invert_yaxis()

                ax2 = plt.subplot(122)
                x2 = eval("excel.d"+str(jj)+"prop_sum")
                y2 = excel.bin

                ax2.plot(x2, y2, label=type_of_cases[ind] + c_num,
                         color=colors[ind], linestyle=lines[ind], linewidth=width[ind])
                if ind == versions.__len__()-1:
                    if var == 'Depth':
                        plt.gca().invert_yaxis()
                plt.yticks([])
                plt.xticks([0, 0.5, 1], fontsize=font_size_ticks)
                plt.xlabel('CDF', fontsize=font_size_label)
                plt.legend(fontsize=font_size_legend)
                mpl.rcParams["legend.labelspacing"] = 0.1

                ind0 += 1
                #plt.set_cmap("Greens")
                #plt.suptitle(case_name+', '+var)

                # interpolating CDFs to calculate RMSE, R2, Max_dist
                x1[x1.__len__()] = 1
                y1[x1.__len__()] = 10
                f1 = interp1d(y1, x1)
                y1_new = np.linspace(0, 10, num=1000)
                x1_new = f1(y1_new)

                x2[x2.__len__()] = 1
                y2[x2.__len__()] = 10
                f2 = interp1d(y2, x2)
                y2_new = np.linspace(0, 10, num=1000)
                x2_new = f2(y2_new)

                #plt.legend()
                if jj == Q_calculate_error[0]:
                    if version[0] == 'T':
                        TS_x1 = x1_new
                        TS_x2 = x2_new
                        print(var)
                    else:
                        Site_name.append(case_name)
                        Version.append(version)
                        Variable.append(var)
                        Type.append(channel_type)

                        RMSE_PDF.append(np.sqrt(np.mean((x1_new - TS_x1) ** 2)))
                        NRMSD_PDF.append(np.sqrt(np.mean((x1_new - TS_x1) ** 2))/np.mean(TS_x1))
                        R2_PDF.append(np.corrcoef(x1_new,TS_x1)[0,1] ** 2)
                        Max_dist_PDF.append(np.max(np.abs(x1_new-TS_x1)))

                        RMSE_CDF.append(np.sqrt(np.mean((x2_new - TS_x2) ** 2)))
                        NRMSD_CDF.append(np.sqrt(np.mean((x2_new - TS_x2) ** 2)) / np.mean(TS_x1))
                        R2_CDF.append(np.corrcoef(x2_new, TS_x2)[0, 1] ** 2)
                        Max_dist_CDF.append(np.max(np.abs(x2_new - TS_x2)))

                #plt.figure()
                #plt.plot(y2_new, TS_x2, y2_new, x2_new)

                save_fig_title = 'Distribution ' + str(var) + ' Prop selected' + '_' + case_name + '_'
                save_fig_path = 'F:/synthetic_channel_analysis/distributions/'

                if os.path.isdir(save_fig_path) == 0:
                    os.mkdir(save_fig_path)
                plt.savefig(save_fig_path + save_fig_title + str(jj)+'.png')
                plt.savefig(save_fig_path + save_fig_title + str(jj)+ '.pdf')
                plt.savefig(save_fig_path + save_fig_title + str(jj)+ '.svg')

            ind += 1

    #plt.close('all')
    ind_case += 1

datasets = [Site_name, Version, Type, Variable, RMSE_PDF, NRMSD_PDF, R2_PDF, Max_dist_PDF]
datasets = np.array(datasets).transpose()
errors_PDF = pd.DataFrame(data = datasets,
                      columns=['Site_name', 'Version', 'Channel Type', 'Variable', 'RMSE', 'NRMSD', 'R2', 'Max_dist'])
errors_PDF.to_csv('F:/synthetic_channel_analysis/errors/errors_PDF.csv', index=False)

datasets = [Site_name, Version, Type, Variable, RMSE_CDF, NRMSD_CDF, R2_CDF, Max_dist_CDF]
datasets = np.array(datasets).transpose()
errors_CDF = pd.DataFrame(data = datasets,
                      columns=['Site_name', 'Version', 'Channel Type', 'Variable', 'RMSE', 'NRMSD', 'R2', 'Max_dist'])
errors_CDF.to_csv('F:/synthetic_channel_analysis/errors/errors_CDF.csv', index=False)
