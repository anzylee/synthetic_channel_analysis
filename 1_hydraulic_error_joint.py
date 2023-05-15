import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import openpyxl
import rasterio
from py_modules.stats import *
import os
import warnings
warnings.filterwarnings("ignore")

##############################

site_name_all = ['sfe_322', 'sfe_82', 'sfe_4523',
              'sfe_81', 'sfe_24', 'sfe_25','sfe_316',
              'sfe_2248', 'sfe_221', 'sfe_209']
version_all = ['TS', 'n0', 's0', 's1', 's2']
xmax = [7, 0.6, 0.5,
        2, 2, 2, 2,
        1.5, 2, 2.5]
ymax_ratio = 1.4


version_all = ['TS', 'n0', 's0', 's1', 's2']
site_name_all = ['sfe_322']
xmax = [7]
ymax_ratio = 1.4 # 1.4
"""
site_name_all = ['sfe_25']
xmax = [2]
ymax_ratio = 1 # 1.4

site_name_all = ['sfe_209']
xmax = [2.5]
ymax_ratio = 1 # 1.4
"""

save_csv = 0
joint_dist = 1
save_plot = 1

site_names, versions, flow_conditions = [], [], []
KLDs, max_d, max_v  = [], [], []
method = 'fd'
#curve_type = 'HTC_ra'
#curve_type = 'HTC_co'
curve_type = 'HTC'
flow_condition_all = ['bf'] # 'base'

for flow_condition in flow_condition_all:
    ind = 0
    for site_name in site_name_all:
        for version in version_all:
            num = flow_num(site_name, flow_condition)
            if version in ['TS']:
                d = rasterio.open('F:/tuflow_runs/%s_tuflow/results/clipped/T1_%s_d_clip.tif' % (site_name, str(num)))
                v = rasterio.open('F:/tuflow_runs/%s_tuflow/results/clipped/T1_%s_V_clip.tif' % (site_name, str(num)))
            else:
                d = rasterio.open('F:/tuflow-RB/%s_%s/results/clipped/T1_%s_d_clip.tif' % (site_name, version, str(num)))
                v = rasterio.open('F:/tuflow-RB/%s_%s/results/clipped/T1_%s_V_clip.tif' % (site_name, version, str(num)))
            array_d = d.read(1)
            array_v = v.read(1)
            array_d = np.reshape(array_d, (array_d.shape[0] * array_d.shape[1]))
            array_v = np.reshape(array_v, (array_v.shape[0] * array_v.shape[1]))

            # storing maximum values for plotting purposes
            array_d, array_v = remove_zeros_dv(array_d, array_v)
            max_d = np.append(max_d, array_d.max())
            max_v = np.append(max_v, array_v.max())

            if version in ['TS']:
                array_d_TS = array_d
                array_v_TS = array_v
                #xlim = array_d_TS.max() * 1.2
                xlim = xmax[ind]
                ylim = array_v_TS.max() * ymax_ratio
            else:
                KLD, H_TS, xedges_TS, yedges_TS, H, xedges, yedges \
                    = Kullback_div_2d(array_d_TS, array_v_TS, array_d, array_v, 'fd')
                KLDs = np.append(KLDs, KLD)
                site_names.append(site_name)
                versions.append(version)
                flow_conditions.append(flow_condition)

                # For plotting
                if joint_dist == 1:
                    if not os.path.isdir('./joint_distributions/%s' % (site_name)):
                        os.mkdir('./joint_distributions/%s' % (site_name))

                    if version in ['n0']:
                        #level = np.linspace(0, np.floor(np.quantile(H_TS[H_TS>1e-15], 0.8)*100)/100, 6)
                        H_TS_nonzero = H_TS[H_TS > 1e-15]
                        level = np.quantile(H_TS_nonzero, [0.2, 0.4, 0.6, 0.8])
                        level = np.insert(level, 0, 0)
                        #level = np.insert(level, len(level), H_TS.max())
                        CS = plt_contours(H_TS, xedges_TS, yedges_TS, xlim, ylim, level, curve_type)
                        if save_plot == 1:
                            plt.savefig('./joint_distributions/%s/%s.svg' % (site_name, 'TS'))
                            plt.savefig('./joint_distributions/%s/%s.png' % (site_name, 'TS'))

                    CS = plt_contours(H, xedges, yedges, xlim, ylim, level, curve_type)
                    if save_plot == 1:
                        plt.savefig('./joint_distributions/%s/%s.svg' % (site_name, version))
                        plt.savefig('./joint_distributions/%s/%s.png' % (site_name, version))
                    plt.close('all')
        ind += 1


#for ii in range(1,11):
#    plt.figure(ii)
#    plt.xlim([0, max_d.max()])
#    plt.ylim([0, max_v.max()])

if save_csv == 1:

    datasets = [site_names, versions, flow_conditions, KLDs]
    datasets = np.array(datasets).transpose()
    df = pd.DataFrame(data = datasets, columns=['site_name', 'version', 'flow_condition', 'KLDs'])
    df.to_csv('./joint_distributions/DKL.csv')
    df.to_excel('./joint_distributions/DKL.xlsx')




