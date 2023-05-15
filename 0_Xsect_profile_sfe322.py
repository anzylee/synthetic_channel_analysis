import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import simpledbf
import warnings
warnings.filterwarnings("ignore")

######################################################################################
xlim = [-5, 65]
ylim = [992, 1006]
xticks = [0, 20, 40, 60]
yticks = [992, 998, 1004]
font_size = 18

xsecttab1 = 'F:/ArcPro_projects/sfe_322/transect/profile.dbf'
xsectdbf1 = simpledbf.Dbf5(xsecttab1)
xsectdfst1 = xsectdbf1.to_dataframe()

plt.figure()
sns.lineplot(data=xsectdfst1,
             x="FIRST_DIST", y="FIRST_Z", hue="LINE_ID", style="LINE_ID",
             dashes=[(2,0), (2,0), (2,2), (2,2)], palette=['r', 'k', 'r', 'k'])
sns.color_palette()
plt.xticks(xticks, fontsize=font_size)
plt.yticks(yticks, fontsize=font_size)
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig('F:/synthetic_channel_analysis/Xsect_profile/sfe_322_surveyed.svg')

"""
xsecttab2 = 'F:/usu-RiverBuilder/samples/sfe_322/sfe_322_vv1/sfe_322_vv1/transect/profile.dbf'
xsectdbf2 = simpledbf.Dbf5(xsecttab2)
xsectdfst2 = xsectdbf2.to_dataframe()
xsectdfst2['FIRST_Z']=xsectdfst2['FIRST_Z']+398

plt.figure()
sns.color_palette()
sns.lineplot(data=xsectdfst2,
             x="FIRST_DIST", y="FIRST_Z", hue="LINE_ID", style='LINE_ID',
             dashes=[(2,0), (2,0), (2,2), (2,2)], palette=['r', 'k', 'r', 'k'])
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig('F:/synthetic_channel_analysis/Xsect_profile/sfe_322_synthetic_vv1.svg')
"""

xsecttab2 = 'F:/usu-RiverBuilder/samples/sfe_322/sfe_322_s2/sfe_322_s2/transect/profile.dbf'
xsectdbf2 = simpledbf.Dbf5(xsecttab2)
xsectdfst2 = xsectdbf2.to_dataframe()
xsectdfst2['FIRST_Z']=xsectdfst2['FIRST_Z']+398

plt.figure()
sns.color_palette()
sns.lineplot(data=xsectdfst2,
             x="FIRST_DIST", y="FIRST_Z", hue="LINE_ID", style='LINE_ID',
             dashes=[(2,0), (2,0), (2,2), (2,2)], palette=['r', 'k', 'r', 'k'])
plt.xticks(xticks, fontsize=font_size)
plt.yticks(yticks, fontsize=font_size)
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig('F:/synthetic_channel_analysis/Xsect_profile/sfe_322_synthetic_s2.svg')