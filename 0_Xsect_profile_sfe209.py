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
xlim = [-2, 22]
ylim = [992, 1004]
xticks = [0, 10, 20]
yticks = [994, 998, 1002]
font_size = 18

xsecttab1 = 'F:/ArcPro_projects/sfe_209/transect/stack_profile.dbf'
xsectdbf1 = simpledbf.Dbf5(xsecttab1)
xsectdfst1 = xsectdbf1.to_dataframe()
xsectdfst1['FIRST_DIST'][xsectdfst1['LINE_ID'] == 0]+=3

plt.figure()
sns.lineplot(data=xsectdfst1,
             x="FIRST_DIST", y="FIRST_Z", hue="LINE_ID", style="LINE_ID",
             dashes=[(2,0), (2,0), (2,2), (2,2)], palette=['r', 'k', 'r', 'k'])
plt.xticks()
sns.color_palette()
plt.xticks(xticks, fontsize=font_size)
plt.yticks(yticks, fontsize=font_size)
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig('F:/synthetic_channel_analysis/Xsect_profile/sfe_209_surveyed.svg')

########################################################################################
xsecttab2 = 'F:/usu-RiverBuilder/samples/sfe_209/sfe_209_s0/sfe_209_s0/transect/stack_profile.dbf'
xsectdbf2 = simpledbf.Dbf5(xsecttab2)
xsectdfst2 = xsectdbf2.to_dataframe()
xsectdfst2['FIRST_Z']=xsectdfst2['FIRST_Z']+401

plt.figure()
sns.color_palette()
sns.lineplot(data=xsectdfst2,
             x="FIRST_DIST", y="FIRST_Z", hue="LINE_ID", style='LINE_ID',
             dashes=[(2,0), (2,0), (2,2), (2,2)], palette=['r', 'k', 'r', 'k'])
plt.xticks(xticks, fontsize=font_size)
plt.yticks(yticks, fontsize=font_size)
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig('F:/synthetic_channel_analysis/Xsect_profile/sfe_209_synthetic_s0.svg')
