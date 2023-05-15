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
xlim = [-5, 85]
ylim = [997.5, 1002]
xticks = [0, 20, 40, 60, 80]
yticks = [998, 1000, 1002]
font_size = 18

xsecttab1 = 'F:/ArcPro_projects/sfe_25/transect/stack_profile.dbf'
xsectdbf1 = simpledbf.Dbf5(xsecttab1)
xsectdfst1 = xsectdbf1.to_dataframe()
xsectdfst1['FIRST_DIST'][xsectdfst1['LINE_ID'] == 1]+=(40-14.6)
xsectdfst1['FIRST_DIST'][xsectdfst1['LINE_ID'] == 2]+=(40-23.4)
xsectdfst1['FIRST_DIST'][xsectdfst1['LINE_ID'] == 3]+=(40-13.7)

plt.figure()
sns.lineplot(data=xsectdfst1,
             x="FIRST_DIST", y="FIRST_Z", hue="LINE_ID", style="LINE_ID",
             dashes=[(2,0), (2,0), (2,2), (2,2)], palette=['r', 'k', 'r', 'k'])
sns.color_palette()
plt.xticks(xticks, fontsize=font_size)
plt.yticks(yticks, fontsize=font_size)
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig('F:/synthetic_channel_analysis/Xsect_profile/sfe_25_surveyed.svg')

########################################################################################
xsecttab2 = 'F:/usu-RiverBuilder/samples/sfe_25/sfe_25_s1/sfe_25_s1/transect/stack_profile.dbf'
xsectdbf2 = simpledbf.Dbf5(xsecttab2)
xsectdfst2 = xsectdbf2.to_dataframe()
xsectdfst2['FIRST_DIST'][xsectdfst2['LINE_ID'] == 0]+=3
xsectdfst2['FIRST_DIST'][xsectdfst2['LINE_ID'] == 2]+=2.2
xsectdfst2['FIRST_DIST'][xsectdfst2['LINE_ID'] == 3]+=3.5
xsectdfst2['FIRST_Z']=xsectdfst2['FIRST_Z']+400

plt.figure()
sns.color_palette()
sns.lineplot(data=xsectdfst2,
             x="FIRST_DIST", y="FIRST_Z", hue="LINE_ID", style='LINE_ID',
             dashes=[(2,0), (2,0), (2,2), (2,2)], palette=['r', 'k', 'r', 'k'])
plt.xticks(xticks, fontsize=font_size)
plt.yticks(yticks, fontsize=font_size)
plt.xlim(xlim)
plt.ylim(ylim)
plt.savefig('F:/synthetic_channel_analysis/Xsect_profile/sfe_25_synthetic_s1.svg')
