import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

######################################################################################
what_fish = 'Chinnok_Salmon' #'Rainbow_steelhead'

# Errors in
errors_habitat = pd.read_csv('F:/synthetic_channel_analysis/errors.csv')
errors_habitat = errors_habitat[errors_habitat.Site_name != "sfe_95_n0"]

errors_habitat['Period'] = errors_habitat['Fish and period'].str.split("-").str[-1].str[1:]
BfQ = pd.read_csv('F:/synthetic_channel_analysis/BfQ.csv')

errors_habitat = errors_habitat[errors_habitat['Fish and period'].str[0]==what_fish[0]]
BfQ = BfQ[BfQ['Fish and period'].str[0]==what_fish[0]]
BfQ = BfQ[BfQ['Site_name'].str.count('_')==1]

error_heatmap = errors_habitat.pivot("Site_name","Period","R2")
error_heatmap = error_heatmap.reindex(["sfe_322_s2", "sfe_82_s2", "sfe_4523_s2",
                       "sfe_81_s1", "sfe_24_s2", "sfe_25_s1", "sfe_316_s0",
                       "sfe_2248_s1", "sfe_221_s2", "sfe_209_s0"])
error_heatmap = error_heatmap.reindex(["spawning", "fry", "juvenile"], axis="columns")
#error_heatmap = error_heatmap.dropna()
error_heatmap = error_heatmap.fillna(0)

plt.figure()
plt.title('R2 between habitat curves')
ax = sns.heatmap(error_heatmap, cmap="Blues") #norm=LogNorm())
plt.savefig('F:/synthetic_channel_analysis/habitat_curves/RMSE_heatmap.png')

# boxplot
#error_boxplot = errors_habitat[errors_habitat.Site_name != "sfe_82_n0"]
error_boxplot = errors_habitat.dropna()
f, (ax1, ax2) = plt.subplots(ncols=1, nrows=2,
                             sharex=True)
ax1 = sns.boxplot(data=error_boxplot, x='Period', y="R2", ax=ax1)
ax1 = sns.stripplot(data=error_boxplot, x='Period', y="R2", ax=ax1)
ax2 = sns.stripplot(data=error_boxplot, x='Period', y="R2", ax=ax2)

ax1.set_ylim(0.7, 1.)
ax2.set_ylim(0.0, 0.3)
ax1.get_xaxis().set_visible(False)
ax1.set_ylabel("")
ax2.set_ylabel("")
ax1.xaxis.tick_top()
ax2.xaxis.tick_bottom()
ax2.yaxis.set_ticks([0, 0.1, 0.2, 0.3])

#plt.ylabel('R2')
plt.savefig('F:/synthetic_channel_analysis/habitat_curves/error_boxplot.png')