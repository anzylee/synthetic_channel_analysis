import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

######################################################################################
# Errors
df = pd.read_csv('F:/synthetic_channel_analysis/joint_distributions/DKL.csv')
# Heatmap
save_fig_path = 'F:/synthetic_channel_analysis/joint_distributions/'
plt.figure()
plt.title('DKL')
df = df.pivot('site_name','version','KLDs')
df = df.reindex(["sfe_322", "sfe_82", "sfe_4523",
                       "sfe_81", "sfe_24", "sfe_25", "sfe_316",
                       "sfe_2248", "sfe_221", "sfe_209"])
ax = sns.heatmap(df, cmap="Blues_r")
plt.savefig(save_fig_path + 'error_heatmap.png')
