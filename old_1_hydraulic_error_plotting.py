import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

######################################################################################
# Errors in CDFs

errors_PDF = pd.read_csv('F:/synthetic_channel_analysis/errors/errors_PDF.csv')
errors_CDF = pd.read_csv('F:/synthetic_channel_analysis/errors/errors_CDF.csv')
versions = ['TS', 'n0', 's0', 's1', 's2']
barplot = 0
boxplot = 0

# error plots
for error_type in ["RMSE", "NRMSD", "R2", "Max_dist"]:
    for var in ["Depth", "Velocity"]:
        for type in ['CDF']:
            errors = eval('errors_'+type)
            errors[error_type] = pd.to_numeric(errors[error_type], downcast="float")
            ind_var = errors['Variable']==var
            error_points = errors[['Site_name','Version',error_type]][ind_var]
            error_mean = errors[ind_var].groupby(['Variable', 'Version'])[error_type].mean()
            error_std = errors[ind_var].groupby(['Variable', 'Version'])[error_type].std()
            error_max = errors[ind_var].groupby(['Variable', 'Version'])[error_type].max()-error_mean
            error_min = errors[ind_var].groupby(['Variable', 'Version'])[error_type].min()-error_mean

            error_bar = error_std
            error_bar = np.vstack([-error_min, error_max])

            # Depth
            if barplot == 1:
                fig, ax = plt.subplots()
                ax.bar(versions[1:], error_mean, yerr=error_bar,
                       align='center', alpha=0.5, ecolor='black', capsize=10)

                if error_type == "R2":
                    plt.ylim([0.7, 1])
                plt.title(var)
                plt.ylabel(error_type)

                save_fig_title = error_type + '_' +  var
                save_fig_path = 'F:/synthetic_channel_analysis/errors/barplot/'

                plt.savefig(save_fig_path + save_fig_title + '.png')

            if boxplot == 1:
                plt.figure()
                sns.boxplot(data=error_points, x='Version', y=error_type)
                sns.stripplot(data=error_points, x='Version', y=error_type)
                if error_type == "R2":
                    plt.ylim([0.7, 1])
                plt.title(var)
                plt.ylabel(error_type)

                save_fig_title = error_type + '_' + var
                save_fig_path = 'F:/synthetic_channel_analysis/errors/scatterplot/'

                plt.savefig(save_fig_path + save_fig_title + '.png')

# NRMSD of velocity and depth
errors = errors_CDF
errors_depth = errors[errors["Variable"]=="Depth"]
errors_velocity = errors[errors["Variable"]=="Velocity"]
sum_NRMSD_values = errors_depth["NRMSD"].values + errors_velocity["NRMSD"].values

sum_NRMSD = errors_depth # copy the names and versions
sum_NRMSD["sum_NRMSD"]=sum_NRMSD_values

sum_NRMSD = sum_NRMSD[sum_NRMSD.Site_name != "sfe_95"]
error_heatmap = sum_NRMSD.pivot("Site_name","Version","sum_NRMSD")
error_heatmap = error_heatmap.reindex(["sfe_322", "sfe_82", "sfe_4523",
                       "sfe_81", "sfe_24", "sfe_25", "sfe_316",
                       "sfe_2248", "sfe_221", "sfe_209"])

# Heatmap
save_fig_path = 'F:/synthetic_channel_analysis/errors/'
plt.figure()
plt.title('Normalized RMSD')
ax = sns.heatmap(error_heatmap, cmap="Blues_r")
plt.savefig(save_fig_path + 'error_heatmap.png')

# boxplot
plt.figure()
sns.boxplot(data=sum_NRMSD, x='Version', y="sum_NRMSD")
sns.stripplot(data=sum_NRMSD, x='Version', y="sum_NRMSD") #, hue='Channel Type')
if error_type == "R2":
    plt.ylim([0.7, 1])
plt.title('Normalized RMSD')
plt.ylabel('sum NRMSD')
plt.savefig(save_fig_path + 'error_boxplot.png')