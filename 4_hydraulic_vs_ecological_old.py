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

hyd_errors = pd.read_csv('F:/synthetic_channel_analysis/errors/errors_CDF_Final.csv')
Total_NRMSD = []
hyd_errors = hyd_errors.drop(np.arange(8,16))
hyd_errors = hyd_errors.reset_index()

for ii in np.arange(0,hyd_errors.__len__(),8):
    ind_depth = np.arange(ii,ii+4)
    ind_velocity = np.arange(ii+4, ii+8)
    Total_NRMSD_tmp = hyd_errors.loc[ind_depth, "RMSE"].reset_index() + \
                      hyd_errors.loc[ind_velocity, "RMSE"].reset_index()
    Total_NRMSD = np.append(Total_NRMSD, Total_NRMSD_tmp["RMSE"])

eco_errors = pd.read_csv('F:/synthetic_channel_analysis/habitat_curves/errors.csv')
Error_spawning = []
Error_fry = []
Error_juvenile = []
site_name = []
version = []
Error_ind = 5 # 3 for RMSE,

for ii in np.arange(0,eco_errors.__len__()):
    if eco_errors.iloc[ii,2][-1]=='g':
        Error_spawning = np.append(Error_spawning, eco_errors.iloc[ii,Error_ind])
        tmp = eco_errors.iloc[ii,1]
        site_name.append(tmp.split('_')[0]+'_'+tmp.split('_')[1])
        version.append(tmp.split('_')[2])
    elif eco_errors.iloc[ii,2][-1]=='y':
        Error_fry = np.append(Error_fry, eco_errors.iloc[ii, Error_ind])
    else:
        Error_juvenile = np.append(Error_juvenile, eco_errors.iloc[ii, Error_ind])

plt.figure()
plt.plot(Total_NRMSD, Error_spawning, 'o')
plt.title("Spawning")
plt.figure()
plt.plot(Total_NRMSD, Error_fry, 'o')
plt.title("Fry")
plt.figure()
plt.plot(Total_NRMSD, Error_juvenile, 'o')
plt.title("Juvenile")

datasets = [site_name, version, Total_NRMSD, Error_spawning, Error_fry, Error_juvenile]
datasets = np.array(datasets).transpose()

hyd_eco_error = pd.DataFrame(data = datasets,
                             columns=['site_name', 'version', 'Total_NRMSD',
                                      'Error_spawning', 'Error_fry', 'Error_juvenile'])


## Which one was the best?
hyd_eco_error.to_csv('./hyd_eco_error/errors.csv')