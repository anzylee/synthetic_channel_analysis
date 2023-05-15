import os, sys
import pandas as pd
import numpy as np
import simpledbf
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import arcpy
from arcpy import env
from arcpy.sa import *
try:
    sys.path.append("F:/RiverArchitect-development/.site_packages/riverpy")
    import config
    import fGlobal as fGl
except:
    print("ExceptionERROR: Missing RiverArchitect packages (riverpy).")


# Eco series analysis - SHArea
# This python script (1)
# Before you run this code, please put the excel files in SHArea folder to a new folder called "case_name"

#########################
# User defined variables

RA_path = "F:/RiverArchitect-development/"
analysis_path = "F:/synthetic_channel_analysis/"
threshold = 'threshold2'
sites = [322,25,209] #[322,95,82,4523,81,24,25,316,2248,221,209]
sites = [95]
all_sites = [322,95,82,4523,81,24,25,316,2248,221,209]
# ver_1st = ['vv1','n0','n0','vv1','s1','s2','s1','vv1','vv1','vv1','vv1']
ver_1st = ['s2','n0','s2','s2','s1','s2','s1','s0','s1','s2','s0']
font_size_ticks = 15
font_size_label = 17
font_size_legend = 13

interptype = 'linear'
scale_to_one = 0
arcpy.env.overwriteOutput = True
#########################

colors = ["tab:blue", "tab:orange", "tab:blue", "tab:orange", "tab:blue","tab:orange"]
lines = ['-', '-', '--', '-.', '-']
#fish_periods = ["chsp", "chfr", "chju", "rasp", "rafr", "raju", "raad"]
fish_periods = ["rafr", "raju", "cofr", "coju"]

Site_name, Fish_period,BfQ_area, BfQ_Site_name, BfQ_Fish_period = [], [], [], [], []
RMSE, NRMSD, R2 = [], [], []

for site in sites:
    ind_fish = 0
    ver_ind = all_sites.index(int(site))
    #case_of_interest = ['sfe_'+str(site), 'sfe_'+str(site) + '_' + ver_1st[ver_ind]]
    #type_of_cases = ['Surveyed', 'RB', 'RB']
    case_of_interest = ['sfe_'+str(site)]
    type_of_cases = ['Surveyed']
    RB_tmp, TS_tmp = [], []
    for fish_period in fish_periods:
        ind = 0
        for case_name in case_of_interest:
            print(case_name)
            print(fish_period)
            timeseries_path = os.path.join(RA_path, "00_Flows/" + case_name + "/flow_series_" + case_name + "_sample.xlsx")
            figure_path = os.path.join(analysis_path, "habitat_curves")

            fish_name = fish_period[0:2]
            period = fish_period[2:4]

            if fish_name == 'ch':
                fish_full = 'Chinook Salmon'
            elif fish_name == 'ra':
                fish_full = 'Rainbow / Steelhead Trout'
            if period == 'sp':
                period_full = 'spawning'
            elif period == 'fr':
                period_full = 'fry'
            elif period == 'ju':
                period_full = 'juvenile'
            elif period == 'ad':
                period_full = 'adult'

            fish_period_full = fish_full + ' - ' + period_full

            sharea_path = os.path.join(RA_path, "SHArC/SHArea/" + case_name + "_sharea_" + fish_name + period + ".xlsx")

            ######################
            # Reading SHARrC data

            f1 = pd.read_excel(sharea_path, index_col=None, header=None,usecols="B")[3:].values.tolist()
            f2 = pd.read_excel(sharea_path, index_col=None, header=None,usecols="F")[3:].values.tolist()

            Flow = np.array(f1).transpose()[0]
            CalArea = np.array(f2).transpose()[0]

            Flow = np.append(Flow, [0])
            CalArea = np.append(CalArea, [0])

            ######################
            # Bankfull wetted area
            env.workspace = os.path.join(RA_path, "SHArC/HSI/" + case_name)
            BfQ_hsi = "dsi_" + fish_period + fGl.write_Q_str(Flow[0]) + ".tif"

            # Check out the ArcGIS Spatial Analyst extension license
            arcpy.CheckOutExtension("Spatial")

            # Execute ExtractValuesToPoints
            rasters = arcpy.ListRasters("*", "tif")

            for raster in rasters:
                if raster == BfQ_hsi:
                    print(raster)

                    outRas = Raster(BfQ_hsi) > -1

                    outPolygons = "BfQ_polygon.shp"
                    arcpy.RasterToPolygon_conversion(outRas, outPolygons)

                    # Set local variables
                    inZoneData = outPolygons
                    zoneField = "id"
                    inClassData = outPolygons
                    classField = "id"
                    outTable = "BfQ_polygon_table.dbf"
                    processingCellSize = 0.01

                    # Execute TabulateArea
                    TabulateArea(inZoneData, zoneField, inClassData, classField, outTable,
                                 processingCellSize, "CLASSES_AS_ROWS")

                    BfQ_area_dbf = simpledbf.Dbf5(env.workspace + '\\' + outTable)
                    BfQ_partial_area = BfQ_area_dbf.to_dataframe()
                    BfQ_area_tmp = np.sum(np.array(BfQ_partial_area['Area']))
                    BfQ_area.append(BfQ_area_tmp)
                    BfQ_Site_name.append(case_name)
                    BfQ_Fish_period.append(fish_period_full)
                    del BfQ_area_dbf
                    #del BfQ_area_tmp

                    arcpy.Delete_management(outPolygons)
                    arcpy.Delete_management(outTable)

            # Reverse
            #Flow = Flow[::-1]
            #CalArea = CalArea[::-1]

            # Non-dimensionalization
            print(BfQ_area_tmp)
            Norm_Flow = Flow / Flow[0]
            Norm_CalArea = CalArea / BfQ_area_tmp
            #os.system("pause")

            ######################

            Norm_Flow_new = np.linspace(np.min(Norm_Flow), np.max(Norm_Flow), num=10001, endpoint=True)
            Norm_f = interp1d(Norm_Flow, Norm_CalArea, kind=interptype)
            f = interp1d(Flow, CalArea, kind=interptype)

            plt.figure(ind_fish)
            plt.plot(Norm_Flow, Norm_CalArea, marker="o", color=colors[ind], linewidth=0)
            if type_of_cases[ind] == 'Surveyed':
                plt.plot(Norm_Flow_new, Norm_f(Norm_Flow_new), linestyle=lines[ind], color=colors[ind],
                         linewidth=3, label= 'Surveyed')
                TS_tmp = Norm_CalArea
            else:
                plt.plot(Norm_Flow_new, Norm_f(Norm_Flow_new), linestyle=lines[ind], color=colors[ind],
                         linewidth=3, label= 'Scenario ' + case_name.split('_')[-1])
                RB_tmp = Norm_CalArea
            #plt.title(case_name + ', ' + fish_name + ', ' + period)
            #plt.title(case_name)
            plt.xlabel('Q / Q$_{bf}$', fontsize=font_size_label)
            plt.ylabel('Habitat area / A$_{bf}$', fontsize=font_size_label)
            plt.yticks(fontsize=font_size_ticks)
            plt.xticks(fontsize=font_size_ticks)

            if scale_to_one & (ind == fish_periods.__len__()-1):
                bottom, top = plt.ylim()
                plt.ylim(0, 1.3)
            plt.legend(fontsize=font_size_legend)
            plt.show()
            #plt.savefig(figure_path+case_name+'_'+ fish_name + period +'_Area_Q.svg')

            ind += 1
        ind_fish += 1

        Site_name.append(case_name)
        Fish_period.append(fish_period_full)

        RMSE.append(np.sqrt(np.mean((RB_tmp - TS_tmp) ** 2)))
        NRMSD.append(np.sqrt(np.mean((RB_tmp - TS_tmp) ** 2)) / np.mean(TS_tmp))
        print(TS_tmp)
        print(np.mean(TS_tmp))
        R2.append(np.corrcoef(RB_tmp, TS_tmp)[0, 1] ** 2)

        plt.savefig(os.path.join(figure_path, case_name + '_SHArea_Q_'+fish_period+'_select.png'))
        plt.savefig(os.path.join(figure_path, 'svg', case_name + '_SHArea_Q_'+fish_period+'_select.svg'))
        plt.savefig(os.path.join(figure_path, 'pdf', case_name + '_SHArea_Q_'+fish_period+'_select.pdf'))
        plt.close('all')

datasets = [Site_name, Fish_period, RMSE, NRMSD, R2]
datasets = np.array(datasets).transpose()
errors = pd.DataFrame(data = datasets,
                      columns=['Site_name', 'Fish and period', 'RMSE', 'NRMSD', 'R2'])
#errors.to_csv(os.path.join(analysis_path, 'errors.csv'))

datasets = [BfQ_Site_name, BfQ_Fish_period, BfQ_area]
datasets = np.array(datasets).transpose()
BfQ = pd.DataFrame(data = datasets,
                      columns=['Site_name', 'Fish and period', 'BfQ_area'])
#BfQ.to_csv(os.path.join(analysis_path,'BfQ.csv'))
