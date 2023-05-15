import os, sys
import pandas as pd
import numpy as np
from simpledbf import Dbf5
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from py_modules.stats import *
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
site_names = ['sfe_322', 'sfe_82', 'sfe_4523',
              'sfe_81', 'sfe_24', 'sfe_25', 'sfe_316',
              'sfe_2248', 'sfe_221', 'sfe_209']
site_names = ['sfe_322', 'sfe_25', 'sfe_209']
versions = ['TS', 'n0', 's0', 's1', 's2']

font_size_ticks = 15
font_size_label = 17
font_size_legend = 13

save_dbf = 1
close_figure = 1
arcpy.env.overwriteOutput = True
#########################

KLDs, case_names, fish_periods = [], [], []
ind_fish = 0
colors = ["tab:blue", "tab:blue", "tab:blue", "tab:orange", "tab:orange","tab:orange", "tab:green"]
lines = ['-', ':', '--', '-.', '-']
fish_periods_all = ["chsp", "chfr", "chju"]
pre_fish_period = 'na'


for site_name in site_names:
    for fish_period in fish_periods_all:
        ind = 0
        hist1, hist2 = [], []
        for version in versions:
            case_name = site_name + '_' + version
            print(case_name)
            print(fish_period)
            xticks = []
            timeseries_path = os.path.join(RA_path, "00_Flows/" + case_name + "/flow_series_" + case_name + "_sample.xlsx")
            figure_path = os.path.join(analysis_path, "patch_analysis/")

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

            if version in ['TS']:
                case_name = site_name

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
            env.workspace = os.path.join(RA_path, "SHArC/SHArea/Rasters_" + case_name + "/no_cover")
            BfQ_csi = "csi_" + fish_period + fGl.write_Q_str(Flow[0]) + ".tif"

            if save_dbf == 1:
                # Check out the ArcGIS Spatial Analyst extension license
                arcpy.CheckOutExtension("Spatial")

                # Execute ExtractValuesToPoints
                rasters = arcpy.ListRasters("*", "tif")

                for raster in rasters:
                    if raster == BfQ_csi:
                        print(raster)

                        outRas = Raster(BfQ_csi) > 0.9

                        outPolygons = "BfQ_polygon.shp"
                        arcpy.RasterToPolygon_conversion(outRas, outPolygons)

                        # Set local variables
                        inZoneData = outPolygons
                        zoneField = "id"
                        inClassData = outPolygons
                        classField = "id"
                        outTable = "BfQ_polygon_table_" + fish_period + ".dbf"
                        # outTable = "BfQ_polygon_table.dbf"
                        processingCellSize = 0.01

                        # Execute TabulateArea
                        TabulateArea(inZoneData, zoneField, inClassData, classField, outTable,
                                     processingCellSize, "CLASSES_AS_ROWS")
                        #
                        BfQ_area_dbf = Dbf5(env.workspace + '\\' + outTable)
                        BfQ_partial_area = BfQ_area_dbf.to_dataframe()
            else:
                BfQ_area_dbf = Dbf5(env.workspace + '\\' + outTable)
                BfQ_partial_area = BfQ_area_dbf.to_dataframe()

            if pre_fish_period != fish_period:
                BfQ_real = BfQ_partial_area["Area"]
                BfQ_real = np.log(BfQ_real)
            else:
                BfQ_synthetic = BfQ_partial_area["Area"]
                BfQ_synthetic = np.log(BfQ_synthetic)

                #if BfQ_real.max() > BfQ_synthetic.max():
                #    hist1 = np.histogram(BfQ_real, bins='fd',
                #                         density=True)
                #    hist2 = np.histogram(BfQ_synthetic,
                #                         density=True, bins=hist1[1],
                #                         range=(BfQ_real.min(), BfQ_real.max()))
                #else:
                #    hist2 = np.histogram(BfQ_synthetic, bins='fd',
                #                         density=True)
                #    hist1 = np.histogram(BfQ_real, bins=hist2[1],
                #                         range=(BfQ_synthetic.min(), BfQ_synthetic.max()),
                #                         density=True)

                #hist1 = np.histogram(BfQ_real, bins='fd', density=True)
                #hist2 = np.histogram(BfQ_synthetic, bins='fd', density=True)

                KLD, hist1, edges1, hist2, edges2 = Kullback_div_1d(BfQ_real, BfQ_synthetic, 'fd')
                KLDs = np.append(KLDs, KLD)
                case_names.append(case_name)
                fish_periods.append(fish_period)

                #MSE_tmp = np.square(np.subtract(hist1[0],hist2[0])).mean()
                #RMSE_tmp = np.sqrt(MSE_tmp)
                #RMSE.append(RMSE_tmp)
                plt.figure()
                # plt.hist(BfQ_real, bins=hist1[1], density=True, alpha=0.5)
                # plt.hist(BfQ_synthetic, bins=hist2[1], density=True, alpha=0.5)
                plt.hist([BfQ_real, BfQ_synthetic], bins=edges1, density=True)
                #plt.title("sfe " + str(site) + ", " + fish_period_full)
                plt.xlabel("Patch area $(m^{2})$", fontsize=font_size_label)
                plt.ylabel("PDF", fontsize=font_size_label)
                plt.legend(["Surveyed", "Scenario "+version], fontsize=font_size_legend)

                xticks_log = plt.xticks()[0]
                for ii in range(xticks_log.__len__()):
                    tmp = '10$^{%.0f}$' % xticks_log[ii]
                    xticks.append(tmp)
                #plt.xticks(xticks_log[0], np.str(np.power(10,xticks_log[0])),fontsize=font_size_ticks)
                plt.xticks(xticks_log, xticks, fontsize=font_size_ticks)
                plt.yticks(fontsize=font_size_ticks)

                if not os.path.isdir('./patch_analysis/%s' % (site_name)):
                    os.mkdir('./patch_analysis/%s' % (site_name))

                plt.savefig('./patch_analysis/%s/%s_%s.svg' % (site_name, fish_period, version))
                plt.savefig('./patch_analysis/%s/%s_%s.png' % (site_name, fish_period, version))
                #plt.savefig(figure_path + case_name + '_patch_' + fish_period + '_select.pdf')
                if close_figure == 1:
                    plt.close('all')
                # del BfQ_area_dbf
                # del BfQ_partial_area
                # del BfQ_area

                # arcpy.Delete_management(outPolygons)
                # arcpy.Delete_management(outTable)

            ######################
            pre_fish_period = fish_period
            ind += 1
        ind_fish += 1

datasets = [case_names, fish_periods, KLDs]
datasets = np.array(datasets).transpose()
df = pd.DataFrame(data = datasets, columns=['case_name', 'fish_period','KLD'])
df.to_csv('./patch_analysis/DKL_fd_10bins.csv')
df.to_excel('./patch_analysis/DKL_fd_10bins.xlsx')