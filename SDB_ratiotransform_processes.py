#################################################################
#                                                               #
#                   Ratio Transform                             #
#            Functions for SDB Tools Plugin                     #
#           Kiyomi Holman, kholm074@uottawa.ca                  #
#                                                               #
#################################################################

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model, cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from osgeo import gdal, ogr, osr
from qgis.core import QgsRasterLayer, QgsMapLayerRegistry
from PyQt4.QtCore import QFileInfo
import os, sys, csv
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import math
import xml.etree.ElementTree as ET
import scipy.stats


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

##### Read files #####
'''Part 1 opens/reads metadata files and returns relevant data and arrays'''

def openGDALDataset(mtl_filename):
    dataset = gdal.Open(mtl_filename)
    if dataset is None:
        sys.exit("Data set open failed: " + mtl_filename)
    return dataset

def getDatasetProperties(dataset):
    # Get the properties of the rasters in dataset
    xsize = dataset.RasterXSize
    ysize = dataset.RasterYSize
    nbands = dataset.RasterCount
    projection = dataset.GetProjection()
    geotransform = dataset.GetGeoTransform()
    return xsize, ysize, nbands, projection, geotransform

def splitDate(date):
    year = int(date.split("-")[0])
    month = int(date.split("-")[1])
    day = int(date.split("-")[2])
    jday = getJday(year, month, day)
    return year, month, day, jday

def getJday(year, month, day):
    if year % 4 != 0 or year == 2000:
        leap = False
    else:
        leap = True

    if month == 1: jday = int(day)
    if month == 2: jday = int(day) + 31
    if month == 3: jday = int(day) + 59
    if month == 4: jday = int(day) + 90
    if month == 5: jday = int(day) + 120
    if month == 6: jday = int(day) + 151
    if month == 7: jday = int(day) + 181
    if month == 8: jday = int(day) + 212
    if month == 9: jday = int(day) + 243
    if month == 10: jday = int(day) + 273
    if month == 11: jday = int(day) + 304
    if month == 12: jday = int(day) + 334
    if leap and month >= 3:
        jday = jday + 1
    return jday

def getNextDateName(date_name):
    year = int(date_name[0:4])
    jday = int(date_name[4:7])

    if jday < 365:
        jday = jday + 1
    else:
        if year % 4 != 0 or year == 2000:
            leap = False
        else:
            leap = True
        if leap and jday == 365:
            jday = jday + 1
        else:
            year = year + 1
            jday = 0

    # Export as string
    if jday >= 100:
        jday_string = str(jday)
    else:
        jday_string = str(jday).zfill(3)

    return str(year) + jday_string

def getPreviousDateName(date_name):
    year = int(date_name[0:4])
    jday = int(date_name[4:7])

    if jday != 1:
        jday = jday - 1
    else:
        if year - 1 % 4 != 0 or year - 1 == 2000:
            leap = False
            year = year - 1
            jday = 365
        else:
            leap = True
            year = year - 1
            jday = 366

    # Export as string
    if jday >= 100:
        jday_string = str(jday)
    else:
        jday_string = str(jday).zfill(3)

    return str(year) + jday_string

def getEarthSunDistance(jday):
    return (1 - 0.01672 * math.cos(deg2rad(0.9856 * (jday - 4))))

def world2Pixel(geotransform, x, y):
    ulX = geotransform[0]
    ulY = geotransform[3]
    xDist = geotransform[1]
    yDist = geotransform[5]
    # rtnX = geotransform[2]
    # rtnY = geotransform[4]
    col = int((x - ulX) / xDist)
    line = int((ulY - y) / yDist * (-1))
    return (col, line)

def deg2rad(degrees):
    return degrees / 180 * np.pi

def writeGeotiff_Float32(fname, data, geo_transform, projection, compress):
    if isinstance(data, np.ndarray):  # If data is an array
        driver = gdal.GetDriverByName('GTiff')
        rows, cols = data.shape
        if compress:
            dataset = driver.Create(fname, cols, rows, 1, gdal.GDT_Float32, options=['COMPRESS=LZW'])
        else:
            dataset = driver.Create(fname, cols, rows, 1, gdal.GDT_Float32)
        dataset.SetGeoTransform(geo_transform)
        dataset.SetProjection(projection)
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
    elif isinstance(data, list):  # If data is a list (of arrays)
        driver = gdal.GetDriverByName('GTiff')
        rows, cols = data[0].shape
        if compress:
            dataset = driver.Create(fname, cols, rows, len(data), gdal.GDT_Float32, options=['COMPRESS=LZW'])
        else:
            dataset = driver.Create(fname, cols, rows, len(data), gdal.GDT_Float32, )
        dataset.SetGeoTransform(geo_transform)
        dataset.SetProjection(projection)
        for i in range(len(data)):
            #if debug:
            #    print("Writing array #: " + str(i + 1))
            band = dataset.GetRasterBand(i + 1)
            band.WriteArray(data[i])
    dataset = None  # Close the file
    return 0

def writeGeotiff_Byte(fname, data, geo_transform, projection, compress):
    if isinstance(data, np.ndarray):  # If data is an array
        driver = gdal.GetDriverByName('GTiff')
        rows, cols = data.shape
        if compress:
            dataset = driver.Create(fname, cols, rows, 1, gdal.GDT_Byte, options=['COMPRESS=LZW'])
        else:
            dataset = driver.Create(fname, cols, rows, 1, gdal.GDT_Byte)
        dataset.SetGeoTransform(geo_transform)
        dataset.SetProjection(projection)
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
    elif isinstance(data, list):  # If data is a list (of arrays)
        driver = gdal.GetDriverByName('GTiff')
        rows, cols = data[0].shape
        if compress:
            dataset = driver.Create(fname, cols, rows, len(data), gdal.GDT_Byte, options=['COMPRESS=LZW'])
        else:
            dataset = driver.Create(fname, cols, rows, len(data), gdal.GDT_Byte)
        dataset.SetGeoTransform(geo_transform)
        dataset.SetProjection(projection)
        for i in range(len(data)):
            band = dataset.GetRasterBand(i + 1)
            band.WriteArray(data[i])
    dataset = None  # Close the file
    return 0

def read_mtl_landsat(mtl_filename, bandnumber):
    '''Goes through the  landsat metadata file and looks for the specified bandname, reflectance multiplier and reflectance add
    constants, and returns all them along with the bands file name.'''
    txt_f = open(mtl_filename, "r")
    for line in txt_f:
        splitline = line.split()
        firststring = splitline[0]
        if firststring == "FILE_NAME_BAND_" + str(bandnumber):
            BAND = str(splitline[2]).strip('"')
        if firststring == "REFLECTANCE_MULT_BAND_" + str(bandnumber):
            RMULT = float(splitline[2])
        if firststring == "REFLECTANCE_ADD_BAND_" + str(bandnumber):
            RADD = float(splitline[2])
    filename = os.path.dirname(mtl_filename) + "/" + BAND
    #  Dictionary
    read_dict_L8 = {"filename": filename, "bandname": BAND, "rmult": RMULT, "radd": RADD}
    return read_dict_L8

def read_mtl_S2(mtl_filename):
    '''Reads the Sentinel 2 landsat metadata file and looks for the specified band number, and finds the DN
    for TOA Reflectance'''
    tree = ET.parse(mtl_filename)
    root = tree.getroot()
    metadataPath = mtl_filename.split("/")[:-1]  # Remove filename from metadataFile, to get path.
    metadataPath = "/".join(metadataPath) + "/"

    # Get what is in the main file:
    # List of image filepaths
    # ESUN
    # List of solar irradiances
    # Year, month, day, jday, hr
    # SceneCentre
    imagefiles = []
    bandname = []
    ESUN = []

    for child in root:
        if child.tag.split('}')[1] == "General_Info":
            root2 = child
            for child2 in root2:
                if child2.tag == "Product_Info":
                    root3 = child2
                    for child3 in root3:
                        if child3.tag == "PRODUCT_START_TIME":
                            year, month, day, jday = splitDate(child3.text[0:10])
                            hours = int(child3.text[11:13])
                            minutes = int(child3.text[14:16])
                            seconds = int(child3.text[17:19])
                            hr = float(hours) + float(minutes) / 60 + float(seconds) / 3600
                        if child3.tag == "Product_Organisation":
                            root4 = child3
                            for child4 in root4:
                                if child4.tag == "Granule_List":
                                    root5 = child4
                                    for child5 in root5:
                                        if child5.tag == "Granule":
                                            root6 = child5
                                            for child6 in root6:
                                                if child6.tag == "IMAGE_FILE":
                                                    imagefiles.append(child6.text)
                                            imagefiles = sorted(imagefiles)
                if child2.tag == "Product_Image_Characteristics":
                    root3 = child2
                    for child3 in root3:
                        if child3.tag == "QUANTIFICATION_VALUE":
                            quantification = child3.text
                        if child3.tag == "Reflectance_Conversion":
                            root4 = child3
                            for child4 in root4:
                                if child4.tag == "U":
                                    U = float(child4.text)
                                if child4.tag == "Solar_Irradiance_List":
                                    root5 = child4
                                    for child5 in root5:
                                        ESUN.append(float(child5.text))

        if child.tag.split('}')[1] == "Geometric_Info":
            root2 = child
            for child2 in root2:
                if child2.tag == "Product_Footprint":
                    root3 = child2
                    for child3 in root3:
                        if child3.tag == "Product_Footprint":
                            root4 = child3
                            for child4 in root4:
                                if child4.tag == "Global_Footprint":
                                    root5 = child4
                                    for child5 in root5:
                                        if child5.tag == "EXT_POS_LIST":
                                            coordinates = child5.text.rstrip().split(" ")
                                            latitudes = []
                                            longitudes = []
                                            for i in range(len(coordinates)):
                                                if i % 2 == 0:
                                                    latitudes.append(float(coordinates[i]))
                                                else:
                                                    longitudes.append(float(coordinates[i]))
                                            sceneCentre = [np.mean(np.asarray(latitudes)),
                                                           np.mean(np.asarray(longitudes))]

    # Then get path to xml file for the granule
    additionalPath = "/".join(imagefiles[0].split("/")[0:2])  # Get the additional path, and
    granulePath = metadataPath + additionalPath  # add it
    for file in os.listdir(granulePath):
        if file.endswith(".xml"):
            granuleMetadataFile = granulePath + "/" + file

    # Get sza and saa
    tree = ET.parse(granuleMetadataFile)
    root = tree.getroot()

    for child in root:
        if child.tag.split('}')[1] == "Geometric_Info":
            root2 = child
            for child2 in root2:
                if child2.tag == "Tile_Angles":
                    root3 = child2
                    for child3 in root3:
                        if child3.tag == "Mean_Sun_Angle":
                            root4 = child3
                            for child4 in root4:
                                if child4.tag == "ZENITH_ANGLE":
                                    sza = float(child4.text)
                                elif child4.tag == "AZIMUTH_ANGLE":
                                    saa = float(child4.text)

    # Add full filepath and extension to image band paths
    for i in range(len(imagefiles)):
        imagefiles[i] = metadataPath + imagefiles[i] + ".jp2"
        #Strips filename off end to access later on
        #does this change the filename??
        bandname.append(imagefiles[i].split("/")[-1])
        #print type(bandname[2])
    metadata = [imagefiles, bandname, saa, sza, ESUN, U, quantification, year, month, day, jday, hr, sceneCentre]
    return metadata

def read_mtl_WV2(mtl_filename):
    tree = ET.parse(mtl_filename)
    root = tree.getroot()

    metadataPath = mtl_filename.split("/")[:-1]  # Remove filename from metadataFile, to get path.
    metadataPath = "/".join(metadataPath) + "/"
    imagefiles = []

    for child in root:
        if child.tag == "IMD":
            root2 = child
            for child2 in root2:
                if child2.tag == "BAND_C":
                    root3 = child2
                    for child3 in root3:
                        if child3.tag == "ABSCALFACTOR":
                            abscal_1 = float(child3.text)
                        if child3.tag == "EFFECTIVEBANDWIDTH":
                            effective_bandwidth_1 = float(child3.text)
                if child2.tag == "BAND_B":
                    root3 = child2
                    for child3 in root3:
                        if child3.tag == "ABSCALFACTOR":
                            abscal_2 = float(child3.text)
                        if child3.tag == "EFFECTIVEBANDWIDTH":
                            effective_bandwidth_2 = float(child3.text)
                if child2.tag == "BAND_G":
                    root3 = child2
                    for child3 in root3:
                        if child3.tag == "ABSCALFACTOR":
                            abscal_3 = float(child3.text)
                        if child3.tag == "EFFECTIVEBANDWIDTH":
                            effective_bandwidth_3 = float(child3.text)
                if child2.tag == "BAND_Y":
                    root3 = child2
                    for child3 in root3:
                        if child3.tag == "ABSCALFACTOR":
                            abscal_4 = float(child3.text)
                        if child3.tag == "EFFECTIVEBANDWIDTH":
                            effective_bandwidth_4 = float(child3.text)
                if child2.tag == "BAND_R":
                    root3 = child2
                    for child3 in root3:
                        if child3.tag == "ABSCALFACTOR":
                            abscal_5 = float(child3.text)
                        if child3.tag == "EFFECTIVEBANDWIDTH":
                            effective_bandwidth_5 = float(child3.text)
                if child2.tag == "BAND_RE":
                    root3 = child2
                    for child3 in root3:
                        if child3.tag == "ABSCALFACTOR":
                            abscal_6 = float(child3.text)
                        if child3.tag == "EFFECTIVEBANDWIDTH":
                            effective_bandwidth_6 = float(child3.text)
                if child2.tag == "BAND_N":
                    root3 = child2
                    for child3 in root3:
                        if child3.tag == "ABSCALFACTOR":
                            abscal_7 = float(child3.text)
                        if child3.tag == "EFFECTIVEBANDWIDTH":
                            effective_bandwidth_7 = float(child3.text)
                if child2.tag == "BAND_N2":
                    root3 = child2
                    for child3 in root3:
                        if child3.tag == "ABSCALFACTOR":
                            abscal_8 = float(child3.text)
                        if child3.tag == "EFFECTIVEBANDWIDTH":
                            effective_bandwidth_8 = float(child3.text)
                if child2.tag == "IMAGE":
                    root3 = child2
                    for child3 in root3:
                        if child3.tag == "FIRSTLINETIME":
                            time = child3.text
                            year = int(time[0:4])
                            month = int(time[5:7])
                            day = int(time[8:10])
                            hours = int(time[11:13])
                            minutes = int(time[14:16])
                            seconds = int(time[17:19])
                            hr = float(hours) + float(minutes) / 60 + float(seconds) / 3600
                            jday = getJday(year, month, day)
                        if child3.tag == "MEANSUNEL":
                            sza = 90 - float(child3.text)
                        if child3.tag == "MEANSUNAZ":
                            saa = float(child3.text)
                        if child3.tag == "MEANSATAZ":
                            vaa = float(child3.text)
                        if child3.tag == "MEANOFFNADIRVIEWANGLE":
                            vza = float(child3.text)

        if child.tag == "TIL":
            root2 = child
            for child2 in root2:
                if child2.tag == "TILE":
                    root3 = child2
                    for child3 in root3:
                        if child3.tag == "FILENAME":
                            imagefiles.append(os.path.dirname(mtl_filename) + "/" + child3.text)
                        if child3.tag == "ULLON":
                            ulx = float(child3.text)
                        if child3.tag == "ULLAT":
                            uly = float(child3.text)
                        if child3.tag == "URLON":
                            urx = float(child3.text)
                        if child3.tag == "URLAT":
                            ury = float(child3.text)
                        if child3.tag == "LRLON":
                            lrx = float(child3.text)
                        if child3.tag == "LRLAT":
                            lry = float(child3.text)
                        if child3.tag == "LLLON":
                            llx = float(child3.text)
                        if child3.tag == "LLLAT":
                            lly = float(child3.text)

    abscals = [abscal_1, abscal_2, abscal_3, abscal_4, abscal_5, abscal_6, abscal_7, abscal_8]
    effective_bandwidths = [effective_bandwidth_1, effective_bandwidth_2, effective_bandwidth_3, effective_bandwidth_4,
                            effective_bandwidth_5, effective_bandwidth_6, effective_bandwidth_7, effective_bandwidth_8]
    calibration = [abscal_1 / effective_bandwidth_1, abscal_2 / effective_bandwidth_2, abscal_3 / effective_bandwidth_3,
                   abscal_4 / effective_bandwidth_4, abscal_5 / effective_bandwidth_5, abscal_6 / effective_bandwidth_6,
                   abscal_7 / effective_bandwidth_7, abscal_8 / effective_bandwidth_8]
    sceneCentre = [np.mean([uly, ury, lry, lly]), np.mean([ulx, urx, lrx, llx])]
    metadata = [imagefiles, calibration, saa, sza, vaa, vza, year, month, day, jday, hr, sceneCentre]
    return metadata

def read_raster_landsat(read_dict_L8):
    '''Opens the raster file and reads it as an array, also it gets some basic information from the raster file.'''
    raster_filename = read_dict_L8['filename']
    raster_ds = gdal.Open(raster_filename)
    if raster_ds is None:
        sys.exit(1)
    xsize = raster_ds.RasterXSize
    ysize = raster_ds.RasterYSize
    nbands = raster_ds.RasterCount
    projection = raster_ds.GetProjection()
    geotransform = raster_ds.GetGeoTransform()
    #  Get raster srs
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(projection)
    #  Read the bands
    bands = []
    print("read_raster_L8() is now appending " + str(range(nbands)) + " bands to a list.")
    for i in range(nbands):
        bands.append(raster_ds.GetRasterBand(i + 1))
    #  Read data to array
    array = []
    for i in range(nbands):
        array.append(bands[i].ReadAsArray(0, 0, xsize, ysize))
    #  check if data are in integers or floats
    if np.issubdtype(array[0].dtype, np.integer) is True:
        integerv = True
    else:
        integerv = False

    #  Dictionary
    raster_dict_L8 = {"xsize": xsize, "ysize": ysize, "nbands": nbands, "projection": projection,
                       "geotransform": geotransform, "rastersrs": raster_srs, "filename": raster_filename,
                       "rasterds": raster_ds, "array": array}
    return raster_dict_L8

def read_raster_S2(bluegreendict):
    '''Opens the raster file and reads it as an array, also it gets some basic information from the raster file.'''
    raster_filename = bluegreendict['filename']
    raster_ds = gdal.Open(raster_filename)
    if raster_ds is None:
        sys.exit(1)
    xsize = raster_ds.RasterXSize
    ysize = raster_ds.RasterYSize
    nbands = raster_ds.RasterCount
    projection = raster_ds.GetProjection()
    geotransform = raster_ds.GetGeoTransform()
    #  Get raster srs
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(projection)
    #  Read the bands
    bands = []
    print("read_raster_S2() is now appending " + str(range(nbands)) + " bands to a list.")
    for i in range(nbands):
        bands.append(raster_ds.GetRasterBand(i + 1))
    #  Read data to array
    array = []
    for i in range(nbands):
        array.append(bands[i].ReadAsArray(0, 0, xsize, ysize))
    #  check if data are in integers or floats
    if np.issubdtype(array[0].dtype, np.integer) is True:
        integerv = True
    else:
        integerv = False

    #  Dictionary
    raster_dict_S2 = {"xsize": xsize, "ysize": ysize, "nbands": nbands, "projection": projection,
                       "geotransform": geotransform, "rastersrs": raster_srs, "filename": raster_filename,
                       "rasterds": raster_ds, "array": array}

    print("read_raster_S2() has created a dictionary. The value of the rasterds key is: " + str(raster_ds))
    return raster_dict_S2

def read_raster_WV2(bluegreendict):
    '''Opens the raster file and reads it as an array, also it gets some basic information from the raster file.'''
    raster_filename = bluegreendict['filename']
    raster_ds = gdal.Open(raster_filename)
    if raster_ds is None:
        sys.exit(1)
    xsize = raster_ds.RasterXSize
    ysize = raster_ds.RasterYSize
    nbands = raster_ds.RasterCount
    projection = raster_ds.GetProjection()
    geotransform = raster_ds.GetGeoTransform()
    #  Get raster srs
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(projection)
    #  Read the bands
    bands = []
    for i in range(nbands):
        bands.append(raster_ds.GetRasterBand(i + 1))
    #  Read data to array
    array = []
    for i in range(nbands):
        array.append(bands[i].ReadAsArray(0, 0, xsize, ysize))
    #  check if data are in integers or floats
    if np.issubdtype(array[0].dtype, np.integer) is True:
        integerv = True
    else:
        integerv = False

    #  Dictionary
    raster_dict_WV2 = {"xsize": xsize, "ysize": ysize, "nbands": nbands, "projection": projection,
                       "geotransform": geotransform, "rastersrs": raster_srs, "filename": raster_filename,
                       "rasterds": raster_ds, "array": array}
    return raster_dict_WV2

def TOA_refl_landsat(read_dict_L8, raster_dict_L8, outputfolder):
    '''Creates a Top Of Atmosphere reflectance image'''
    # Calculate TOA reflectance
    TOA_refl_landsat = ((raster_dict_L8['array'][0] * read_dict_L8['rmult']) + read_dict_L8['radd'])
    #  Write result to file
    toa_raster_filename = outputfolder + "/" + read_dict_L8['bandname'].split(".TIF")[0] + "_TOA.tif"
    toa_bandname = read_dict_L8['bandname'].split(".TIF")[0] + "_TOA.tif"
    raster_driver = gdal.GetDriverByName("GTiff")
    toa_raster_dataset = raster_driver.Create(toa_raster_filename, raster_dict_L8['xsize'], raster_dict_L8['ysize'],
                                              1, gdal.GDT_Float32)
    toa_raster_dataset.SetGeoTransform(raster_dict_L8['geotransform'])
    toa_raster_dataset.SetProjection(raster_dict_L8['projection'])
    band = toa_raster_dataset.GetRasterBand(1)
    band.WriteArray(TOA_refl_landsat)
    toa_raster_dataset = None

    toa_LandsatDict = {"toabandname": toa_bandname, 'filename': toa_raster_filename, "xsize": raster_dict_L8['xsize'],
                "ysize": raster_dict_L8['ysize'], "projection": raster_dict_L8['projection'],
                "geotransform": raster_dict_L8['geotransform']}
    return toa_LandsatDict

def TOA_refl_S2(imagefile, bandname, outfolder):
    '''Creates a Top Of Atmosphere reflectance image'''
    # Calculate TOA reflectance
    #Sentinel data already in TOA Refl, just need to divide the DN by 10000
    dataset = openGDALDataset(imagefile)
    '''for i in range(nbands):
        #dataset = openGDALDataset(imagefile)
        xsize, ysize, fileNbands, projection, GT = getDatasetProperties(dataset)
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(dataset.GetProjection())
        band = dataset.getRasterBand(1)'''
        #dataset = openGDALDataset(imagefile)
    xsize, ysize, fileNbands, projection, GT = getDatasetProperties(dataset)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(dataset.GetProjection())
    band = dataset.GetRasterBand(1)
    #ulXPixel, ulYPixel = world2Pixel(GT, minX, maxY)
    #lrXPixel, lrYPixel = world2Pixel(GT, maxX, minY)
    DN = band.ReadAsArray(0, 0, xsize, ysize).astype(float)
    DN[DN == 0] = np.nan
    TOA_refl_S2 = DN / 10000

    #  Write result to file
    toa_bandname = bandname.split(".jp2")[0] + "_TOA.tif"
    toa_raster_filename = outfolder + "/" + bandname.split(".jp2")[0] + "_TOA.tif"
    raster_driver = gdal.GetDriverByName("GTiff")
    toa_raster_dataset = raster_driver.Create(toa_raster_filename, xsize, ysize,
                                              1, gdal.GDT_Float32)
    toa_raster_dataset.SetGeoTransform(GT)
    toa_raster_dataset.SetProjection(projection)
    toa_band = toa_raster_dataset.GetRasterBand(1)
    toa_band.WriteArray(TOA_refl_S2)
    toa_raster_dataset = None

    toa_dict_S2 = {"toabandname": toa_bandname, 'filename': toa_raster_filename, "nbands": fileNbands, "xsize": xsize,
                   "ysize": ysize, "rastersrs": raster_srs, "rasterds": dataset, "projection": projection,
                   "geotransform": GT, }
    return toa_dict_S2

def TOA_refl_WV2(imagefile, outfolder):
    '''Creates a Top Of Atmosphere reflectance image'''
    nbands = 5

    for i in range(nbands):
        dataset = openGDALDataset(imagefile)
        xsize, ysize, fileNbands, projection, GT = getDatasetProperties(dataset)
        raster_srs = osr.SpatialReference()
        raster_srs.ImportFromWkt(dataset.GetProjection())
        band = dataset.getRasterBand(i + 1)

    DN = band.ReadAsArray(0, 0, xsize, ysize).astype(float)
    DN[DN == 0] = np.nan
    #  Write result to file
    toa_bandname = bandname.split(".tif")[0] + "_TOA.tif"
    toa_raster_filename = outfolder + "/" + bandname.split(".tif")[0] + "_TOA.tif"
    raster_driver = gdal.GetDriverByName("GTiff")
    toa_raster_dataset = raster_driver.Create(toa_raster_filename, xsize, ysize,
                                              1, gdal.GDT_Float32)
    toa_raster_dataset.SetGeoTransform(GT)
    toa_raster_dataset.SetProjection(projection)
    toa_band = toa_raster_dataset.GetRasterBand(1)
    toa_band.WriteArray(TOA_refl_WV2)
    toa_raster_dataset = None

    toa_dict_WV2 = {"toabandname": toa_bandname, 'filename': toa_raster_filename, "nbands": fileNbands, "xsize": xsize,
                   "ysize": ysize, "rastersrs": raster_srs, "rasterds": dataset, "projection": projection,
                   "geotransform": GT, }
    return toa_dict_WV2

def deep_kernel_landsat(read_dict_L8, kernelsize):
    '''Opens the raster file as an array, creates a kernel and evaluates the kernel on the image file it's given.'''
    # find and return bandname
    imagefilename = read_dict_L8['filename']
    imagery = gdal.Open(imagefilename)
    image_array = np.array(imagery.GetRasterBand(1).ReadAsArray())
    # Defines kernel size:
    kernel_size = kernelsize
    #print 'Kernel size =', kernel_size
    # Calculates the size of the image array:
    image_array_shape = image_array.shape
    image_x_size = image_array_shape[1]
    image_y_size = image_array_shape[0]
    #print 'x size =', image_x_size
    #print 'y size =', image_y_size

    # Creates processing kernel and
    kernel = np.zeros((kernel_size, kernel_size))

    # Setting starting point for left corner
    left_corner_x = 0
    left_corner_y = 0

    # Defining LC value & B7_mean for beginning
    darkest_LCX = 0
    darkest_LCY = 0
    darkest_B7 = 999999
    mean_B7 = 999999
    # Tracks kernel across image:
    # while
    while left_corner_y < (image_y_size - kernel_size):
        while left_corner_x < (image_x_size - kernel_size):
            # Populates kernel and calculates mean:
            kernel[0:kernel_size, 0:kernel_size] = image_array[left_corner_y:(left_corner_y + kernel_size), left_corner_x:(left_corner_x + kernel_size)]
            if np.all(kernel[:, :] > 0):
                mean_B7 = np.average(kernel)
                # print mean_B7
                # print 'dark',darkest_B7
                if mean_B7 < darkest_B7:
                    darkest_B7 = mean_B7
                    darkest_LCX = left_corner_x
                    darkest_LCY = left_corner_y
                    # Ticks up the left corner position
            left_corner_x = left_corner_x + kernel_size
        left_corner_x = 0  # Resets us back to the left of the image after completing the row.
        left_corner_y = left_corner_y + kernel_size
    kernelLandsatDict = {"darkestmean": darkest_B7, "lcy": darkest_LCY, "lcx": darkest_LCX, "kernelsize": kernel_size}
    return kernelLandsatDict

def deep_kernel_S2(imagefile, kernelsize):
    '''Opens the raster file as an array, creates a kernel and evaluates the kernel on the image file it's given.'''
    # find and return bandname
    imagefilename = imagefile
    imagery = gdal.Open(imagefilename)
    image_array = np.array(imagery.GetRasterBand(1).ReadAsArray())
    # Defines kernel size:
    kernel_size = kernelsize
    # print 'Kernel size =', kernel_size
    # Calculates the size of the image array:
    image_array_shape = image_array.shape
    image_x_size = image_array_shape[1]
    image_y_size = image_array_shape[0]
    # print 'x size =', image_x_size
    # print 'y size =', image_y_size

    # Creates processing kernel and
    kernel = np.zeros((kernel_size, kernel_size))

    # Setting starting point for left corner
    left_corner_x = 0
    left_corner_y = 0

    # Defining LC value & B7_mean for beginning
    darkest_LCX = 0
    darkest_LCY = 0
    darkest_B11 = 999999
    mean_B11 = 999999
    # Tracks kernel across image:
    # while
    while left_corner_y < (image_y_size - kernel_size):
        while left_corner_x < (image_x_size - kernel_size):
            # Populates kernel and calculates mean:
            kernel[0:kernel_size, 0:kernel_size] = image_array[left_corner_y:(left_corner_y + kernel_size),
                                                   left_corner_x:(left_corner_x + kernel_size)]
            if np.all(kernel[:, :] > 0):
                mean_B11 = np.average(kernel)
                # print mean_B7
                # print 'dark',darkest_B7
                if mean_B11 < darkest_B11:
                    darkest_B11 = mean_B11
                    darkest_LCX = left_corner_x
                    darkest_LCY = left_corner_y
                    # Ticks up the left corner position
            left_corner_x = left_corner_x + kernel_size
        left_corner_x = 0  # Resets us back to the left of the image after completing the row.
        left_corner_y = left_corner_y + kernel_size
    kernel_dict_S2 = {"darkestmean": darkest_B11, "lcy": darkest_LCY, "lcx": darkest_LCX, "kernelsize": kernel_size}
    return kernel_dict_S2

#Not running yet - need to determine which band calibration to use.
def deep_kernel_WV2(nir1dict, kernelsize):
    '''Opens the raster file as an array, creates a kernel and evaluates the kernel on the image file it's given.'''
    # find and return bandname
    imagefilename = nir1dict['filename']
    imagery = gdal.Open(imagefilename)
    image_array = np.array(imagery.GetRasterBand(1).ReadAsArray())
    # Defines kernel size:
    kernel_size = kernelsize
    # print 'Kernel size =', kernel_size
    # Calculates the size of the image array:
    image_array_shape = image_array.shape
    image_x_size = image_array_shape[1]
    image_y_size = image_array_shape[0]
    # print 'x size =', image_x_size
    # print 'y size =', image_y_size

    # Creates processing kernel and
    kernel = np.zeros((kernel_size, kernel_size))

    # Setting starting point for left corner
    left_corner_x = 0
    left_corner_y = 0

    # Defining LC value & nir1_mean for beginning
    #Not the correct bands
    darkest_LCX = 0
    darkest_LCY = 0
    darkest_nir1 = 999999
    mean_nir1 = 999999
    # Tracks kernel across image:
    # while
    while left_corner_y < (image_y_size - kernel_size):
        while left_corner_x < (image_x_size - kernel_size):
            # Populates kernel and calculates mean:
            kernel[0:kernel_size, 0:kernel_size] = image_array[left_corner_y:(left_corner_y + kernel_size),
                                                   left_corner_x:(left_corner_x + kernel_size)]
            if np.all(kernel[:, :] > 0):
                mean_nir1 = np.average(kernel)
                # print mean_nir1
                # print 'dark',darkest_nir1
                if mean_nir1 < darkest_nir1:
                    darkest_nir11 = mean_nir1
                    darkest_LCX = left_corner_x
                    darkest_LCY = left_corner_y
                    # Ticks up the left corner position
            left_corner_x = left_corner_x + kernel_size
        left_corner_x = 0  # Resets us back to the left of the image after completing the row.
        left_corner_y = left_corner_y + kernel_size
    kernel_dict_WV2 = {"darkestmean": darkest_nir1, "lcy": darkest_LCY, "lcx": darkest_LCX, "kernelsize": kernel_size}
    return kernel_dict_WV2

##### Transform that ratio! #####
'''Part 2 performs ln ratio algorithm and returns initial plot of data'''

def log_value(array):
    '''Calculates the logarithm for an array.'''
    log_value = np.log(array)
    inf = np.isinf(log_value)
    log_value[inf] = 0

    return log_value

def log_bluegreen(bluedict, greendict, readshp_dict, fieldname, outputfolder):
    '''Creates the logarithm division raster image blue / green'''
    #  Open the blue band
    blue_image = gdal.Open(bluedict['filename'])
    blue_band = blue_image.GetRasterBand(1)
    gt = blue_image.GetGeoTransform()

    #  Open the green band
    green_image = gdal.Open(greendict['filename'])
    green_band = green_image.GetRasterBand(1)

    blue_values = []
    green_values = []
    depth_values = []

    #Open shapefile to get depths data
    shp_driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = shp_driver.Open(readshp_dict['shpfilename'], 0)
    if shp_driver is None:
        sys.exit("Could not open depth file.")
    else:
        lyr = ds.GetLayer()
        lyrDefn = lyr.GetLayerDefn()
        nfields = lyrDefn.GetFieldCount()
        for i in range(nfields):
            fieldDefn = lyrDefn.GetFieldDefn(i)
            fieldName = fieldDefn.GetName()
            if fieldName == fieldname:  # Ask for this fieldname under plugins instruction
                break
        for feat in lyr:
            # Find all values in depth, blue and green datasets, append to lists.
            depth = feat.GetField(i)
            geom = feat.GetGeometryRef()
            x, y = geom.GetX(), geom.GetY()  # gets coordinates in the map units
            px = int((x - gt[0]) / gt[1])
            py = int((y - gt[3]) / gt[5])

            #print depth, geom, x, y, px, py

            blueval = blue_band.ReadAsArray(px, py, 1, 1)[0][0]
            greenval = green_band.ReadAsArray(px, py, 1, 1)[0][0]

            depth_values.append(depth)
            blue_values.append(blueval)
            green_values.append(greenval)

    depth_values = np.asarray(depth_values)
    blue_values = np.asarray(blue_values)
    green_values = np.asarray(green_values)

    print "looking for best r for constant n..."
    # Looks for best value of r for n in the Stumpf equation and applies it to algorithm.
    #This value is variable and creates the linear relationship between blue and green values.
    best_cor = 0
    best_n = 0
    for j in range(1, 2000):
        bg_ratio = (np.log(j*blue_values))/(np.log(j*green_values))
        cor = np.corrcoef(depth_values, bg_ratio)[0,1]
        if (cor > best_cor):
            best_cor = cor
            best_n = j
    #close all the open files from finding best n so can re-open them to perform the actual function.
    #blue_image = None
    #blue_band = None
    #blueval = None
    #green_image = None
    #green_band = None
    #greenval = None
    #bg_ratio = None

    print ("The best value of n is " + str(best_n) + ".")

    blue_ds = gdal.Open(bluedict['filename'])
    blue_image_array = np.array(blue_ds.GetRasterBand(1).ReadAsArray())
    green_ds = gdal.Open(greendict['filename'])
    green_image_array = np.array(green_ds.GetRasterBand(1).ReadAsArray())

    #  Create the division array using best n value
    blue_green = (np.log(best_n*blue_image_array)) / (np.log(best_n*green_image_array))
    #nan = np.isnan(blue_green)
    #inf = np.isinf(blue_green)

    # Testing way to get rid of inf, ninf, and nan values
    blue_green[blue_green == np.NINF] = 0
    blue_green[blue_green == np.inf] = 0
    blue_green[blue_green == np.nan] = 0

    #  Create the raster file
    blue_green_filename = outputfolder + "/" + "Blue_Green_log.tif"
    blue_green_bandname = "Blue_Green_log.tif"
    raster_driver = gdal.GetDriverByName("GTiff")
    blue_green_dataset = raster_driver.Create(blue_green_filename, bluedict['xsize'], bluedict['ysize'],
                                              1, gdal.GDT_Float32)
    blue_green_dataset.SetGeoTransform(bluedict['geotransform'])
    blue_green_dataset.SetProjection(bluedict['projection'])
    band = blue_green_dataset.GetRasterBand(1)
    band.WriteArray(blue_green)
    blue_green_dataset = None
    blue_green_dict = {"bandname": blue_green_bandname, 'filename': blue_green_filename, "bluegreen_log": blue_green,
                       "geotransform": bluedict['geotransform'], "projection": bluedict['projection']}

    return blue_green_dict

def log_bluegreen_WV2(bluedict, greendict, readshp_dict, fieldname, outputfolder):
    '''Creates the logarithm division raster image blue / green'''
    #  Open the blue band
    blue_image = gdal.Open(bluedict['filename'])
    blue_band = blue_image.GetRasterBand(1)
    print type(blue_band)
    gt = blue_image.GetGeoTransform()

    #  Open the green band
    green_image = gdal.Open(greendict['filename'])
    green_band = green_image.GetRasterBand(1)

    blue_values = []
    green_values = []
    depth_values = []

    #Open shapefile to get depths data
    shp_driver = ogr.GetDriverByName('ESRI Shapefile')
    print type(shp_driver)
    ds = shp_driver.Open(readshp_dict['shpfilename'], 0)
    print type(ds)
    if shp_driver is None:
        sys.exit("Could not open depth file.")
    else:
        lyr = ds.GetLayer()
        lyrDefn = lyr.GetLayerDefn()
        nfields = lyrDefn.GetFieldCount()
        for i in range(nfields):
            fieldDefn = lyrDefn.GetFieldDefn(i)
            fieldName = fieldDefn.GetName()
            if fieldName == fieldname:  # From the plugin instructions.
                break
        for feat in lyr:
            # Find all values in depth, blue and green datasets, append to lists.
            depth = feat.GetField(i)
            geom = feat.GetGeometryRef()
            x, y = geom.GetX(), geom.GetY()  # gets coordinates in the map units
            px = int((x - gt[0]) / gt[1])
            py = int((y - gt[3]) / gt[5])

            blueval = blue_band.ReadAsArray(px, py, 1, 1)[0][0]
            greenval = green_band.ReadAsArray(px, py, 1, 1)[0][0]

            depth_values.append(depth)
            blue_values.append(blueval)
            green_values.append(greenval)

    depth_values = np.asarray(depth_values)
    blue_values = np.asarray(blue_values)
    green_values = np.asarray(green_values)

    # Added by Anders, to get rid of points that have nan in either depth_values OR blue_values OR green_values
    good_values = np.logical_not(np.logical_or(np.isnan(depth_values), np.isnan(blue_values), np.isnan(green_values)))
    depth_values = depth_values[good_values]
    blue_values = blue_values[good_values]
    green_values = green_values[good_values]

    print "looking for best r for constant n..."
    # Looks for best value of r for n in the Stumpf equation and applies it to algorithm.
    #This value is variable and creates the linear relationship between blue and green values.
    best_cor = -1
    best_n = 100
    for j in range(1, 2000):
        bg_ratio = (np.log(j*blue_values))/(np.log(j*green_values))
        cor = np.corrcoef(depth_values, bg_ratio)[0,1]
        if (cor > best_cor):
            best_cor = cor
            best_n = j

    print ("The best value of n is " + str(best_n) + ".")
    print ("The correlation for that value is " + str(best_cor) + ".")

    blue_ds = gdal.Open(bluedict['filename'])
    blue_image_array = np.array(blue_ds.GetRasterBand(1).ReadAsArray())
    green_ds = gdal.Open(greendict['filename'])
    green_image_array = np.array(green_ds.GetRasterBand(1).ReadAsArray())

    #  Create the division array using best n value
    blue_green = (np.log(best_n*blue_image_array)) / (np.log(best_n*green_image_array))
    #nan = np.isnan(blue_green)
    #inf = np.isinf(blue_green)

    # Testing way to get rid of inf, ninf, and nan values
    blue_green[blue_green == np.NINF] = 0
    blue_green[blue_green == np.inf] = 0
    blue_green[blue_green == np.nan] = 0

    #  Create the raster file
    blue_green_filename = outputfolder + "/" + "Blue_Green_log.tif"
    blue_green_bandname = "Blue_Green_log.tif"
    raster_driver = gdal.GetDriverByName("GTiff")
    blue_green_dataset = raster_driver.Create(blue_green_filename, bluedict['xsize'], bluedict['ysize'],
                                              1, gdal.GDT_Float32)
    blue_green_dataset.SetGeoTransform(bluedict['geotransform'])
    blue_green_dataset.SetProjection(bluedict['projection'])
    band = blue_green_dataset.GetRasterBand(1)
    band.WriteArray(blue_green)
    blue_green_dataset = None
    blue_green_dict = {"bandname": blue_green_bandname, 'filename': blue_green_filename, "bluegreen_log": blue_green,
                       "geotransform": bluedict['geotransform'], "projection": bluedict['projection']}

    return blue_green_dict

def bluegreen_kernel_landsat(bluegreen_logarray, kernelLandsatDict):
    '''Uses the deep kernel found and evaluates it in the division array so we obtain a standard deviation
    for the kernel, and a mean value'''
    imagefilename = bluegreen_logarray['filename']
    imagery = gdal.Open(imagefilename)
    image_array = np.array(imagery.GetRasterBand(1).ReadAsArray())
    left_corner_y = kernelLandsatDict['lcy']
    left_corner_x = kernelLandsatDict['lcx']
    kernel_size = kernelLandsatDict['kernelsize']
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[0:kernel_size, 0:kernel_size] = image_array[left_corner_y:(left_corner_y + kernel_size),
                                           left_corner_x:(left_corner_x + kernel_size)]
    #  Get the kernels standard deviation
    if np.all(kernel[:, :] > 0):
        stdev = np.std(kernel)
    #  Get the kernels mean
        bluegreen_mean = np.mean(kernel)
        bluegreen_max = np.max(kernel)
        bluegreen_min = np.min(kernel)

    bgk = {"std": stdev, "mean": bluegreen_mean, "mid": kernel[(kernel_size/2), (kernel_size/2)], "max": bluegreen_max,
           "min": bluegreen_min}
    return bgk

def bluegreen_kernel_S2(bluegreen_logarray, kernel_dict_S2):
    '''Uses the deep kernel found and evaluates it in the division array so we obtain a standard deviation
    for the kernel, and a mean value'''
    imagefilename = bluegreen_logarray['filename']
    imagery = gdal.Open(imagefilename)
    image_array = np.array(imagery.GetRasterBand(1).ReadAsArray())
    left_corner_y = kernel_dict_S2['lcy']
    left_corner_x = kernel_dict_S2['lcx']
    kernel_size = kernel_dict_S2['kernelsize']
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[0:kernel_size, 0:kernel_size] = image_array[left_corner_y:(left_corner_y + kernel_size),
                                           left_corner_x:(left_corner_x + kernel_size)]
    #  Get the kernels standard deviation
    if np.all(kernel[:, :] > 0):
        stdev = np.std(kernel)
    #  Get the kernels mean
        bluegreen_mean = np.mean(kernel)
        bluegreen_max = np.max(kernel)
        bluegreen_min = np.min(kernel)

    bgk = {"std": stdev, "mean": bluegreen_mean, "mid": kernel[(kernel_size/2), (kernel_size/2)], "max": bluegreen_max,
           "min": bluegreen_min}
    return bgk

def bluegreen_kernel_WV2(bluegreen_logarray, kernel_dict_WV2):
    '''Uses the deep kernel found and evaluates it in the division array so we obtain a standard deviation
    for the kernel, and a mean value'''
    imagefilename = bluegreen_logarray['filename']
    imagery = gdal.Open(imagefilename)
    image_array = np.array(imagery.GetRasterBand(1).ReadAsArray())
    left_corner_y = kernel_dict_WV2['lcy']
    left_corner_x = kernel_dict_WV2['lcx']
    kernel_size = kernel_dict_WV2['kernelsize']
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[0:kernel_size, 0:kernel_size] = image_array[left_corner_y:(left_corner_y + kernel_size),
                                           left_corner_x:(left_corner_x + kernel_size)]
    #  Get the kernels standard deviation
    if np.all(kernel[:, :] > 0):
        stdev = np.std(kernel)
    #  Get the kernels mean
        bluegreen_mean = np.mean(kernel)
        bluegreen_max = np.max(kernel)
        bluegreen_min = np.min(kernel)

    bgk = {"std": stdev, "mean": bluegreen_mean, "mid": kernel[(kernel_size/2), (kernel_size/2)], "max": bluegreen_max,
           "min": bluegreen_min}
    return bgk

def read_shp(shpfilename):
    '''Reads and opens the shape file.'''
    #  Open Shapefile
    shp_driver = ogr.GetDriverByName('ESRI Shapefile')
    shp_ds = shp_driver.Open(shpfilename, 1)  # '1' allows us to open the file and modify it
    if shp_ds is None:
        sys.exit(2)
    #  Getting basic information
    layer = shp_ds.GetLayer(0)
    featureCount = layer.GetFeatureCount()
    shp_srs = layer.GetSpatialRef()

    #  Dictionary
    readshp_dict = {"layer": layer, "shpsrs": shp_srs, "shpdriver": shp_driver, "shpds": shp_ds,
                    "featurecount": featureCount}
    return readshp_dict

def reproject_data(shpfilename, shpsrs, rastersrs, shpdriver, layer):
    '''reprojects the data from the shape file reference system to the image reference system.'''
    if shpsrs.IsSame(rastersrs):
        shp_filename = shpfilename
    else:         #  maybe else?
        CoordTransform = osr.CoordinateTransformation(shpsrs, rastersrs)
        #  Creating new shapefiles
        shp_filename = shpfilename.split(".shp")[0] + "_" + "RD" + ".shp"
        if os.path.exists(shp_filename):
            shpdriver.DeleteDataSource(shp_filename)
        shp_ds = shpdriver.CreateDataSource(shp_filename)
        if shp_ds is None:
            sys.exit(3)
        else:
            reprojected_layer = shp_ds.CreateLayer("layer", geom_type=ogr.wkbPoint)
        # Add fields
        layerDefn = layer.GetLayerDefn()
        for i in range(0, layerDefn.GetFieldCount()):
            fieldDefn = layerDefn.GetFieldDefn(i)
            reprojected_layer.CreateField(fieldDefn)

        reprojected_layerDefn = reprojected_layer.GetLayerDefn()
        layer.ResetReading()

        for feature in layer:
            geom = feature.GetGeometryRef()  # Get the input geometry
            geom.Transform(CoordTransform)  # Reproject the geometry
            newFeature = ogr.Feature(reprojected_layerDefn)
            newFeature.SetGeometry(geom)  # Set geometry and attribute
            for i in range(0, reprojected_layerDefn.GetFieldCount()):
                newFeature.SetField(reprojected_layerDefn.GetFieldDefn(i).GetNameRef(), feature.GetField(i))
            reprojected_layer.CreateFeature(newFeature)  # Add the feature to the shapefile
            newFeature.Destroy()  # Destroy the features and get the next input feature
        #vectords.Destroy()  # Destroy the shapefile
        rastersrs.MorphToESRI()  # Addan ESRI .prj file
        prjFilename = shp_filename.split(".shp")[0] + ".prj"
        prjFile = open(prjFilename, 'w')
        prjFile.write(rastersrs.ExportToWkt())
        prjFile.close()

    reprojectdata_dict = {"shpfilename": shp_filename}
    return reprojectdata_dict

def extract_raster_shp_L8(readshp_dict, raster_dict_L8, fieldname):
    band = raster_dict_L8["rasterds"].GetRasterBand(1)
    bandArray = band.ReadAsArray(0, 0, raster_dict_L8['xsize'], raster_dict_L8['ysize'])
    maxR = len(bandArray)
    maxC = len(bandArray[0])
    bg_ratio = []
    '''Returns a list with the values of depth for the shape file'''
    layerDefn = readshp_dict['layer'].GetLayerDefn()
    nfields = layerDefn.GetFieldCount()
    for i in range(nfields):
        fieldDefn = layerDefn.GetFieldDefn(i)
        fieldName = fieldDefn.GetName()
        fieldWidth = fieldDefn.GetWidth()
        fieldPrecision = fieldDefn.GetPrecision()
        fieldTypeCode = fieldDefn.GetType()
        fieldType = fieldDefn.GetFieldTypeName(fieldTypeCode)
        if fieldName == fieldname:  # Ask for this fieldname under plugins instruction
            break
    depth_value = []
    #  Get features
    for k in range(readshp_dict['featurecount']):  # If having a problem could be about the fieldDefn
        feature = readshp_dict['layer'].GetFeature(k)
        #  Get x, y coordinates
        geom = feature.GetGeometryRef()
        x = geom.GetX()
        y = geom.GetY()

        ulx = raster_dict_L8["geotransform"][0]  # x coordinate for upper left corner
        uly = raster_dict_L8["geotransform"][3]  # y coordinate for upper lefr corner
        xres = raster_dict_L8["geotransform"][1]  # pixel size in the x dimension
        yres = raster_dict_L8["geotransform"][5]  # pixel size in the y dimension

        #  Convert x, y coordinates to row, column
        col = int((x - ulx) / xres)
        row = int((uly - y) / yres) * (-1)
        if col < maxC and row <= maxR:
            #  extract value
            bandValue = float(bandArray[row, col])  # Can we work with floats?
            if not np.isnan(bandValue):
                depth = feature.GetField(i)
                depth_value.append(depth)
                bg_ratio.append(bandValue)
    readshp_dict['shpds'] = None
    raster_dict_L8['rasterds'] = None
    extract_dict_landsat = {"depths": depth_value, "bgratio": bg_ratio}
    return extract_dict_landsat

def extract_raster_shp_S2(readshp_dict, toa_dict_S2, fieldname):
    print("Running extract raster shp_s2...")
    band = toa_dict_S2['rasterds'].GetRasterBand(1)
    print("Successfully accessed toa_dict_S2[rasterds])")
    bandArray = band.ReadAsArray(0, 0, toa_dict_S2['xsize'], toa_dict_S2['ysize'])
    maxR = len(bandArray)
    maxC = len(bandArray[0])
    bg_ratio = []
    #Returns a list with the values of depth for the shape file
    layerDefn = readshp_dict['layer'].GetLayerDefn()
    nfields = layerDefn.GetFieldCount()
    for i in range(nfields):
        fieldDefn = layerDefn.GetFieldDefn(i)
        fieldName = fieldDefn.GetName()
        fieldWidth = fieldDefn.GetWidth()
        fieldPrecision = fieldDefn.GetPrecision()
        fieldTypeCode = fieldDefn.GetType()
        fieldType = fieldDefn.GetFieldTypeName(fieldTypeCode)
        if fieldName == fieldname:  # Ask for this fieldname under plugins instruction
            break
    depth_value = []
    #  Get features
    for k in range(readshp_dict['featurecount']):  # If having a problem could be about the fieldDefn
        feature = readshp_dict['layer'].GetFeature(k)
        #  Get x, y coordinates
        geom = feature.GetGeometryRef()
        x = geom.GetX()
        y = geom.GetY()

        ulx = toa_dict_S2["geotransform"][0]  # x coordinate for upper left corner
        uly = toa_dict_S2["geotransform"][3]  # y coordinate for upper lefr corner
        xres = toa_dict_S2["geotransform"][1]  # pixel size in the x dimension
        yres = toa_dict_S2["geotransform"][5]  # pixel size in the y dimension

        #  Convert x, y coordinates to row, column
        col = int((x - ulx) / xres)
        row = int((uly - y) / yres) * (-1)
        if col < maxC and row <= maxR:
            #  extract value
            bandValue = float(bandArray[row, col])  # Can we work with floats?
            if not np.isnan(bandValue):
                depth = feature.GetField(i)
                depth_value.append(depth)
                bg_ratio.append(bandValue)
    readshp_dict['shpds'] = None
    toa_dict_S2['rasterds'] = None
    extract_dict_S2 = {"depths": depth_value, "bgratio": bg_ratio}
    return extract_dict_S2

def extract_raster_shp_WV2(readshp_dict, toa_dict_WV2, fieldname):
    print("Running extract raster shp_WV2...")
    band = toa_dict_WV2['rasterds'].GetRasterBand(1)
    print("Successfully accessed toa_dict_WV2[rasterds])")
    bandArray = band.ReadAsArray(0, 0, toa_dict_WV2['xsize'], toa_dict_WV2['ysize'])
    maxR = len(bandArray)
    maxC = len(bandArray[0])
    bg_ratio = []
    #Returns a list with the values of depth for the shape file
    layerDefn = readshp_dict['layer'].GetLayerDefn()
    nfields = layerDefn.GetFieldCount()
    for i in range(nfields):
        fieldDefn = layerDefn.GetFieldDefn(i)
        fieldName = fieldDefn.GetName()
        fieldWidth = fieldDefn.GetWidth()
        fieldPrecision = fieldDefn.GetPrecision()
        fieldTypeCode = fieldDefn.GetType()
        fieldType = fieldDefn.GetFieldTypeName(fieldTypeCode)
        if fieldName == fieldname:  # Ask for this fieldname under plugins instruction
            break
    depth_value = []
    #  Get features
    for k in range(readshp_dict['featurecount']):  # If having a problem could be about the fieldDefn
        feature = readshp_dict['layer'].GetFeature(k)
        #  Get x, y coordinates
        geom = feature.GetGeometryRef()
        x = geom.GetX()
        y = geom.GetY()

        ulx = toa_dict_WV2["geotransform"][0]  # x coordinate for upper left corner
        uly = toa_dict_WV2["geotransform"][3]  # y coordinate for upper lefr corner
        xres = toa_dict_WV2["geotransform"][1]  # pixel size in the x dimension
        yres = toa_dict_WV2["geotransform"][5]  # pixel size in the y dimension

        #  Convert x, y coordinates to row, column
        col = int((x - ulx) / xres)
        row = int((uly - y) / yres) * (-1)
        if col < maxC and row <= maxR:
            #  extract value
            bandValue = float(bandArray[row, col])  # Can we work with floats?
            if not np.isnan(bandValue):
                depth = feature.GetField(i)
                depth_value.append(depth)
                bg_ratio.append(bandValue)
    readshp_dict['shpds'] = None
    toa_dict_WV2['rasterds'] = None
    extract_dict_WV2 = {"depths": depth_value, "bgratio": bg_ratio}
    return extract_dict_WV2

#This function also does not get used as far as I can tell.
def extract_values(readshp_dict, fieldname):
    '''Returns a list with the values of depth for the shape file'''
    layerDefn = readshp_dict['layer'].GetLayerDefn()
    nfields = layerDefn.GetFieldCount()
    for i in range(nfields):
        fieldDefn = layerDefn.GetFieldDefn(i)
        fieldName = fieldDefn.GetName()
        fieldWidth = fieldDefn.GetWidth()
        fieldPrecision = fieldDefn.GetPrecision()
        fieldTypeCode = fieldDefn.GetType()
        fieldType = fieldDefn.GetFieldTypeName(fieldTypeCode)
        if fieldName == fieldname:  # Ask for this fieldname under plugins instruction
            field = fieldTypeCode
            break
    values = []
    #  Get features
    for k in range(readshp_dict['featurecount']):  # If having a problem could be about the fieldDefn
        feature = readshp_dict['layer'].GetFeature(k)
        feature.GetField(i)
        values.append(feature.GetField(i))
    return values

#This function does not get used either...as far as I can tell.
def raster_shp_landsat(readshp_dict, raster_dict_L8):
    '''Uses the raster and shape files to extract the values of the shape file coordinates
    from the green and blue TOA bands'''
    #  Use reprojected shpfile
    #  Read band and convert it to numpy Array
    band = raster_dict_L8["rasterds"].GetRasterBand(1)
    bandArray = band.ReadAsArray(0, 0, raster_dict_L8['xsize'], raster_dict_L8['ysize'])
    maxR = len(bandArray)
    maxC = len(bandArray[0])
    #  Get x, y coordinates
    values_list = []
    for feature in readshp_dict["layer"]:
        geom = feature.GetGeometryRef()
        x = geom.GetX()
        y = geom.GetY()

        ulx = raster_dict_L8["geotransform"][0]  # x coordinate for upper left corner
        uly = raster_dict_L8["geotransform"][3]  # y coordinate for upper lefr corner
        xres = raster_dict_L8["geotransform"][1]  # pixel size in the x dimension
        yres = raster_dict_L8["geotransform"][5]  # pixel size in the y dimension

        #  Convert x, y coordinates to row, column
        col = int((x - ulx) / xres)
        row = int((uly - y) / yres) * (-1)
        if col < maxC and row <= maxR:
            #  extract value
            bandValue = float(bandArray[row, col])  # Can we work with floats?
            values_list.append(bandValue)
    readshp_dict['shpds'] = None
    raster_dict_L8['rasterds'] = None
    return values_list

def regression_array(extract_rshp_dict):
    '''Creates the regression array with the depth and blue/green logarithm lists.'''
    regressionarray = np.zeros((len(extract_rshp_dict['depths']), 2))
    regressionarray[:, 0] = extract_rshp_dict['depths']
    regressionarray[:, 1] = extract_rshp_dict['bgratio']
    return regressionarray

def plot(scatter_ar, plots, plot_title, outfolder):
    plt.scatter(scatter_ar[:, 1], scatter_ar[:, 0], c='navy', marker='.', label='Data')
    plt.legend(loc='best')
    plt.xlim(0, 5)
    plt.xlabel('Blue Green ratio (Ln)')
    plt.ylabel('Depth (m)')
    plt.title(plot_title)
    filename = outfolder + "/" + plot_title + ".png"
    plt.savefig(filename)
    plt.close()
    plt_dict = {"filename": filename}
    return plt_dict

def ratio_L8(mtl_filename, shp_filename, out_folder, depthColName):

    # Read metadata for blue band
    blueband = read_mtl_landsat(mtl_filename, 2)
    read_blueband = read_raster_landsat(blueband)
    toa_blueband = TOA_refl_landsat(blueband, read_blueband, out_folder)
    #  Close blueband
    read_blueband['rasterds'] = None
    #  Green band
    greenband = read_mtl_landsat(mtl_filename, 3)
    read_greenband = read_raster_landsat(greenband)
    toa_greenband = TOA_refl_landsat(greenband, read_greenband, out_folder)
    #  Close greenband
    read_greenband['rasterds'] = None
    #  Deep water
    swir7band = read_mtl_landsat(mtl_filename, 7)
    #  Get the kernel mean and corners
    kernel = deep_kernel_landsat(swir7band, 20)
    #  Shapefile stuff
    # Open
    shpdict = read_shp(shp_filename)
    #  Reproject shapefile
    rprj = reproject_data(shp_filename, shpdict['shpsrs'], read_blueband['rastersrs'],
                                  shpdict['shpdriver'],
                                  shpdict['layer'])
    #  Close the old shapefile
    shpdict['shpds'] = None

    #  Extract the depth values from the shapefile
    #  Make the blue green logarithm array
    blue_green_log = log_bluegreen(toa_blueband, toa_greenband, rprj, depthColName, out_folder)

    #  Get the blue green kernels std and mean
    blue_green_kernel = bluegreen_kernel_landsat(blue_green_log, kernel)

    #  Extract the depths and shape values from the blue and green toa bands
    #  Open the blue green logarithmic division
    blue_green = read_raster_landsat(blue_green_log)

    #  Open the reprojected shapefile
    rprjshp = read_shp(rprj['shpfilename'])
    #  extract the values into a list
    blue_green_values = extract_raster_shp_L8(rprjshp, blue_green, depthColName)  # Blue/ Green List
    #  Create the regression array
    regr_ar = regression_array(blue_green_values)
    plotlist = []
    # plot the data
    data_plot = plot(regr_ar, plotlist, 'Data', out_folder)
    #  Dictionary
    ratioPart1Dict = {"read_blue": read_blueband, "blue_toa": toa_blueband, "read_green": read_greenband,
                      "green_toa": toa_greenband, "log_blue_green": blue_green_log, "rprj_shapefile": rprj['shpfilename'],
                      "bluegreen_ratio": blue_green_values,
                      "regression_array": regr_ar, "plot_data": data_plot}
    blue_green = None
    return ratioPart1Dict

def ratio_S2(mtl_filename, shp_filename, out_folder, depthColName):
    print("ratio_S2 has been called.")

    metadata = read_mtl_S2(mtl_filename)
    # Read metadata for blue band
    blue_bandname = metadata[1][1]
    blueband = metadata[0][1]
    toa_blueband = TOA_refl_S2(blueband, blue_bandname, out_folder)

    #  Green band
    print ("Blue band read, reading green band.")
    green_bandname = metadata[1][2]
    greenband = metadata[0][2]
    toa_greenband = TOA_refl_S2(greenband, green_bandname, out_folder)

    #  Deep water
    print ("Green band read, kerneling.")
    swir11band = metadata[0][10]
    #  Get the kernel mean and corners
    kernel = deep_kernel_S2(swir11band, 20)

    #  Shapefile stuff
    # Open

    shpdict = read_shp(shp_filename)
    #  Reproject shapefile
    print ("Reprojecting shapefile")
    rprj = reproject_data(shp_filename, shpdict['shpsrs'], toa_blueband['rastersrs'],
                                  shpdict['shpdriver'],
                                  shpdict['layer'])
    #  Close the old shapefile
    shpdict['shpds'] = None
    print ("User shapefile has been reprojected.")


    #  Make the blue green logarithm array
    print("Running log_bluegreen()")
    blue_green_log = log_bluegreen(toa_blueband, toa_greenband, rprj, depthColName, out_folder)

    print("Successfully run log_bluegreen() ")
    #  Get the blue green kernels std and mean
    blue_green_kernel = bluegreen_kernel_S2(blue_green_log, kernel)

    #  Open the reprojected shapefile
    rprjshp = read_shp(rprj['shpfilename'])
    #  Extract the depth values from the shapefile
    #  Extract the depths and shape values from the blue and green toa bands
    #  Open the blue green logarithmic division
    print("read_raster_S2 has been called.")
    blue_green = read_raster_S2(blue_green_log)

    print("read_raster_S2 has finished running and blue_green variable has data assigned to it")

    #  extract the values into a list

    print("about to call extract_raster_shp_s2")

    blue_green_values = extract_raster_shp_S2(rprjshp, blue_green, depthColName)  # Blue/ Green List
    print("Finished running extract_raster_shp_S2 ")
    #  Create the regression array
    print("Running regression_array")
    regr_ar = regression_array(blue_green_values)
    print("Finished running regression_array")
    print regr_ar
    plotlist = []
    # plot the data
    data_plot = plot(regr_ar, plotlist, 'Data', out_folder)

    #  Dictionary
    ratioPart1Dict = {"read_blue": blueband, "blue_toa": toa_blueband, "read_green": greenband,
                      "green_toa": toa_greenband, "log_blue_green": blue_green_log, "rprj_shapefile": rprj['shpfilename'],
                      "bluegreen_ratio": blue_green_values,
                      "regression_array": regr_ar, "plot_data": data_plot}
    blue_green = None
    return ratioPart1Dict

def ratio_WV2(mtl_filename, shp_filename, out_folder, depthColName):
    print("ratio_WV2 has been called.")
    #Get metadata
    metadata = read_mtl_WV2(mtl_filename)

    sza = metadata[3]
    jday = metadata[9]

    #Gains and offsets
    gains = [1.151, 0.988, 0.936, 0.949, 0.952, 0.974, 0.961, 1.002]
    offsets = [-7.478, -5.736, -3.546, -3.564, -2.512, -4.120, -3.300, -2.891]

    # ESUN
    ESUNs = [1773.81, 2007.27, 1829.62, 1701.85, 1538.85, 1346.09, 1053.21, 856.599]

    # Fixed central wavelengths
    nbands = 5
    central_wavelengths = [428.4, 479.2, 547.6, 608.0, 659.2, 723.8, 827.7, 923.3]

    # Get the calibration for green and blue bands
    blue_cal = metadata[1][1]
    green_cal = metadata[1][2]
    nir1_cal = metadata[1][6]

    # Read WV2 image
    imagefile = metadata[0]
    dataset = gdal.Open(imagefile[0])
    xsize, ysize, fileNbands, projection, GT = getDatasetProperties(dataset)
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(dataset.GetProjection())
    band = dataset.GetRasterBand(1)
    DN = band.ReadAsArray(0, 0, xsize, ysize).astype(float)
    DN[DN == 0] = np.nan
    #Calculate blue TOA reflectance and save as new image.
    print ("Calculating blue TOA reflectance.")
    blue_image = ((gains[1] * DN * blue_cal + offsets[1]) * pow(getEarthSunDistance(jday),
                                                                                       2) * np.pi) / (
                                          ESUNs[1] * math.cos(deg2rad(sza)))

    print ("Saving as new raster.")
    blue_toa = "Blue_TOA.tif"
    blue_toa_filename = out_folder + "/" + blue_toa
    blue_raster_driver = gdal.GetDriverByName("GTiff")
    blue_toa_dataset = blue_raster_driver.Create(blue_toa_filename, xsize, ysize, 1, gdal.GDT_Float32)
    blue_toa_dataset.SetGeoTransform(GT)
    blue_toa_dataset.SetProjection(projection)
    blueband_toa = blue_toa_dataset.GetRasterBand(1)
    blueband_toa.WriteArray(blue_image)
    blue_toa_dataset = None
    toa_blueband = {"toabandname": blue_toa, 'filename': blue_toa_filename, "nbands": fileNbands, "xsize": xsize,
                   "ysize": ysize, "rastersrs": raster_srs, "rasterds": dataset, "projection": projection,
                   "geotransform": GT}

    print ("Save completed, calculating green TOA reflectance.")
    #Calculate green TOA reflectance and save as new raster image.
    green_image = ((gains[2] * DN * green_cal + offsets[2]) * pow(getEarthSunDistance(jday),
                                                                                       2) * np.pi) / (
                                          ESUNs[2] * math.cos(deg2rad(sza)))

    print ("Saving as new raster.")
    green_toa = "Green_TOA.tif"
    green_toa_filename = out_folder + "/" + green_toa
    green_raster_driver = gdal.GetDriverByName("GTiff")
    green_toa_dataset = green_raster_driver.Create(green_toa_filename, xsize, ysize, 1, gdal.GDT_Float32)
    green_toa_dataset.SetGeoTransform(GT)
    green_toa_dataset.SetProjection(projection)
    greenband_toa = green_toa_dataset.GetRasterBand(1)
    greenband_toa.WriteArray(green_image)
    green_toa_dataset = None
    toa_greenband = {"toabandname": green_toa, 'filename': green_toa_filename, "nbands": fileNbands, "xsize": xsize,
                   "ysize": ysize, "rastersrs": raster_srs, "rasterds": dataset, "projection": projection,
                   "geotransform": GT}
    print ("Save completed. Calculating deep water file.")

    #  Deep water
    nir1_image = ((gains[6] * DN * nir1_cal + offsets[6]) * pow(getEarthSunDistance(jday),
                                                                                       2) * np.pi) / (
                                          ESUNs[6] * math.cos(deg2rad(sza)))

    print ("Saving NIR1 raster.")

    nir1_toa = "nir1_TOA.tif"
    nir1_toa_filename = out_folder + "/" + nir1_toa
    nir1_raster_driver = gdal.GetDriverByName("GTiff")
    nir1_toa_dataset = nir1_raster_driver.Create(nir1_toa_filename, xsize, ysize, 1, gdal.GDT_Float32)
    nir1_toa_dataset.SetGeoTransform(GT)
    nir1_toa_dataset.SetProjection(projection)
    nir1band_toa = nir1_toa_dataset.GetRasterBand(1)
    nir1band_toa.WriteArray(nir1_image)
    nir1_toa_dataset = None
    toa_nir1band = {"bandname": nir1_toa, 'filename': nir1_toa_filename, "nbands": fileNbands, "xsize": xsize,
                   "ysize": ysize, "rastersrs": raster_srs, "rasterds": dataset, "projection": projection,
                   "geotransform": GT}
    print ("Save completed.")

    #  Shapefile stuff
    # Open

    shpdict = read_shp(shp_filename)
    #  Reproject shapefile
    print ("Reprojecting shapefile")
    rprj = reproject_data(shp_filename, shpdict['shpsrs'], raster_srs,
                                  shpdict['shpdriver'],
                                  shpdict['layer'])
    #  Close the old shapefile
    shpdict['shpds'] = None
    print ("User shapefile has been reprojected.")


    #  Make the blue green logarithm array
    print("Running log_bluegreen()")
    blue_green_log = log_bluegreen_WV2(toa_blueband, toa_greenband, rprj, depthColName, out_folder)

    print("Successfully run log_bluegreen() ")
    '''kernel = deep_kernel_WV2(toa_nir1band, 20)
    #  Get the blue green kernels std and mean
    #blue_green_kernel = bluegreen_kernel_WV2(blue_green_log, kernel)'''

    #  Open the reprojected shapefile
    rprjshp = read_shp(rprj['shpfilename'])
    #  Extract the depth values from the shapefile
    #  Extract the depths and shape values from the blue and green toa bands
    #  Open the blue green logarithmic division
    print("read_raster_WV2 has been called.")
    blue_green = read_raster_WV2(blue_green_log)

    print("read_raster_WV2 has finished running and blue_green variable has data assigned to it")

    #  extract the values into a list

    print("about to call extract_raster_shp_WV2")

    blue_green_values = extract_raster_shp_WV2(rprjshp, blue_green, depthColName)  # Blue/ Green List
    print("Finished running extract_raster_shp_WV2 ")
    #  Create the regression array
    print("Running regression_array")
    regr_ar = regression_array(blue_green_values)
    print("Finished running regression_array")
    plotlist = []
    # plot the data
    data_plot = plot(regr_ar, plotlist, 'Data', out_folder)

    #  Dictionary
    ratioPart1Dict = {"imagefile": imagefile, "blue_toa": toa_blueband,
                      "green_toa": toa_greenband, "log_blue_green": blue_green_log, "rprj_shapefile": rprj['shpfilename'],
                      "bluegreen_ratio": blue_green_values,
                      "regression_array": regr_ar, "plot_data": data_plot}
    blue_green = None
    return ratioPart1Dict

def show_image(string):
    image = PIL.Image.open(string)
    image.show()

def extract_array_landsat(reg_ar_L8, startvalue, endvalue):
    narray = []
    startvalue = float(startvalue)
    endvalue = float(endvalue)
    for i in range(len(reg_ar_L8)):
        #dv = reg_ar[:, np.newaxis, 0][i][0]
        dv = float(reg_ar_L8[i, 0])  # Should be simpler
        #if startvalue <= dv < endvalue:
        if (dv < endvalue) and (startvalue <= dv):
            narray.append(reg_ar_L8[i])
    return np.asarray(narray)

def extract_array_S2(reg_ar_S2, startvalue, endvalue):
    narray = []
    startvalue = float(startvalue)
    endvalue = float(endvalue)
    for i in range(len(reg_ar_S2)):
        #dv = reg_ar[:, np.newaxis, 0][i][0]
        dv = float(reg_ar_S2[i, 0])  # Should be simpler
        #if startvalue <= dv < endvalue:
        if (dv < endvalue) and (startvalue <= dv):
            narray.append(reg_ar_S2[i])
    return np.asarray(narray)

def extract_array_WV2(reg_ar_WV2, startvalue, endvalue):
    narray = []
    startvalue = float(startvalue)
    endvalue = float(endvalue)
    for i in range(len(reg_ar_WV2)):
        #dv = reg_ar[:, np.newaxis, 0][i][0]
        dv = float(reg_ar_WV2[i, 0])  # Should be simpler
        #if startvalue <= dv < endvalue:
        if (dv < endvalue) and (startvalue <= dv):
            narray.append(reg_ar_WV2[i])
    return np.asarray(narray)

##### Statistics #####
'''Part 3 performs statistical analysis with data derived above a certain threshold and returns 2 new plots.'''
#Depth vs ln ratio and acoustic depth vs derived depth plots

def train_test(regressionarray, trainsize):
    '''Creates the split for the test and train datasets'''
    #  Define X
    X = regressionarray[:, np.newaxis, 1]
    #  Define the targets
    Y = regressionarray[:, np.newaxis, 0]
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y, train_size=trainsize, random_state=42)

    traintest_dict = {"xtrain": x_train, "xtest": x_test, "ytrain": y_train, "ytest": y_test, "x": X, "y": Y}
    return traintest_dict

def debug_data(traintest_dict, regression_list, trainsize, output_folder):
    reg_ar = np.asarray(regression_list)
    min = int(np.min(reg_ar[:, np.newaxis, 0])) - 1
    max = int(np.max(reg_ar[:, np.newaxis, 0])) + 1
    range_list = range(min, max)
    Slope_list = []
    Intercept = []
    MSE_list = []
    Predict_list = []
    for i in range_list:
        start = min
        end = i + 1
        split_array = extract_array_landsat(reg_ar_L8, start, end)
        train_test_sets = train_test(split_array, trainsize)
        ols = OLS_regression(train_test_sets)
        cmap = get_cmap(len(range_list))
        predict_for_plot = plt.plot(train_test_sets['xtrain'], ols['predict'], color=cmap(i+1), linewidth=1)
        slope = ols['coefficient'][0, 0]
        intercept = ols['intercept'][0]
        mse = ols['MSE']
        Slope_list.append(slope)
        Intercept.append(intercept)
        MSE_list.append(mse)
        Predict_list.append(predict_for_plot)
    #  Plot
    plt.scatter(traintest_dict['x'], traintest_dict['y'], c='navy', marker='.', label='Data')
    #for j in Predict_list:
    #    j
    plt.legend(loc='best')
    plt.title("Split data regressions")
    plt.savefig(output_folder + "\split_data" + ".png")         #randomNumber +
    #  Write lists to excel csv

    debug_data_dict = {"slope": Slope_list, "intercept": Intercept, "MSE": MSE_list, "predict": Predict_list,
                       "rl_median": np.median(range_list)}
    return debug_data_dict

def OLS_regression(traintest_dict):
    '''applies the OLS regression to the train set.'''
    #  Create the linear regression object
    ols_regression = linear_model.LinearRegression()

    #  Build Linear Regression using Train_test datasets
    ols_regression.fit(traintest_dict['xtrain'], traintest_dict['ytrain'])

    # Calculate the Mean Square Error
    MSE = np.mean((ols_regression.predict(traintest_dict['xtest']) - traintest_dict['ytest']) ** 2)
    #  Calculate variance score
    var_score = ols_regression.score(traintest_dict['xtest'], traintest_dict['ytest'])
    #  Attributes: coef_, residues_, intercept_
    olsregression_dict = {"OLS": ols_regression, "coefficient": ols_regression.coef_,
                          "predict": ols_regression.predict(traintest_dict['xtrain']),
                          "intercept": ols_regression.intercept_, "residues": ols_regression.residues_, "MSE": MSE,
                          "varscore": var_score, "rtype": "OLS"}

    return olsregression_dict

def theil_sen_regression(traintest_dict):
    '''Applies the Theil Sen regression to the train set.'''
    #  Create the linear regression object
    theilsen_regression = linear_model.TheilSenRegressor()
    #  Build Linear Regression using Train_test datasets
    theilsen_regression.fit(traintest_dict['xtrain'], traintest_dict['ytrain'])
    # Calculate the Mean Square Error
    MSE = np.mean((theilsen_regression.predict(traintest_dict['xtest']) - traintest_dict['ytest']) ** 2)
    var_score = theilsen_regression.score(traintest_dict['xtest'], traintest_dict['ytest'])
    #  Attributes: coef_, intercept_, breakdown_, n_iter_, n_subpopulation_
    theilsen_dict = {"TheilSen": theilsen_regression, "coefficient": theilsen_regression.coef_,
                     "intercept": theilsen_regression.intercept_,
                     "predict": theilsen_regression.predict(traintest_dict['xtrain']), "MSE": MSE, "varscore": var_score,
                     "rtype": "Theil_Sen"}
    return theilsen_dict

def ransac_regression(traintest_dict):
    '''Applies the RANSAC regression to the Train set.'''
    #  Robustly fit linear model with RANSAC algorithm
    RANSAC_regression = linear_model.RANSACRegressor(linear_model.LinearRegression())
    RANSAC_regression.fit(traintest_dict['xtrain'], traintest_dict['ytrain'])
    inlier_mask = RANSAC_regression.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    coefficient = RANSAC_regression.estimator_.coef_
    intercept = RANSAC_regression.estimator_.intercept_
    MSE = np.mean((RANSAC_regression.predict(traintest_dict['xtest']) - traintest_dict['ytest']) ** 2)
    var_score = RANSAC_regression.score(traintest_dict['xtest'], traintest_dict['ytest'])
    #  Attributes: estimator_, n_trials, inlier_mask_
    RANSAC_dict = {"RANSAC": RANSAC_regression, "coefficient": coefficient, "inliers": inlier_mask,
                   "outliers": outlier_mask, "intercept": intercept, "MSE": MSE, "varscore": var_score,
                   "predict": RANSAC_regression.predict(traintest_dict['xtrain']), "rtype": "RANSAC"}
    return RANSAC_dict

def plot_ols(regression_dict, traintest_dict, outputfolder):
    '''Plots the regression line and the dataset'''
    plt.scatter(traintest_dict['xtrain'], traintest_dict['ytrain'], c='navy', marker='.', label='Data')
    plt.plot(traintest_dict['xtrain'], regression_dict['predict'], color='yellowgreen', label='OLS Regressor',
             linewidth=2)
    plt.legend(loc='best')
    plt.title("Ordinary Least Squares Regression")
    plt.xlabel('Blue Green ratio (Ln)')
    plt.ylabel('Depth (m)')
    plt.savefig(outputfolder + "\OLS_plot.png")
    plt.close()
    plotname = outputfolder + "\OLS_plot.png"
    ols_dict = {"plotname": plotname}
    return ols_dict

def plot_theilsen(regression_dict, traintest_dict, outputfolder):
    '''Plots the regression line and the dataset'''
    plt.scatter(traintest_dict['xtrain'], traintest_dict['ytrain'], c='navy', marker='.', label='Data')
    plt.plot(traintest_dict['xtrain'], regression_dict['predict'], color='turquoise', label='Theil - Sen Regressor',
             linewidth=2)
    plt.legend(loc='best')
    plt.xlabel('Blue Green ratio (Ln)')
    plt.ylabel('Depth (m)')
    plt.title("Theil Sen Regression")
    plt.savefig(outputfolder + "\Theil_Sen_plot.png")
    plt.close()
    plotname = outputfolder + "\Theil_Sen_plot.png"
    theil_dict = {"plotname": plotname}
    return theil_dict

def plot_RANSAC(regression_dict, traintest_dict, outputfolder):
    '''Plots the regression line and the dataset'''
    plt.scatter(traintest_dict['xtrain'][regression_dict['inliers']],
                traintest_dict['ytrain'][regression_dict['inliers']], c='yellowgreen', marker='o', label='Inliers')
    plt.scatter(traintest_dict['xtrain'][regression_dict['outliers']],
                traintest_dict['ytrain'][regression_dict['outliers']], c='gold', marker='x', label='Outliers')
    plt.plot(traintest_dict['xtrain'], regression_dict['predict'], c='cornflowerblue', linestyle='-', linewidth=2, label='RANSAC Regressor')
    plt.legend(loc='best')
    plt.xlabel('Blue Green ratio (Ln)')
    plt.ylabel('Depth (m)')
    plt.title("RANSAC Regression")
    plt.savefig(outputfolder + "\RANSAC_plot.png")
    plt.close()
    plotname = outputfolder + "\RANSAC_plot.png"
    ransac_dict = {"plotname": plotname}
    return ransac_dict

def depth_array(coefficient, intercept, bluegreen_log_dict, outputfolder, regression_type):
    '''Applies the depth algorithm to the blue green log array and creates a new raster file.'''
    depth = (coefficient * (bluegreen_log_dict['bluegreen_log'])) + intercept
    depth[depth < 0.0] = np.nan
    # Write array
    depth_raster_filename = outputfolder + "/" + "Depth" + "_" + regression_type + ".tif"
    depth_bandname = "Depth" + "_" + regression_type + ".tif"
    raster_driver = gdal.GetDriverByName("GTiff")
    rows, cols = depth.shape
    depth_raster_dataset = raster_driver.Create(depth_raster_filename, cols, rows, 1, gdal.GDT_Float32)
    depth_raster_dataset.SetGeoTransform(bluegreen_log_dict['geotransform'])
    depth_raster_dataset.SetProjection(bluegreen_log_dict['projection'])
    band = depth_raster_dataset.GetRasterBand(1)
    band.WriteArray(depth)
    depth_dict = {"depthfilename": depth_raster_filename, "depthbandname": depth_bandname,
                  "depthraster_ds": depth_raster_dataset, "deptharray": depth}
    return depth_dict

def check_forLand(depth_dict):
    depth_image = gdal.Open(depth_dict['depthfilename'])
    depth_image_array = np.array(depth_image.GetRasterBand(1).ReadAsArray())
    #  Calculate image shape
    image_array_shape = depth_image_array.shape
    image_x_size = image_array_shape[1]
    image_y_size = image_array_shape[0]

def StringtoRaster(raster):
    fileInfo = QFileInfo(raster)
    path = fileInfo.filePath()
    baseName = fileInfo.baseName()
    layer = QgsRasterLayer(raster, baseName)
    if layer.isValid() == True:
        QgsMapLayerRegistry.instance().addMapLayer(layer)
        print 'Layer was loaded successfully'
    else:
        print 'Unable to read base name and file path.'

def extract_all_depths(rprj_shp_filename, depth_dict, fieldname, out_folder):
    '''Extracts acoustic depths from locations of derived depths in array'''
    band = depth_dict['depthraster_ds']
    xsize, ysize, fileNbands, projection, GT = getDatasetProperties(band)
    bandArray = band.ReadAsArray(0, 0, xsize, ysize)
    #bandArray = depth_dict["deptharray"]
    maxR = len(bandArray)
    maxC = len(bandArray[0])
    der_depths = []
    '''Returns a list with the values of depth for the shape file'''
    readshp_dict = read_shp(rprj_shp_filename)
    layerDefn = readshp_dict['layer'].GetLayerDefn()
    nfields = layerDefn.GetFieldCount()
    for i in range(nfields):
        fieldDefn = layerDefn.GetFieldDefn(i)
        fieldName = fieldDefn.GetName()
        fieldWidth = fieldDefn.GetWidth()
        fieldPrecision = fieldDefn.GetPrecision()
        fieldTypeCode = fieldDefn.GetType()
        fieldType = fieldDefn.GetFieldTypeName(fieldTypeCode)
        if fieldName == fieldname:  # Ask for this fieldname under plugins instruction
            break
    ac_depths = []
    #  Get features
    for k in range(readshp_dict['featurecount']):  # If having a problem could be about the fieldDefn
        feature = readshp_dict['layer'].GetFeature(k)
        #  Get x, y coordinates
        geom = feature.GetGeometryRef()
        x = geom.GetX()
        y = geom.GetY()

        ulx = GT[0]  # x coordinate for upper left corner
        uly = GT[3]  # y coordinate for upper lefr corner
        xres = GT[1]  # pixel size in the x dimension
        yres = GT[5]  # pixel size in the y dimension

        #  Convert x, y coordinates to row, column
        col = int((x - ulx) / xres)
        row = int((uly - y) / yres) * (-1)
        if col < maxC and row <= maxR:
            #  extract value
            derDepth = float(bandArray[row, col])  # Can we work with floats?
            if not np.isnan(derDepth):
                depth = feature.GetField(i)
                ac_depths.append(depth)
                der_depths.append(derDepth)

    readshp_dict['shpds'] = None
    depth_dict['depthbandname'] = None
    extract_all_depths_dict = {"acousticdepths": ac_depths, "deriveddepths": der_depths}
    #Returns dictionary to be used for plot
    return extract_all_depths_dict

def depthPlot(scatter_ar, plot_title, threshold, outfolder):
    #Takes the acoustic and derived depths array and plots them
    plt.scatter(scatter_ar[:, 1], scatter_ar[:, 0], c='navy', marker='.', label='Data')
    #plt.plot(range(threshold))
    ymax = int(threshold)
    plt.ylim(0, (ymax + 2))
    #plt.xlim(0, threshold)
    plt.legend(loc='best')
    plt.xlabel('Derived Depths (m)')
    plt.ylabel('Acoustic Depths (m)')
    plt.title(plot_title)
    plt.savefig(outfolder + "/" + plot_title + ".png")
    plt.close()
    filename = outfolder + "/" + plot_title + ".png"
    plt_dict = {"filename": filename}
    return plt_dict

def all_depths_plot(depth_regr_ar, depthColName, reprj, threshold, out_folder):
    #Function calls other functions needed to make the depth plot
    all_depths = extract_all_depths(reprj, depth_regr_ar, depthColName, out_folder)
    all_depth_ar = create_all_depths_array(all_depths)
    depth_plot = depthPlot(all_depth_ar, 'Acoustic vs Derived Depths', threshold, out_folder)
    filename = depth_plot['filename']
    all_depths_dict = {"depths_data": depth_plot, "filename": filename}
    print ("Dictionary written. You nailed it, Holman!")
    #Returns the data needed to plot in QGIS
    return all_depths_dict

def create_all_depths_array(all_depths_dict):
    #Creates an array of the acoustic and derived depths
    ad_depth_ar = np.zeros((len(all_depths_dict['deriveddepths']),2))
    ad_depth_ar[:, 0] = all_depths_dict['acousticdepths']
    ad_depth_ar[:, 1] = all_depths_dict['deriveddepths']
    print ("Save complete.")
    return ad_depth_ar






