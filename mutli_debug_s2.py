from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model, cross_validation
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from osgeo import gdal, ogr, osr
#from qgis.core import QgsRasterLayer, QgsMapLayerRegistry
#from PyQt4.QtCore import QFileInfo
import os, sys, csv
import numpy as np
#import PIL.Image
import matplotlib.pyplot as plt
import math
import xml.etree.ElementTree as ET
import scipy.stats
import scipy.ndimage

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

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

    if ("B01.jp2" in bandname):
        TOA_refl_S2 = scipy.ndimage.zoom(TOA_refl_S2, 6, order=0)  # Resample to match other bands

        GT = (GT[0], 10.0, GT[2], GT[3], GT[4], -10.0)
        #GT[1] = 10.0
        #GT[5] = -10.0
        xsize = xsize * 6
        ysize = ysize * 6
        ### Maybe need to update rasterds as well, if needed later in code ###


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

def multipleLinearRegression(coastaldict, bluedict, greendict, reddict, readshp_dict,
                                fieldname, outputfolder):
    '''Creates the multiple linear regression method using the coastal, blue, green, and red bands as inputs.'''
    #  Open the blue band
    coastal_image = gdal.Open(coastaldict['filename'])
    coastal_band = coastal_image.GetRasterBand(1)
    gt = coastal_image.GetGeoTransform()

    #  Open the blue band
    blue_image = gdal.Open(bluedict['filename'])
    blue_band = blue_image.GetRasterBand(1)

    #  Open the green band
    green_image = gdal.Open(greendict['filename'])
    green_band = green_image.GetRasterBand(1)

    #  Open the green band
    red_image = gdal.Open(reddict['filename'])
    red_band = red_image.GetRasterBand(1)

    coastal_values = []
    blue_values = []
    green_values = []
    depth_values = []
    red_values = []

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
            coastalval = coastal_band.ReadAsArray(px, py, 1, 1)[0][0]
            blueval = blue_band.ReadAsArray(px, py, 1, 1)[0][0]
            greenval = green_band.ReadAsArray(px, py, 1, 1)[0][0]
            redval = red_band.ReadAsArray(px, py, 1, 1)[0][0]

            depth_values.append(depth)
            coastal_values.append(coastalval)
            blue_values.append(blueval)
            green_values.append(greenval)
            red_values.append(redval)


    depth_values = np.asarray(depth_values)
    coastal_values = np.asarray(coastal_values)
    blue_values = np.asarray(blue_values)
    green_values = np.asarray(green_values)
    red_values = np.asarray(red_values)

    # Find deep-water values for all bands
    deep_water = np.percentile(depth_values, 95)
    deep_water_index = np.where(depth_values > deep_water, 1, 0)

    deep_coastal = np.sum(deep_water_index*coastal_values) / sum(deep_water_index)
    deep_blue = np.sum(deep_water_index * blue_values) / sum(deep_water_index)
    deep_green = np.sum(deep_water_index * green_values) / sum(deep_water_index)
    deep_red = np.sum(deep_water_index * red_values) / sum(deep_water_index)

    coastal_log = np.log(coastal_values - deep_coastal)
    blue_log = np.log(blue_values - deep_blue)
    green_log = np.log(green_values - deep_green)
    red_log = np.log(red_values - deep_red)

    good_values = np.where(~np.isnan(coastal_log + blue_log + green_log + red_log), 1, 0) # Those that are not nan in any array
    good_coastal_log = coastal_log[good_values == 1]
    good_blue_log = blue_log[good_values == 1]
    good_green_log = green_log[good_values == 1]
    good_red_log = red_log[good_values == 1]
    good_depth_values = depth_values[good_values == 1]

    X = np.transpose(np.array([good_coastal_log, good_blue_log, good_green_log, good_red_log]))
    multiband = LinearRegression()
    multiband.fit(X, good_depth_values)

    # Make predictions
    coastalval = np.log(coastal_band.ReadAsArray() - deep_coastal)
    blueval = np.log(blue_band.ReadAsArray() - deep_blue)
    greenval = np.log(green_band.ReadAsArray() - deep_green)
    redval = np.log(red_band.ReadAsArray() - deep_red)


    for_prediction = np.transpose(np.array([coastalval.flatten(), blueval.flatten(), greenval.flatten(), redval.flatten()]))

    goodval = np.sum(for_prediction, axis=1)
    good_coastalval = np.where(~np.isnan(goodval), coastalval.flatten(), 0)
    good_blueval = np.where(~np.isnan(goodval), blueval.flatten(), 0)
    good_greenval = np.where(~np.isnan(goodval), greenval.flatten(), 0)
    good_redval = np.where(~np.isnan(goodval), redval.flatten(), 0)

    for_prediction = np.transpose(
        np.array([good_coastalval, good_blueval, good_greenval, good_redval]))

    predicted = multiband.predict(for_prediction)
    predicted_2d = predicted.reshape(coastalval.shape[0], coastalval.shape[1])

    # Remove NAN predictions (that are not currently NAN
    value_to_remove = multiband.predict(np.array([[0, 0, 0, 0]]))
    predicted_2d_clean = np.where(predicted_2d == value_to_remove, 0, predicted_2d)

    #close all the open files from finding best n so can re-open them to perform the actual function.
    coastal_image = None
    coastal_band = None
    coastalval = None
    blue_image = None
    blue_band = None
    blueval = None
    green_image = None
    green_band = None
    greenval = None
    red_image = None
    red_band = None
    redval = None
    swir_image = None
    swir_band = None
    swirval = None

    #  Create the raster file
    multiband_filename = outputfolder + "/" + "multiband.tif"
    multiband_bandname = "multiband.tif"
    raster_driver = gdal.GetDriverByName("GTiff")
    multiband_dataset = raster_driver.Create(multiband_filename, bluedict['xsize'], bluedict['ysize'],
                                              1, gdal.GDT_Float32)
    multiband_dataset.SetGeoTransform(bluedict['geotransform'])
    multiband_dataset.SetProjection(bluedict['projection'])
    band = multiband_dataset.GetRasterBand(1)
    band.WriteArray(predicted_2d_clean)
    multiband_dataset = None
    multiband_dict = {"bandname": multiband_bandname, 'filename': multiband_filename, "multiband_log":
                        predicted_2d_clean, "geotransform": bluedict['geotransform'], "projection":
                        bluedict['projection']}

    return multiband_dict

def mutliband_kernel_landsat(multiband_logarray, kernelLandsatDict):
    '''Uses the deep kernel found and evaluates it in the division array so we obtain a standard deviation
    for the kernel, and a mean value'''
    imagefilename = multiband_logarray['filename']
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
        multiband_mean = np.mean(kernel)
        multiband_max = np.max(kernel)
        multiband_min = np.min(kernel)

    bgk = {"std": stdev, "mean": multiband_mean, "mid": kernel[(kernel_size/2), (kernel_size/2)], "max": multiband_max,
           "min": multiband_min}
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

def extract_raster_shp_S2(readshp_dict, toa_dict_S2, fieldname):
    print("Running extract raster shp_s2...")
    band = toa_dict_S2['rasterds'].GetRasterBand(1)
    print("Successfully accessed toa_dict_S2[rasterds])")
    bandArray = band.ReadAsArray(0, 0, toa_dict_S2['xsize'], toa_dict_S2['ysize'])
    maxR = len(bandArray)
    maxC = len(bandArray[0])
    multi = []
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
                multi.append(bandValue)
    readshp_dict['shpds'] = None
    toa_dict_S2['rasterds'] = None
    extract_dict_S2 = {"depths": depth_value, "multi": multi}
    return extract_dict_S2

def regression_array(extract_rshp_dict):
    '''Creates the regression array with the depth and blue/green logarithm lists.'''
    regressionarray = np.zeros((len(extract_rshp_dict['depths']), 2))
    regressionarray[:, 0] = extract_rshp_dict['depths']
    regressionarray[:, 1] = extract_rshp_dict['multi']
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

mtl_filename = ("D:/Kiyomi/Nunavut/Imagery/Sentinel2/CambridgeBay/"
                "S2A_MSIL1C_20160809T190922_N0204_R056_T13WDS_20160809T190917.SAFE/MTD_MSIL1C.xml")
shp_filename = ("D:/Kiyomi/Nunavut/CHSData/Depths/CB_lidar_S2.shp")
out_folder = ("D:/Kiyomi/Nunavut/SDBOutputs/Mutliband/SentinelTest")
depthColName = ("field_3")

# Get metadata
print ("multiband_S2 has been called.")

# Read metadata
metadata = read_mtl_S2(mtl_filename)

# read metadata for coastal band
coastal_bandname = metadata[1][0]
coastalband = metadata[0][0]
toa_coastalband = TOA_refl_S2(coastalband, coastal_bandname, out_folder)


# Read metadata for blue band
blue_bandname = metadata[1][1]
blueband = metadata[0][1]
toa_blueband = TOA_refl_S2(blueband, blue_bandname, out_folder)

#  Read metadata for green band
green_bandname = metadata[1][2]
greenband = metadata[0][2]
toa_greenband = TOA_refl_S2(greenband, green_bandname, out_folder)

# Read metadata for red band
red_bandname = metadata[1][3]
redband = metadata[0][3]
toa_redband = TOA_refl_S2(redband, red_bandname, out_folder)
print ("All bands open and corrected for TOA reflectances.")
#  Shapefile stuff
# Open
print ("Opening shapefile.")
shpdict = read_shp(shp_filename)
#  Reproject shapefile
print ("Reprojecting shapefile")
rprj = reproject_data(shp_filename, shpdict['shpsrs'], toa_blueband['rastersrs'],
                      shpdict['shpdriver'],
                      shpdict['layer'])
#  Close the old shapefile
shpdict['shpds'] = None
print ("User shapefile has been reprojected.")

#  Extract the depth values from the shapefile
print ("Now calling multiband_MLR function.")
multiband_MLR = multipleLinearRegression(toa_coastalband, toa_blueband, toa_greenband, toa_redband,
                                            rprj, depthColName, out_folder)
print ("Multiple linear regression completed. Now reading image to extract data.")
#  Get the blue green kernels std and mean
#multiband_kernel = multiband_kernel_landsat(multiband_log, kernel)

#  Extract the depths and shape values from the blue and green toa bands
#  Open the blue green logarithmic division
multiband = read_raster_S2(multiband_MLR)

#  Open the reprojected shapefile
rprjshp = read_shp(rprj['shpfilename'])
#  extract the values into a list
multiband_values = extract_raster_shp_S2(rprjshp, multiband, depthColName)  # Blue/ Green List
#  Create the regression array
regr_ar = regression_array(multiband_values)
plotlist = []
# plot the data
data_plot = plot(regr_ar, plotlist, 'Data', out_folder)
#  Dictionary
multibandDict = {"coastal_toa": toa_coastalband, "blue_toa": toa_blueband, "green_toa": toa_greenband,
                 "red_toa": toa_redband, "multiband_MLR": multiband_MLR,
                 "rprj_shapefile": rprj['shpfilename'], "multiband_values": multiband_values,
                 "regression_array": regr_ar, "plot_data": data_plot}
multiband = None
#return multibandDict
