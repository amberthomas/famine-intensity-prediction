#import ee
import pandas as pd
import itertools
from datetime import date
import numpy as np
import gdal



#ee.Initialize()
BINS = 45 + 1

imgcoll_dict = {
    'MODIS': ('MODIS/006/MOD09GA', '', 7),
    'NDVI' : ('MODIS/006/MOD13Q1', 'VEG_', 2)
}



def divide_image(img,first,step,num):
    image_list=[]
    for i in range(0,num-1):
        image_list.append(img[:, :, first:first+step])
        first+=step
    image_list.append(img[:, :, first:])
    return image_list


def dummy_years(img, start, num_days):
    y, x, bands = img.shape
    num_img = (bands / 7)
    print(num_days)
    result = []
    img_steps = int(num_days/num_img)
    for i in range(0, num_days, img_steps):
        result.append(pd.to_timedelta(i, unit="D") + start)
    return result


def get_county_data(country, admin_level, county_name, coll_code = 'MODIS', MODIS_path = None, data_path = None):

    num_bands = 7 #imgcoll_dict[coll_code][2] #7 for me 
    #coll_id = imgcoll_dict[coll_code][1] #don't know if thats correct for me

    country_admin = "%s_%s" % (country, admin_level) #set of input strings 
    path_to_bucket = '/mnt/mounted_bucket'

    # NOTE: this looks like it is a path to an individual tiff. Can make wrapper to get all
    # path to tiffs
    MODIS_img = np.transpose(np.array(gdal.Open(MODIS_path).ReadAsArray(), dtype='uint16'),axes=(1,2,0))

    print (MODIS_img.shape)
   
    ##Divide tif file into individual images
    MODIS_img_list = divide_image(MODIS_img, 0, num_bands, int(MODIS_img.shape[2]/num_bands))
   
    #TODO can't use until found way to save google earth objects to bucket 
    #Get the dates for the images collected
    #imgcoll = get_imgcoll(country_admin, path_to_bucket, coll_code)


    #TODO might be able to get around this by generating a list file during image import. 
    #dates = get_years(imgcoll)
    start = pd.to_datetime('2016-6-30', format='%Y-%m-%d', utc=True)
    dates = dummy_years(MODIS_img, start, 184)
    return MODIS_img_list 

def calc_histograms(image, num_bands, num_bins):

    if image.shape[2] % num_bands != 0:
        raise Exception('Number of bands does not match image depth.')
    num_times = image.shape[2]/num_bands
    hist = np.zeros([num_bins, int(num_times), num_bands]) # (32,1.0,7)
    for i in range(image.shape[2]):
        band = i % num_bands
        #density, _ = np.histogram(image[:, :, i], bin_seq_list[band], density=False)
        density, _ = np.histogram(image[:, :, i], np.linspace(1, 4999, num_bins + 1), density=False)
        total = density.sum() # normalize over only values in bins
        hist[:, int(i / num_bands), band] = density/float(total) if total > 0 else 0
    return hist


