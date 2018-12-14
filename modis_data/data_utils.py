import os
import numpy as np
import math
import re
import gdal



class ModisHistogramExtractor():
    def __init__(self, dir_name, num_bands, num_bins = 32, flatten = False):
        num_bins = 32 # arbitrary 
        all_regions = []
        self.county_code = []
        for filename in os.listdir(dir_name):
            ext = os.path.splitext(filename)[-1].lower()
            if ext != '.tif':
                continue
            prefix = dir_name + '/'
            print ('LOADING ', filename)
            print (prefix + filename)
            #MODIS_img = np.transpose(np.array(gdal.Open(prefix + filename).ReadAsArray(), dtype='uint16'),axes=(1,2,0))
            MODIS_img = np.transpose(np.array(gdal.Open(prefix + filename).ReadAsArray(), dtype='int16'),axes=(1,2,0))
            print ('LOAD DONE')
            print ('Divide Image')
            timeseries = self.divide_image(MODIS_img, 0, num_bands, int(MODIS_img.shape[2]/num_bands))
            #nan_mask = np.where(np.mean(image, axis = 2) == 0, np.nan, 1)[:, :, np.newaxis]
            #timeseries = timeseries*nan_mask
            print ('Processed ' + str(len(timeseries)) + ' Images')
            new_timeseries = []
         
            for image in timeseries:
                
                hist = self.calc_histograms(image, num_bands, num_bins)
                if flatten:
                    hist = (hist.squeeze().ravel())

                new_timeseries.append(hist)#this should be all the image captures for a region
               
            all_regions.append(new_timeseries)
            print(filename)
            print (re.split('_|\.', filename))
            self.county_code.append(int(re.split('_|\.', filename)[1]))

        print ('county codes: ', self.county_code)
        print ('length of county code list: ' + str(len(self.county_code)))
        print ('length of all regions: ' + str(len(all_regions)))
        self.data = list(zip(self.county_code, all_regions))                


    def calc_histograms(self, image, num_bands, num_bins):
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
    
    
    def divide_image(self, img,first,step,num):
        image_list=[]
        for i in range(0,num-1):
            image_list.append(img[:, :, first:first+step])
            first+=step
        image_list.append(img[:, :, first:])
        return image_list
    
    
    def get(self, i):
        return self.data[i]

    def get_ids(self):
        return self.ids

    def get_data(self):
        return self.data

    def get_labels(self):
        return zip(*self.data)[1]





class ModisFeatureExtractor():
    # http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0093107
    def __init__(self, dir_name, num_bands, inc_indices = True, inc_histogram= False):
        # TODO - you shouldn't have to do this!!!
        # n time series is the length of time steps included I believe 
        #n_timeseries = 35
        # this is loading a dictionary or csv perhaps 
        #self.ids = content['ids']
        num_bins = 32 # arbitrary 
        all_regions = []
        self.county_code = []
        # for us this could be regions perhaps, loaded over a time series.
        # just looking at it though seems like we are pulling in literally all the data
        # each timeseries is a list of prebroken images 
        #tif_names = [o.key for o in storage.Bucket(dir_name).objects() if o.key.startswith('test')]
        #print tif_names
        for filename in os.listdir(dir_name):
            ext = os.path.splitext(filename)[-1].lower()
            if ext != '.tif':
                continue
            prefix = dir_name + '/'
            print ('LOADING ', filename)
            print (prefix + filename)
            MODIS_img = np.transpose(np.array(gdal.Open(prefix + filename).ReadAsArray(), dtype='int16'),axes=(1,2,0))
            print ('LOAD DONE')
            print ('Divide Image')
            timeseries = self.divide_image(MODIS_img, 0, num_bands, int(MODIS_img.shape[2]/num_bands))
            print ('Processed ' + str(len(timeseries)) + ' Images')
            new_timeseries = []
            #if len(timeseries) < n_timeseries:
                #continue
            # i think that image here must be just a single 7 band image
            for image in timeseries:
                nan_mask = np.where(np.mean(image, axis = 2) == 0, np.nan, 1)[:, :, np.newaxis]
                image = image*nan_mask
                R_R = np.nanmean(image[:,:,0])
                R_NIR = np.nanmean(image[:,:,1])
                R_B = np.nanmean(image[:,:,2])
                R_G = np.nanmean(image[:,:,3])
                GPP = np.nanmean(image[:,:,-1])
                if (inc_indices):
                    features = [
                      R_R,
                      R_NIR,
                      R_B,
                      R_G,
                      self.SR(R_R, R_NIR),
                      self.NVDI(R_R, R_NIR),
                      self.GNVDI(R_G, R_NIR),
                      self.TVI(R_R, R_G, R_NIR),
                      self.SAVI(R_R, R_NIR),
                      self.OSAVI(R_R, R_NIR),
                      self.NLI(R_R, R_NIR),
                      self.RVDI(R_R, R_NIR),
                      self.CARI(R_R, R_G, R_NIR),
                      self.PSRI(R_R, R_B, R_NIR)
                    ]
                else:
                    features = []
               
                n_features = len(features)
                # derivatives
                # literally dude whyyyy
                """
                if len(new_timeseries) == 0:
                    features += [0] * len(features)
                else:
                    features += list(np.array(features) - np.array(new_timeseries[-1][:n_features]))
                """
                new_timeseries.append(features)#this should be all the image captures for a region
                #print('appended')
            all_regions.append(new_timeseries)
            print(filename)
            print (re.split('_|\.', filename))
            self.county_code.append(int(re.split('_|\.', filename)[1]))
        #indices = np.array(indices)
        #labels = content['labels'][indices]
        #print len(labels), np.count_nonzero(np.array(labels))
        #self.data = [(x, y, None) for x, y in zip(out_vectors, labels)]   # need none to fit into (example, label,length) format
        print ('county codes: ', self.county_code)
        print ('length of county code list: ' + str(len(self.county_code)))
        print ('length of all regions: ' + str(len(all_regions)))
        self.data = list(zip(self.county_code, all_regions))   # need none to fit into (example, label, length) format

    def divide_image(self, img,first,step,num):
        image_list=[]
        for i in range(0,num-1):
            image_list.append(img[:, :, first:first+step])
            first+=step
        image_list.append(img[:, :, first:])
        return image_list

    def SR(self, R_R, R_NIR):
        """ simple ratio """
        return R_NIR * 1.0 / R_R
    def NVDI(self, R_R, R_NIR):
        """ normalized difference vegetation index """
        return (R_NIR - R_R) * 1.0 / (R_NIR + R_R)
    def GNVDI(self, R_G, R_NIR):
        """ green normalized difference vegetation index """
        return (R_NIR - R_G) * 1.0 / (R_NIR + R_G)
    def TVI(self, R_R, R_G, R_NIR):
        """ triangular vegetation index """
        return 0.5 * (120 * (R_NIR - R_G) - 200 * (R_R - R_G))
    def SAVI(self, R_R, R_NIR):
        """ soil adjusted vegetation index """
        return 1.5 * (R_NIR - R_R) / (R_NIR + R_R + 1.5)
    def OSAVI(self, R_R, R_NIR):
        """ optimized soil adjusted vegetation index """
        return (R_NIR / R_R) / (R_NIR + R_R + 16)
    def MSR(self, R_R, R_NIR):
        """ modified simple ratio """

    def NLI(self, R_R, R_NIR):
        """non-linear vegetation index """
        return (R_NIR**2 - R_R) / (R_NIR**2 + R_R)
    def RVDI(self, R_R, R_NIR):
        """ re-normalized difference vegetation index """
        return (R_NIR - R_R) / (R_NIR + R_R)**0.5
    def CARI(self, R_R, R_G, R_NIR):
        """ clorophyll absorption ratio """
        a = (R_NIR - R_R) * 1.0 / 150
        b = R_G - (a * 550)
        return ((a * 670 + R_R + b) * 1.0  / math.sqrt(a**2 + 1)) * (R_NIR * 1.0 / R_R)
    def PSRI(self, R_R, R_B, R_NIR):
        """ plant senescence reflectance index """
        return (R_R - R_B) / R_NIR
    def get(self, i):
        return self.data[i]
    def get_data(self):
        return self.data
    #def get_labels(self):
        #return zip(*self.data)[1]
    def get_counties(self):
        return self.county_codes
