import ee
import time
import sys
import numpy as np
import pandas as pd
import itertools
import os
import urllib

ee.Initialize()

def export_oneimage(img,folder,name,scale,crs):
  task = ee.batch.Export.image.toDrive(img, name, folder, name, None, None, scale, crs)
  """
  task = ee.batch.Export.image(img, name, {
      'driveFolder':folder,
      'driveFileNamePrefix':name,
      'scale':scale,
      'crs':crs
  })
  """
  task.start()
  while task.status()['state'] == 'RUNNING':
    print ('Running...')
    # Perhaps task.cancel() at some point.
    time.sleep(10)
  print ('Done.', task.status())




locations = pd.read_csv('locations_remedy.csv')


# Transforms an Image Collection with 1 band per Image into a single Image with items as bands
# Author: Jamie Vleeshouwer

"""
var stackCollection = function(collection) {
  // Create an initial image.
  var first = ee.Image(collection.first()).select([]);

  // Write a function that appends a band to an image.
  var appendBands = function(cur, previous) {
    return ee.Image(previous).addBands(cur);
  };
  return ee.Image(collection.iterate(appendBands, first));
};
var stacked = stackCollection(imgcoll);
bandNames = stacked.bandNames();
print('Band names: ', bandNames);"""

def super_image(collection):
    first = ee.Image(collection.first()).select([])
    def appendBand(current, previous):
        # Rename the band
        previous=ee.Image(previous)
        current = current.select([0])
        return ee.Image(previous).addBands(current)
    return ee.Image(collection.iterate(appendBand, first))

county_region = ee.FeatureCollection('ft:1W1vJtONWc6kNJHj4hMo2wJ6iZk2cxTMCENA8Qlmh')

#imgcoll = ee.ImageCollection('MODIS/MOD09A1') \
#imgcoll = ee.ImageCollection('MODIS/006/MOD13Q1') \
imgcoll = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')\
    .filterBounds(ee.Geometry.Rectangle(38, 42, 87, 22))\
    .filterDate('2014-04-30','2016-12-31')

image = ee.Image(imgcoll.first())
print('FIRST INFO:\n',image.getInfo())

bandNames = image.bandNames();
print('Band names: ', bandNames);

count = imgcoll.size();
print('\nCount: ', count);

img=super_image(imgcoll)

bandNames = img.bandNames();
print('\nBand names total: ', bandNames);

#img_0=ee.Image(ee.Number(-100))
#img_16000=ee.Image(ee.Number(16000))
img_0=ee.Image(ee.Number(-1.5))
img_16000=ee.Image(ee.Number(60))

img=img.min(img_16000)
img=img.max(img_0)

# img=ee.Image(ee.Number(100))
# img=ee.ImageCollection('LC8_L1T').mosaic()

for loc1, loc2, lat, lon in locations.values:
    fname = '{}_{}'.format(int(loc1), int(loc2))

    # offset = 0.11
    scale  = 500
    crs='EPSG:4326'

    # filter for a county
    region = county_region.filterMetadata('adm0_code', 'equals', int(loc1))
    region = ee.FeatureCollection(region).filterMetadata('adm1_code', 'equals', int(loc2))
    region = ee.Feature(region.first())
    # region = region.geometry().coordinates().getInfo()[0]

    # region = str([
    #     [lat - offset, lon + offset],
    #     [lat + offset, lon + offset],
    #     [lat + offset, lon - offset],
    #     [lat - offset, lon - offset]])
    while True:
        try:
            export_oneimage(img.clip(region), 'Data_county', fname, scale, crs)
        except:
            print ('retry')
            time.sleep(10)
            continue
        break
