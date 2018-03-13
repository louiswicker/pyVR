import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as P
P.ioff()
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import interp as Binterp 

import warnings
warnings.filterwarnings("ignore")

import time as timeit
import netCDF4 as nc
import numpy as np
import xarray as xr
import sys
import pyart
import xml.etree.ElementTree as et
import pyresample
import pandas as pd
from dart_tools import *


# Create uneven height bins because of MRMS and other radar scans.
height_bins = [(0.0, 1000.), (1000., 2000.), (2000., 3000.), (3000., 4000.),
               (4000., 6000.), (6000., 8000.), (8000.,10000.)]

# Colortables
_vr_ctable  = pyart.graph.cm.Carbone42

_file = "./20160525-010021"

radars = []
lats   = []
lons   = []
struct = {}

#=========================================================================================
# Class variable used as container

class Generic_Container(object):

  def __init__(self, name, data=None, **kwargs):
    self.name = name
    self.data = data

    if kwargs != None:
      for key in kwargs:  setattr(self, key, kwargs[key])

  def keys(self):
    return self.__dict__

#===============================================================================

def nearlyequal(a, b, sig_digit=None):
    """ Measures the equality (for two floats), in unit of decimal significant 
        figures.  If no sigificant digit is specified, default is 7 digits. """

    if sig_digit == None or sig_digit > 7:
        sig_digit = 7
    if a == b:
        return True
    difference = abs(a - b)
    avg = (a + b)/2
    
    return np.log10(avg / difference) >= sig_digit
    
#===============================================================================
    
def nice_mxmnintvl( dmin, dmax, outside=True, max_steps=15, cint=None, sym=False):
    """ Description: Given min and max values of a data domain and the maximum
                     number of steps desired, determines "nice" values of 
                     for endpoints and spacing to create a series of steps 
                     through the data domainp. A flag controls whether the max 
                     and min are inside or outside the data range.
  
        In Args: float   dmin 		    the minimum value of the domain
                 float   dmax       the maximum value of the domain
                 int     max_steps	the maximum number of steps desired
                 logical outside    controls whether return min/max fall just
                                    outside or just inside the data domainp.
                     if outside: 
                         min_out <= min < min_out + step_size
                                         max_out >= max > max_out - step_size
                     if inside:
                         min_out >= min > min_out - step_size
                                         max_out <= max < max_out + step_size
      
                 float    cint      if specified, the contour interval is set 
                                    to this, and the max/min bounds, based on 
                                    "outside" are returned.

                 logical  sym       if True, set the max/min bounds to be anti-symmetric.
      
      
        Out Args: min_out     a "nice" minimum value
                  max_out     a "nice" maximum value  
                  step_size   a step value such that 
                                     (where n is an integer < max_steps):
                                      min_out + n * step_size == max_out 
                                      with no remainder 
      
        If max==min, or a contour interval cannot be computed, returns "None"
     
        Algorithm mimics the NCAR NCL lib "nice_mxmnintvl"; code adapted from 
        "nicevals.c" however, added the optional "cint" arg to facilitate user 
        specified specific interval.
     
        Lou Wicker, August 2009 """

    table = np.array([1.0,2.0,2.5,4.0,5.0,10.0,20.0,25.0,40.0,50.0,100.0,200.0,
                      250.0,400.0,500.0])

    if nearlyequal(dmax,dmin):
        return None
    
    # Help people like me who can never remember - flip max/min if inputted reversed
    if dmax < dmin:
        amax = dmin
        amin = dmax
    else:
        amax = dmax
        amin = dmin

    if sym:
        smax = max(amax.max(), amin.min())
        amax = smax
        amin = -smax

    d = 10.0**(np.floor(np.log10(amax - amin)) - 2.0)
    if cint == None or cint == 0.0:
        t = table * d
    else:
        t = cint
    if outside:
        am1 = np.floor(amin/t) * t
        ax1 = np.ceil(amax/t)  * t
        cints = (ax1 - am1) / t 
    else:
        am1 = np.ceil(amin/t) * t
        ax1 = np.floor(amax/t)  * t
        cints = (ax1 - am1) / t
    
    # DEBUG LINE BELOW
    # print t, am1, ax1, cints
    
    if cint == None or cint == 0.0:   
        try:
            index = np.where(cints < max_steps)[0][0]
            return am1[index], ax1[index], cints[index]
        except IndexError:
            return None
    else:
        return am1, ax1, cint

#===============================================================================
def nice_clevels( *args, **kargs):
    """ Extra function to generate the array of contour levels for plotting 
        using "nice_mxmnintvl" code.  Removes an extra step.  Returns 4 args,
        with the 4th the array of contour levels.  The first three values
        are the same as "nice_mxmnintvl". """
    
    try:
        amin, amax, cint = nice_mxmnintvl(*args, **kargs)
        return amin, amax, cint, np.arange(amin, amax+cint, cint) 
    except:
        return None
#===============================================================================
def mymap(xwidth, ywidth, cntr_lat, cntr_lon, scale = 1.0, ax = None, ticks = True, resolution='c',\
          area_thresh = 10., shape_env = False, states=True, counties=True, pickle = False):

    tt = timeit.clock()

    map = Basemap(width=xwidth, height=ywidth, \
                  lat_0=cntr_lat,lon_0=cntr_lon, \
                  projection = 'lcc',      \
                  resolution=resolution,   \
                  area_thresh=area_thresh, \
                  suppress_ticks=ticks, \
                  ax=ax)

    if counties:
        map.drawcounties()

    if states:
        map.drawstates()
        
# Shape file stuff

    if shape_env:

        try:
            shapelist = os.getenv("PYESVIEWER_SHAPEFILES").split(":")

            if len(shapelist) > 0:

                for item in shapelist:
                    items = item.split(",")
                    shapefile  = items[0]
                    color      = items[1]
                    linewidth  = items[2]

                    s = map.readshapefile(shapefile,'counties',drawbounds=False)

                    for shape in map.counties:
                        xx, yy = zip(*shape)
                        map.plot(xx,yy,color=color,linewidth=linewidth)

        except OSError:
            print "GIS_PLOT:  NO SHAPEFILE ENV VARIABLE FOUND "

# pickle the class instance.

    print(timeit.clock()-tt,' secs to create original Basemap instance')

    if pickle:
        pickle.dump(map,open('mymap.pickle','wb'),-1)
        print(timeit.clock()-tt,' secs to create original Basemap instance and pickle it')

    return map

#===============================================================================
def plot_contour(fld, xwidth, ywidth, title = None, cntr_lat=None, cntr_lon=None, 
                 clevels = None, ctables = None, ax = None, mask = None, map = None):
               
    if map == None:
       map = mymap(xwidth, ywidth, cntr_lat, cntr_lon, ax = ax)
    
    if clevels == None:
        _, _, _, clevels = nice_clevels( fld.min(), fld.max() )
        
    if ctables == None:
        ctables = P.cm.viridis
        
    if title == None:
        title = "No Title"
        
    if mask == None:
        fld2d = fld
    else:
        fld2d = np.ma.masked_where(mask, fld)
     
# get coordinates for contour plots

    lon2d, lat2d, xx, yy = map.makegrid(fld.shape[0], fld.shape[1], returnxy=True)

    plot    = map.contourf(xx, yy, fld2d, clevels, cmap= ctables)
    cbar    = map.colorbar(plot, location='right',pad="5%")
    plot    = map.contour(xx, yy,  fld2d, clevels[::2], colors='k', linewidths=0.5)
    
    P.title(title, fontsize=10)
# 
#     at = AnchoredText("Max: %4.1f \n Min: %4.1f" % (fld.max(),fld.min()), 
#                       loc=4, prop=dict(size=6), frameon=True,)            
#     at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
#                       
#     ax.add_artist(at)

    return map
    
#-------------------------------------------------------------------------------
#
def read_MRMS_VR_XML(filename, ret_index=False):

    if filename[-3:] != 'xml':  filename = "%s.xml" % filename

#load xml file
    tree = et.parse(filename)

#loop through to find rowname then iterate through rownames extracting radars

    for i in tree.iter('datacolumn'):
        if i.attrib.get('name') == 'RowName':
            for radar in i.iter('item'):
                radars.append(radar.get('value'))
        if i.attrib.get('name') == 'Latitude':
            for lat in i.iter('item'):
                lats.append(lat.get('value'))
        if i.attrib.get('name') == 'Longitude':
            for lon in i.iter('item'):
                lons.append(lon.get('value'))

    if ret_index == False:
        struct = {}
        for n, radar in enumerate(radars):
            struct[radar] = (n,lats[n],lons[n])
        return struct
    else:
        struct = []
        for n, radar in enumerate(radars):
            struct.append((radar,lats[n],lons[n]))
        return struct

#-------------------------------------------------------------------------------
#
def read_MRMS_VR_NCDF(filename, retFileAttr = False):

    if filename[-6:] != 'netcdf':  filename = "%s.netcdf" % filename

    if retFileAttr == False:
        return xr.open_dataset(filename).to_dataframe()
    else:
        xa = xr.open_dataset(filename)
        return xa.to_dataframe(), xa.attrs


#-------------------------------------------------------------------------------
#
def vr_parse(df, filename, index, radar_table):

    df.rename(columns={'v': 'vr', 'lats': 'lat', 'lons': 'lon', 'heights': 'hgt'}, inplace=True)

    rindex = df['i'].values
    rhgt = np.zeros(rindex.shape[0])
    rlat = np.zeros(rindex.shape[0])
    rlon = np.zeros(rindex.shape[0])

    for n in np.arange(rindex.shape[0]):
        ii = rindex[n]
        radar = index[ii][0]
        rlat[n], rlon[n], rhgt[n] = radar_table[radar][:]
       
    return Generic_Container(filename, data  = df['vr'].values, 
                                       hgts  = df['hgt'].values,
                                       lats  = df['lat'].values,
                                       lons  = df['lon'].values,
                                       rlat  = rlat,
                                       rlon  = rlon,
                                       rhgt  = rhgt,
                                       xR    = df['xOverR'].values,
                                       yR    = df['yOverR'].values,
                                       zR    = df['zOverR'].values,
                                       field = "VR", 
                                       time = 0,
                                       missingData = -9.9e9 )

#-------------------------------------------------------------------------------
#
def vr_query(df, filename, height=None, radar=None):


    vr  = []
    lat = []
    lon = []
    hgt = []
    rad = []

    # This string is used to bin data in height
    query_string = '%f < heights <= %f' % (height[0], height[1])

    if radar:
        radar_string = "i == %d" % radar
        query_string = "%s & %s" % (query_string, radar_string)
        
#     print query_string

    # Create coordinate list for heights
    
    new_df = df.query(query_string)
    
    new_df.rename(columns={'v': 'vr', 'lats': 'lat', 'lons': 'lon', 'heights': 'hgt'}, inplace=True)

        
    return Generic_Container(filename, data  = new_df['vr'].values, 
                                       rhgt  = new_df['hgt'].values,
                                       hgts  = new_df['hgt'].values,
                                       lats  = new_df['lat'].values,
                                       lons  = new_df['lon'].values,
                                       xR    = new_df['xOverR'].values,
                                       yR    = new_df['yOverR'].values,
                                       zR    = new_df['zOverR'].values,
                                       rind  = new_df['i'].values,
                                       field = "DOPPLER_VELOCITY", 
                                       radar_hgt  = 500., 
                                       time = 0,
                                       missingData = -9.9e9 )

#-------------------------------------------------------------------------------
#
# Main
#

#fig, axes = P.subplots(1, 2, sharey=True, figsize=(20,8))

vr_table = read_MRMS_VR_XML(_file)
vr_index = read_MRMS_VR_XML(_file, ret_index=True)
vr_obs   = read_MRMS_VR_NCDF(_file)

obs = vr_parse(vr_obs, _file, vr_index, read_radar_location_file())

obs.nyquist   = 32.

write_DART_ascii_list(obs, obs_error=3.0)

print "done"


#bmap = mymap(300000., 300000., vr_table['KDDC'][1], vr_table['KDDC'][2], ax=axes[0])
#
#xpts,ypts = bmap(data['lon'].values,data['lat'].values)
#radar_x, radar_y = bmap(vr_table['KDDC'][2], vr_table['KDDC'][1])

#bs = bmap.scatter(xpts,ypts,c=data['vr'].values, vmin=-30., vmax=30., cmap=_vr_ctable, s=10)
#cbar = bmap.colorbar(bs, location='right',pad="5%")
#bmap.scatter(radar_x, radar_y, s=50, c='k')
#P.title("Radar: %s   Min Hgt: %3.1f km   Max Hgt:  %3.1f km" % (radar, 0.001*data['hgt'].min(), 0.001*data['hgt'].max()))

#P.savefig('scattterPlot.png',dpi=300)


#xg, yg = 5000.*np.arange(60), 5000.*np.arange(60)
#lons,lats = bmap(xg, yg, inverse=True)
#glons, glats = np.meshgrid(lons,lats)

#dd = data['vr'].values
#dx = data['lon'].values
#dy = data['lat'].values
#
#grid_vr = np.array((60,60))

#grid  = pyresample.geometry.GridDefinition(lats=glats, lons=glons)
#swath = pyresample.geometry.SwathDefinition(lons=dx, lats=dy)

#wf = lambda r: 1/r**2

#grid_vr = pyresample.kd_tree.resample_custom(swath, dd, grid, radius_of_influence=10000, neighbours=10, weight_funcs = wf, fill_value=None)

#_, _, _, clevels = nice_clevels( -30., 30., cint=5.0)

#ret = plot_contour(grid_vr, 300000., 300000., cntr_lat=vr_table['KDDC'][1], cntr_lon=vr_table['KDDC'][2], 
#                  ax=axes[1], clevels = clevels, ctables=_vr_ctable, title ="VR Map")
#ret.scatter(radar_x, radar_y, s=50, c='k')

#P.savefig('contourPlot.png',dpi=300)



# fig, axes = P.subplots(2, 2, sharey=True, figsize=(15,15))
# 
# axes = axes.flatten()
# 
# for m in np.arange(4):
#     n = 1 + m
#     axes[m].scatter(data['lon'][n],data['lat'][n],c=data['vr'][n], vmin=-30., vmax=30., cmap=_vr_ctable, s=10)
#     axes[m].set_title("Avg Height:  %3.1f km / Std:  %3.1f km" % (0.001*data['hgt'][n].mean(), 0.001*data['hgt'][n].std()))
#     axes[m].scatter(vr_table['KDDC'][2], vr_table['KDDC'][1], s=50, c='k')
