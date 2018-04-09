import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as P
P.ioff()
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

import warnings
warnings.filterwarnings("ignore")

import time as timeit
import netCDF4 as nc
import numpy as np
import xarray as xr
import sys
import pyart
import xml.etree.ElementTree as et
import pandas as pd


# Range rings in km
_plot_RangeRings = True
_range_rings = [25, 50, 75, 100, 125]

# Colortables
_vr_ctable  = pyart.graph.cm.Carbone42

_file = "./20160525-010021"
_file = "/work/anthony.reinhart/VRtest/20170516/Point/KAMA/Velocity_Threshold_cut_smoothedKAMACollection/00.50/20170516-232455.netcdf"
_file = "/work/anthony.reinhart/VRtest/20170516/Point/KAMA/Velocity_Threshold_cut_smoothedKAMACollection/00.50/20170516-210506.netcdf"
_file = "/work/anthony.reinhart/VRtest/20170516/Point/KAMA/Velocity_Threshold_cut_smoothedKAMACollection/00.50/20170516-210118.netcdf"
_file = "/work/wicker/realtime/Point/KFDR/Velocity_Threshold_cut_smoothed_Collection/00.50/20170516-233037.netcdf"
_file = "/work/anthony.reinhart/VRtest/20170516/Point/KAMA/Velocity_Threshold_cut_smoothed_Collection/00.50/20170516-210118.netcdf"

radars = []
lats   = []
lons   = []
struct = {}

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
def plot_rectangles(fld, xp, yp, title = None, cntr_lat=None, cntr_lon=None, 
                    c_width=2500., ctables=None, norm=None, ax=None, fig=None):
               
    if ctables == None:
        ctables = P.cm.viridis
        
    if title == None:
        title = "No Title"

    if norm == None:
        norm = [fld.min(), fld.max()]
        
      
    Fnorm = mpl.colors.Normalize(vmin=norm[0], vmax=norm[1], clip=True)
    # Here's where you have to make a ScalarMappable with the colormap
    mappable = P.cm.ScalarMappable(norm=Fnorm, cmap=ctables)
    
    # normalize the data for the colormapping
    colors_norm = fld/(norm[1]-norm[0])
    
    # Give it your non-normalized color data
    mappable.set_array(fld)

    rects = []
    for p in zip(xp, yp):
        xpos = p[0] - c_width/2 # The x position will be half the width from the center
        ypos = p[1] - c_width/2 # same for the y position, but with height
        rect = Rectangle( (xpos, ypos), c_width, c_width ) # Create a rectangle
        rects.append(rect) # Add the rectangle patch to our list

    # Create a collection from the rectangles
    col = PatchCollection(rects)
    
    # set the alpha for all rectangles
    col.set_alpha(0.8)
    
    # Set the colors using the colormap
    col.set_facecolor( ctables(colors_norm) )
    
    # create figure if fig == None
    if fig == None:
        fig = plt.figure()
        
    if ax == None:
        ax = fig.add_subplot(111)

    # plot collection of rectangles
    ax.add_collection(col)    
    # add a colorbar
    P.colorbar(mappable)
    
    P.title(title, fontsize=10)

    return
    
#-------------------------------------------------------------------------------
#
def read_Merged_Radar_Table(filename):

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

    for n, radar in enumerate(radars):
        struct[radar] = (n,lats[n],lons[n])

    return struct
#-------------------------------------------------------------------------------
#
def read_MRMS_VR_NCDF(filename, retFileAttr = False):

    if filename[-6:] != 'netcdf':  filename = "%s.netcdf" % filename

    try:
        if retFileAttr == False:
            return xr.open_dataset(filename).to_dataframe()
        else:
            xa = xr.open_dataset(filename)
            return xa.to_dataframe(), xa.attrs
    except:
        print(" read_MRMS_VR_NCDF:  cannot read data file, return None\n")
        sys.exit(1)

#-------------------------------------------------------------------------------
#
def vr_filter(df, missing = -999.0):

    # This string is used to bin data in height
    query_string = 'vr > %f' % (missing+1.0)
    # Create coordinate list for heights

    new_df = df.query(query_string)
    
    new_df.rename(columns={'v': 'vr', 'height': 'hgt', 
                           'xOverR': 'xR', 'yOverR': 'yR', 'zOverR': 'zR'}, inplace=True)

    return new_df.sort_values(by=['lon', 'lat'])

#-------------------------------------------------------------------------------
#
# Main
#

fig = P.figure(figsize=(10,8))

ax = fig.add_subplot(111)

vr_obs, vr_attrs   = read_MRMS_VR_NCDF(_file, retFileAttr=True)

data = vr_filter(vr_obs, missing=vr_attrs['MissingData'])

xR = data['xR'].values
yR = data['yR'].values
zR = data['zR'].values

mag = xR**2 + yR**2 + zR**2

print "Checking vector magnitude: ",mag.max(), mag.min()

rlat = np.float(vr_attrs['Latitude'])
rlon = np.float(vr_attrs['Longitude'])

print "Radar lat/lon:  ", rlat, rlon

bmap = mymap(300000., 300000., rlat, rlon, ax = ax)

xpts, ypts = bmap(data['lon'].values,data['lat'].values)
vr_data = data['vr'].values
main_title = ("Radar: %s   Min Vr: %3.1f km   Max Vr:  %3.1f km" % ('KAMA', vr_data.min(), vr_data.max()))

plot_rectangles(vr_data, xpts, ypts, title=main_title, c_width=2500., \
                ctables = _vr_ctable, norm=[-25.,25], ax=ax, fig=fig)

# plot scatter points
# bs = bmap.scatter(xpts,ypts,c=data['vr'].values, vmin=-30., vmax=30., cmap=_vr_ctable, s=10)
# cbar = bmap.colorbar(bs, location='right',pad="5%")
# 
# # plot radar loc
radar_x, radar_y = bmap(rlon, rlat)
# print "Radar x, Radar y: " , radar_x, radar_y
bmap.scatter(radar_x, radar_y, s=50, c='k')
# P.title("Radar: %s   Min Hgt: %3.1f km   Max Hgt:  %3.1f km" % ('KAMA', 0.001*data['hgt'].min(), 0.001*data['hgt'].max()))

if _plot_RangeRings:
  angle = np.linspace(0., 2.0 * np.pi, 360)
  for ring in _range_rings:
     xpts = radar_x + ring * 1000. * np.sin(angle)
     ypts = radar_y + ring * 1000. * np.cos(angle)
     bmap.plot(xpts, ypts, color = 'gray', alpha = 0.5, linewidth = 1.0)


P.savefig("ScatterPlotTest.png")

