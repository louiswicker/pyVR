#!/usr/bin/env python
#############################################################
# grid3d: A program to process new MRMS volumes             #
#         netCDF4 MRMS files are read in and processed to   #
#         to DART format for assimilationp.                 #
#                                                           #
#       Python package requirements:                        #
#       ----------------------------                        #
#       Numpy                                               #
#       Scipy                                               #
#       matplotlib                                          #
#############################################################
#
# created by Lou Wicker Feb 2017
#
#############################################################
import os
import sys
import glob
import time as timeit

# Need to set the backend BEFORE loading pyplot
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
from mpl_toolkits.basemap import Basemap

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from optparse import OptionParser
import netCDF4 as ncdf

from pyart.graph import cm
import datetime as dtime
import xarray as xr
import pandas as pd

from netcdftime import utime

# missing value
_missing = -9999.
_plot_counties = True

# Colorscale information
_vel_scale   = [-30., 30.]
_plot_type   = 'png'
_vel_ctable  = cm.Carbone42

# Range rings in km
_range_rings = [25, 50, 75, 100, 125]

_vr_subdir_name = "Velocity_Threshold_cut_smoothed_Collection"

# time in seconds for radar window

_dt_window = [-300,120]

_vr_obs_error = 3.0
_obs_nyquist  = 32.
QC_default = 0

# Debug

_debug = True

time_format = "%Y-%m-%d_%H:%M:%S"
day_utime   = utime("days since 1601-01-01 00:00:00")
sec_utime   = utime("seconds since 1970-01-01 00:00:00")

#=========================================================================================
# Class variable used as container

class Gridded_Field(object):
  
  def __init__(self, name, data=None, **kwargs):    
    self.name = name    
    self.data = data    
    
    if kwargs != None:
      for key in kwargs:  setattr(self, key, kwargs[key])
      
  def keys(self):
    return self.__dict__

#=========================================================================================
# DART obs definitions (handy for writing out DART files)

def ObType_LookUp(name,DART_name=False,Print_Table=False):
   """ObType_LookUp returns the DART kind number for an input variable type.  There seems
      to be several ways observations names are written in the observation inputs
      and output files, e.g., in the DART ascii files and in the ***.obs.nc files,
      so this function is designed to handle the variety of cases and return the
      integer corresponding to the DART definitionp.
   
      Exampled:   REFLECTIVITY is sometimes stored as REFL
                  T_2M         is sometimes stored as TEMP2m
                  TD_2M        is sometimes stored as DEWPT2m
   
      If you come across a variable that is not defined, you can add it to the lookup
      table (dictionary) below - just make sure you know the official DART definition
      
      You can add any unique variable name or reference on the left column, and then use
      the pyDART INTERNAL definition on the right - so that you can refer to more than
      one type of data in different ways internally in this code
   
      Usage:  variable_kind = ObType_Lookup(variable_name)   where type(variable_name)=str
   
      If you need the return the actual DART_name as well, set the input flag to be True"""

# Create local dictionary for observation kind definition - these can include user abbreviations

#                      user's observation type            kind   DART official name

   Look_Up_Table={ "DOPPLER_VELOCITY":                 [11,   "DOPPLER_RADIAL_VELOCITY"] ,
                   "UNFOLDED VELOCITY":                [11,   "DOPPLER_RADIAL_VELOCITY"] ,
                   "VELOCITY":                         [11,   "DOPPLER_RADIAL_VELOCITY"] ,
                   "DOPPLER_RADIAL_VELOCITY":          [11,   "DOPPLER_RADIAL_VELOCITY"] ,
                   "REFLECTIVITY":                     [12,   "RADAR_REFLECTIVITY"],
                   "RADAR_REFLECTIVITY":               [12,   "RADAR_REFLECTIVITY"],
                   "RADAR_CLEARAIR_REFLECTIVITY":      [13,   "RADAR_CLEARAIR_REFLECTIVITY"],  
                   "CLEARAIR_REFLECTIVITY":            [13,   "RADAR_CLEARAIR_REFLECTIVITY"],
                   "DIFFERENTIAL_REFLECTIVITY":        [300,  "DIFFERENTIAL_REFLECTIVITY"],
                   "SPECIFIC_DIFFERENTIAL_PHASE":      [301,  "SPECIFIC_DIFFERENTIAL_PHASE"],
                   "METAR_U_10_METER_WIND":            [1,    "METAR_U_10_METER_WIND"],
                   "METAR_V_10_METER_WIND":            [2,    "METAR_V_10_METER_WIND"],
                   "METAR_TEMPERATURE_2_METER":        [4,    "METAR_TEMPERATURE_2_METER"],
                   "METAR_DEWPOINT_2_METER":           [9,    "METAR_DEWPOINT_2_METER"],
                   "METAR_SPECIFIC_HUMIDITY_2_METER":  [5,    "METAR_SPECIFIC_HUMIDITY_2_METER"],
                   "VR":                               [11,   "DOPPLER_RADIAL_VELOCITY"],
                   "DBZ":                              [12,   "RADAR_REFLECTIVITY"],
                   "0DBZ":                             [13,   "RADAR_CLEARAIR_REFLECTIVITY"],
                   "ZDR":                              [300,  "DIFFERENTIAL_REFLECTIVITY"],
                   "KDP":                              [301,  "SPECIFIC_DIFFERENTIAL_PHASE"],
                   "U10M":                             [1,    "METAR_U_10_METER_WIND"],
                   "V10M":                             [2,    "METAR_V_10_METER_WIND"],
                   "T2M":                              [4,    "METAR_TEMPERATURE_2_METER"],
                   "TD2M":                             [9,    "METAR_DEWPOINT_2_METER"],
                   "H2M":                              [5,    "METAR_SPECIFIC_HUMIDITY_2_METER"],
                   "U_10M":                            [1,    "METAR_U_10_METER_WIND"],
                   "V_10M":                            [2,    "METAR_V_10_METER_WIND"],
                   "T_2M":                             [4,    "METAR_TEMPERATURE_2_METER"],
                   "TD_2M":                            [9,    "DEW_POINT_2_METER"],
                   "H_2M":                             [5,    "METAR_SPECIFIC_HUMIDITY_2_METER"],
                   "TEMP2M":                           [4,    "METAR_TEMPERATURE_2_METER"],
                   "DEWPT2M":                          [9,    "DEW_POINT_2_METER"],
                   "REFL":                             [12,   "REFLECTIVITY"],
                   "FLASH_RATE_2D":                    [2014, "FLASH_RATE_2D"],
                   "METAR_ALTIMETER":                  [71,   "METAR_ALTIMETER"],
                   "LAND_SFC_ALTIMETER":               [70,   "LAND_SFC_ALTIMETER"],
                   "LAND_SFC_DEWPOINT":                [58,   "LAND_SFC_DEWPOINT"],
                   "LAND_SFC_TEMPERATURE":             [25,   "LAND_SFC_TEMPERATURE"],
                   "LAND_SFC_U_WIND_COMPONENT":        [23,   "LAND_SFC_U_WIND_COMPONENT"],
                   "LAND_SFC_V_WIND_COMPONENT":        [24,   "LAND_SFC_V_WIND_COMPONENT"],
                   "GOES_CWP_PATH":                    [80,   "GOES_CWP_PATH"]
                 }
   
   if Print_Table:
         print
         print "VALID INPUT VARIABLE NAME              KIND  DART NAME"
         print "=========================================================================="
         for key in Look_Up_Table.keys():
               print "%35s    %3d    %s" % (key, Look_Up_Table[key][0], Look_Up_Table[key][1])
         return
  
   name2 = name.upper().strip()
   if Look_Up_Table.has_key(name2):
         if DART_name == True:
               return Look_Up_Table[name2][0], Look_Up_Table[name2][1]
         else:
              return Look_Up_Table[name2][0]
   else:
         print "ObType_LookUp cannot find variable:  ", name, name2
         raise SystemExit
         
####################################################################################### 
#
# write_DART_ascii is a program to dump radar data to DART ascii files.
#
# Usage: 
#
#   obs:  a gridded data object (described below)
#
#   fsuffix:  a string containing a label for the radar fields
#             the string "obs_seq" will be prepended, and
#             the filename will have ".txt" appended after fsuffix.
#             If fsuffix is not supplied, data will be write to
#             "obs_seq.txt".
#
#   obs_error:  the observation error for the field (2 m/s, 5 dbz)
#               this is NOT the variance, the stddev!
#               YOU MUST SPECIFY the obs_error, or program will quit.
#
#   obs object spec:  The obs object must have the following  attributes...
#
#       obs.data:       List of observational values
#
#       obs.field:  string name of field (valid:  "reflectivity", "velocity") 
#
#       obs.lats/lons:  list of len(data) of lats and lons of horizontal grid 
#                       locations in degrees.
#
#       obs.hgts:  list of len(data) of the heights of observations
#
#       obs.time:  datetime object of observation times
# 
#  -->  for radial velocity, more metadata is needed
#
#        obs.nyquist:  a 1D array of dimension z, containing nyquist velo for each tilt
#
#        obs.radar_lat:  latitude of radar
#        obs.radar_lon:  longitude of radar
#
#
#        Vr = xR*u + yR*v + zR*(w-vt)
#
#        obs.xR:  list of len(data) for the velocity transformation
#        obs.yR:  list of len(data) for the velocity transformation
#        obs.zR:  list of len(data) for the velocity transformation
#
#
########################################################################################  
def write_DART_ascii_vr(obs, dtime, filename=None, obs_error=None):

   if filename == None:
       print("\n write_DART_ascii:  No output file name is given, writing to %s" % "obs_seq.txt")
       filename = "obs_seq.out"
   else:
       dirname = os.path.dirname(filename)
       basename = "%s.out" % (os.path.basename(filename))
       filename =  os.path.join(dirname, basename)
      
   if obs_error == None:
       print "write_DART_ascii:  No obs error defined for observation, exiting"
       raise SystemExit

# Open ASCII file for DART obs to be written into.  We will add header info afterward
  
   fi = open(filename, "w")
  
   print("\n Writing %s to file...." % filename)
   
   data       = obs.vr.values
   lats       = np.radians(obs.lat.values)
   lons       = np.radians(obs.lon.values)
   hgts       = obs.hgt.values 
   vert_coord = 3
   kind       = ObType_LookUp("VELOCITY")

# Fix the negative lons...

   lons       = np.where(lons > 0.0, lons, lons+(2.0*np.pi))

# extra information

   if kind == ObType_LookUp("VR"):
       platform_nyquist    = _obs_nyquist
       platform_key        = 1
       platform_vert_coord = 3
       
   days    = day_utime.date2num(dtime)
   seconds = np.int(86400.*(days - np.floor(days)))
  
   print("\n Number of good observations:  %d" % data.shape[0])

# Loop over 1D arrays of data
 
   nobs = 0

   for k in np.arange(data.shape[0]):
      
       nobs += 1
       
       fi.write(" OBS            %d\n" % (nobs) )
                
       fi.write("   %20.14f\n" % data[k]  )

       fi.write("   %20.14f\n" % QC_default )
            
       if nobs == 1: 
           fi.write(" %d %d %d\n" % (-1, nobs+1, -1) ) # First obs.
       elif nobs == data.shape[0]:
           fi.write(" %d %d %d\n" % (nobs-1, -1, -1) ) # Last obs.
       else:
           fi.write(" %d %d %d\n" % (nobs-1, nobs+1, -1) ) 
      
       fi.write("obdef\n")
       fi.write("loc3d\n")

# dont know why I needed to have the next set of logic.           
       z = hgts[k]
               
       fi.write("    %20.14f          %20.14f          %20.14f     %d\n" % 
                      (lons[k], lats[k], z, vert_coord))
      
       fi.write("kind\n")

 # If we created zeros, and 0dbz_obtype == True, write them out as a separate data type

       o_error = obs_error
                
       platform_dir1 = obs.xR.values[k]
       platform_dir2 = obs.yR.values[k]
       platform_dir3 = obs.zR.values[k]
          
       platform_lat = np.radians(obs.radarLat.values[k])
       platform_lon = np.radians(obs.radarLon.values[k])
       platform_hgt = obs.radarHeight.values[k]
          
       fi.write("platform\n")
       fi.write("loc3d\n")

       if platform_lon < 0.0:  platform_lon = platform_lon+2.0*np.pi

       fi.write("    %20.14f          %20.14f        %20.14f    %d\n" % 
               (platform_lon, platform_lat, platform_hgt, platform_vert_coord) )
      
       fi.write("dir3d\n")
      
       fi.write("    %20.14f          %20.14f        %20.14f\n" % (platform_dir1, platform_dir2, platform_dir3) )
       fi.write("    %20.14f     \n" % _obs_nyquist  )
       fi.write("    %d          \n" % platform_key )

 # Done with special radial velocity obs back to dumping out time, day, error variance info
  
       fi.write("    %d          %d     \n" % (seconds, days) )

 # Logic for command line override of observational error variances

       fi.write("    %20.14f  \n" % o_error**2 )

       if nobs % 1000 == 0: print(" write_DART_ascii:  Processed observation # %d" % nobs)
  
   fi.close()
  
# To write out header information AFTER we know how big the observation data set is, we have
# to read back in the entire contents of the obs-seq file, store it, rewrite the file
# with header information first, and then dump the contents of obs-seq back inp.  Yuck.

   with file(filename, 'r') as f: f_obs_seq = f.read()

   fi = open(filename, "w")
  
   fi.write(" obs_sequence\n")
   fi.write("obs_kind_definitions\n")

# Deal with case that for reflectivity, 2 types of observations might have been created

   fi.write("       %d\n" % 1)
   akind, DART_name = ObType_LookUp("VELOCITY", DART_name=True)
   fi.write("    %d          %s   \n" % (akind, DART_name) )

   fi.write("  num_copies:            %d  num_qc:            %d\n" % (1, 1))
  
   fi.write(" num_obs:       %d  max_num_obs:       %d\n" % (nobs, nobs) )
      
   fi.write("observations\n")
   fi.write("QC radar\n")
          
   fi.write("  first:            %d  last:       %d\n" % (1, nobs) )

 # Now write back in all the actual DART obs data

   fi.write(f_obs_seq)
  
   fi.close()
  
   print("\n write_DART_ascii:  Created ascii DART file, N = %d written" % nobs)
  
   if kind == ObType_LookUp("REFLECTIVITY") and zero_dbz_obtype and nobs_clearair > 0:
       print(" write_DART_ascii:  Number of clear air obs:             %d" % nobs_clearair)
       print(" write_DART_ascii:  Number of non-zero reflectivity obs: %d" % (nobs - nobs_clearair))

   return
#-------------------------------------------------------------------------------
#
def read_RADAR_VR_NCDF(filename, retFileAttr = False):

    try:
        if retFileAttr == False:
            return xr.open_dataset(filename).to_dataframe()
        else:
            xa = xr.open_dataset(filename)
            return xa.to_dataframe(), xa.attrs
    except:
        print(" read_MRMS_VR_NCDF:  cannot read data file, return None\n")
        return None, None


#-------------------------------------------------------------------------------
# Main function defined to return correct sys.exit() calls

def main(argv=None):
   if argv is None:
       argv = sys.argv
#
# Command line interface 
#
   parser = OptionParser()
   parser.add_option("-d", "--dir", dest="dir",  default=None,  type="string", help = "Directory where VR files are")
   
   parser.add_option(      "--realtime",  dest="realtime",   default=None,  \
               help = "Boolean flag to uses this YYYYMMDDHHMM time stamp for the realtime processing")
 
   parser.add_option("-w", "--write", dest="write",   default=False, \
                           help = "Boolean flag to write DART ascii file", action="store_true")

   parser.add_option("-o", "--out",      dest="out_dir",  default="vr_files",  type="string", \
                           help = "Directory to place output files in")
                           
   (options, args) = parser.parse_args()

#-------------------------------------------------------------------------------

   if options.dir == None:          
       print "\n\n ***** USER MUST SPECIFY A DIRECTORY WHERE FILES ARE *****"
       print "\n                         EXITING!\n\n"
       parser.print_help()
       print
       sys.exit(1)
            
   if options.realtime != None:
       year   = int(options.realtime[0:4])
       mon    = int(options.realtime[4:6])
       day    = int(options.realtime[6:8])
       hour   = int(options.realtime[8:10])
       minute = int(options.realtime[10:12])
       a_time = dtime.datetime(year, mon, day, hour, minute, 0)
   else:
       print "\n\n ***** USER MUST SPECIFY A YYYYMMDDMMHH for realtime argument *****"
       print "\n                         EXITING!\n\n"
       parser.print_help()
       print
       sys.exit(1)

#-------------------------------------------------------------------------------
# Make sure there is a directory to write files into....
 
   if not os.path.exists(options.out_dir):
       try:
           os.mkdir(options.out_dir)
       except:
           print("\n**********************   FATAL ERROR!!  ************************************")
           print("\n PREP_VR:  Cannot create output dir:  %s\n" % options.out_dir)
           print("\n**********************   FATAL ERROR!!  ************************************")      
            
#-------------------------------------------------------------------------------
#
   if options.realtime != None:
   
       suffix = "%s/*%s.nc" % (options.dir, a_time.strftime("%Y%m%d%H%M"))
       print suffix
       in_filenames = glob.glob(suffix)
       
       print in_filenames
       

       try:
           print("\n Prep_VRs:  First file is %s\n" % (in_filenames[0]))
           rlt_filename = "%s_%s" % ("obs_seq_VR", a_time.strftime("%Y%m%d%H%M"))
       except:
           print("\n============================================================================")
           print("\n Prep_VR cannot find a radar VR file in %s" % options.dir)
           print("\n============================================================================")
           sys.exit(1)

#-------------------------------------------------------------------------------
   out_filename = os.path.join(options.out_dir, rlt_filename)
   time         = a_time
   print(" Out filename:  %s\n" % out_filename)

   dataset = []

   begin_time = timeit.time()

   for n, file in enumerate(in_filenames):
                  
       vr_obs = read_RADAR_VR_NCDF(file, retFileAttr=False)

       dataset.append(vr_obs)
      
   end_time = timeit.time()

   print("\n Reading took {0} seconds since the loop started \n".format(end_time - begin_time))

   # Concat the obs_seq files together

   a = pd.concat(dataset, ignore_index=True)

   # Create an xarray dataset for file I/O
   xa = xr.Dataset(a)

   # Reset index to be a master index across all obs
   xa.rename({'dim_0': 'index'}, inplace=True)

   # Write the xarray file out (this is all there is, very nice guys!)
   xa.to_netcdf("%s.nc" % out_filename, mode='w')
   xa.close()

   # Add attributes to the files

   fnc = ncdf.Dataset("%s.nc" % out_filename, mode = 'a')
   fnc.history = "Created " + dtime.datetime.today().strftime(time_format)
   
   fnc.sync()
   fnc.close()
   
   if options.write == True:      
       ret = write_DART_ascii_vr(xa, a_time, filename=out_filename, obs_error=_vr_obs_error)

   end_time = timeit.time()

   print("\n Total radar site procesing took {0} seconds \n".format(end_time - begin_time))
    
#-------------------------------------------------------------------------------
# Main program for testing...
#
if __name__ == "__main__":
    sys.exit(main())
