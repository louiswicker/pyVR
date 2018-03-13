
#############################################################
#
# Create for NEWSe processing
#
#############################################################
import os
import sys
import glob
import time as timeit

import numpy as np
import netCDF4 as ncdf

QC_default = 0
# missing value
_missing = -99999.

# This flag adds an k/j/i index to the DART file, which is the index locations of the gridded data
_write_grid_indices = False

# if data object is 0dbz, then write it as a separate type if true

_zero_dbz_obtype = True

_radar_file = "nexrad_stations.txt"

#==================================================================================
def read_radar_location_file(radar_file = None, meters=False):
    """
    Reads the nexrad_stations.txt file, sometimes the altitudes are in feet, others
    in meters.  Default setting is for altitude to be in English units (feet),
    convert to meters.
    """

    if radar_file == None:
         f = open(_radar_file, 'r')
    else:
         f = open(radar_file, 'r')

    header1 = f.readline()

    radar_locations_dict = {}

    one_sixty = 1./ 60.

    for line in f:
        col = line.split()
        lat = float(col[-5])
        lon = float(col[-4])
        if meters:
            alt = float(col[-3])
        else:
            alt = 0.0003048 * float(col[-3])
        radar_locations_dict[col[1]] = (lat, lon, alt)

    return radar_locations_dict

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

########################################################################

def beam_elv(sfc_range, z):

########################################################################
#
#     PURPOSE:
#
#     Calculate the elevation angle (elvang) and the along
#     ray-path distance (range) of a radar beam
#     crossing through the given height and along-ground
#     distance.
#
#     This method assumes dn/dh is constant such that the
#     beam curves with a radius of 4/3 of the earth's radius.
#     This is dervied from Eq. 2.28 of Doviak and Zrnic',
#     Doppler Radar and Weather Observations, 1st Ed.
#
########################################################################
#
#     AUTHOR: Keith Brewster
#     10/10/95
#
#     MODIFICATION HISTORY: adapted to python by Lou Wicker (thanks Keith)
#
########################################################################
#
#     INPUT:
#       sfc_range:    Distance (meters) along ground from radar 
#       z        :    Height above radar
#
#     OUTPUT
#       elvang   Elevation angle (degrees) of radar beam
#
########################################################################
   eradius=6371000.
   frthrde=(4.*eradius/3.)
   eighthre=(8.*eradius/3.)
   fthsq=(frthrde*frthrde)

   if sfc_range > 0.0:
       hgtdb = frthrde + z
       rngdb = sfc_range/frthrde

       elvrad = np.arctan((hgtdb*np.cos(rngdb) - frthrde)/(hgtdb * np.sin(rngdb)))

       return np.rad2deg(elvrad)

   else:
       return -999.

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
def write_DART_ascii_list(obs, filename=None, obs_error=None, zero_dbz_obtype=_zero_dbz_obtype,
                          levels = None, QC_info=None, zero_levels=[]):

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
  
   print("\n Writing %s to file...." % obs.field.upper())
   
   data       = obs.data
   lats       = np.radians(obs.lats)
   lons       = np.radians(obs.lons)
   hgts       = obs.hgts 
   vert_coord = 3
   kind       = ObType_LookUp(obs.field.upper())

# Fix the negative lons...

   lons       = np.where(lons > 0.0, lons, lons+(2.0*np.pi))

# extra information

   if kind == ObType_LookUp("VR"):
       platform_nyquist    = obs.nyquist
       platform_key        = 1
       platform_vert_coord = 3

# Use the volume mean time for the time of the volume

   dtime   = obs.time
       
#  days    = ncdf.date2num(dtime, units = "days since 1601-01-01 00:00:00")
#  seconds = np.int(86400.*(days - np.floor(days)))

   days = 30000
   seconds = 3000
  
   data_length = data.size
   print("\n Number of good observations:  %d" % data_length)

# Loop over 1D arrays of data
 
   nobs = 0

   for k in np.arange(data.shape[0]):
      
       nobs += 1
  
       if _write_grid_indices:
           fi.write(" OBS            %d     \n" % (nobs,k) )
       else:
           fi.write(" OBS            %d\n" % (nobs) )
              
       fi.write("   %20.14f\n" % data[k]  )

       fi.write("   %20.14f\n" % QC_default )
            
       if nobs == 1: 
           fi.write(" %d %d %d\n" % (-1, nobs+1, -1) ) # First obs.
       elif nobs == data_length:
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

 # If this GEOS cloud pressure observation, write out extra information (NOTE - NOT TESTED FOR HDF2ASCII LJW 04/13/15)
 # 
 #       if kind == ObType_LookUp("GOES_CWP_PATH"):
 #           fi.write("    %20.14f          %20.14f  \n" % (row["satellite"][0], row["satellite"][1]) )
 #           fi.write("    %20.14f  \n" % (row["satellite"][2]) )

 # Check to see if its radial velocity and add platform informationp...need BETTER CHECK HERE!
      
       if kind == ObType_LookUp("VR"):
          
           platform_dir1 = obs.xR[k]
           platform_dir2 = obs.yR[k]
           platform_dir3 = obs.zR[k]
              
           platform_lat = np.radians(obs.rlat[k])
           platform_lon = np.radians(obs.rlon[k])
           platform_hgt = obs.rhgt[k]
              
           fi.write("platform\n")
           fi.write("loc3d\n")

           if platform_lon < 0.0:  platform_lon = platform_lon+2.0*np.pi

           fi.write("    %20.14f          %20.14f        %20.14f    %d\n" % 
                   (platform_lon, platform_lat, platform_hgt, platform_vert_coord) )
          
           fi.write("dir3d\n")
          
           fi.write("    %20.14f          %20.14f        %20.14f\n" % (platform_dir1, platform_dir2, platform_dir3) )
           fi.write("    %20.14f     \n" % obs.nyquist  )
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

   if kind == ObType_LookUp("REFLECTIVITY") and zero_dbz_obtype and nobs_clearair > 0:
       fi.write("       %d\n" % 2)
       akind, DART_name = ObType_LookUp(obs.field.upper(), DART_name=True)
       fi.write("    %d          %s   \n" % (akind, DART_name) )
       akind, DART_name = ObType_LookUp("RADAR_CLEARAIR_REFLECTIVITY", DART_name=True) 
       fi.write("    %d          %s   \n" % (akind, DART_name) )
   else:
       fi.write("       %d\n" % 1)
       akind, DART_name = ObType_LookUp(obs.field.upper(), DART_name=True)
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
  
#####################################################################################################
def write_netcdf_radar_file(ref, vel, filename=None):
    
   _time_units    = 'seconds since 1970-01-01 00:00:00'
   _calendar      = 'standard'

   if filename == None:
       print("\n write_DART_ascii:  No output file name is given, writing to %s" % "obs_seq.txt")
       filename = "obs_seq.nc"
   else:
       dirname = os.path.dirname(filename)
       basename = "%s_%s.nc" % ("obs_seq", os.path.basename(filename))
       filename =  os.path.join(dirname, basename)

   _stringlen     = 8
   _datelen       = 19
     
# Extract grid and ref data
        
   dbz        = ref.data
   lats       = ref.lats
   lons       = ref.lons
   hgts       = ref.zg + ref.radar_hgt
   kind       = ObType_LookUp(ref.field.upper())  
   R_xy       = np.sqrt(ref.xg[20]**2 + ref.yg[20]**2)
   elevations = beam_elv(R_xy, ref.zg[:,20,20])
 
# if there is a zero dbz obs type, reform the data array 
   try:
       nx1, ny1       = ref.zero_dbz.shape
       zero_data      = np.ma.zeros((2, ny1, nx1), dtype=np.float32)
       zero_hgts      = np.ma.zeros((2, ny1, nx1), dtype=np.float32)
       zero_data[0]   = ref.zero_dbz
       zero_data[1]   = ref.zero_dbz
       zero_hgts[0:2] = ref.zero_dbz_zg[0:2]
       cref           = ref.cref
       zero_flag = True
       print("\n write_DART_ascii:  0-DBZ separate type added to netcdf output\n")       
   except AttributeError:
       zero_flag = False
       print("\n write_DART_ascii:  No 0-DBZ separate type found\n")
     
# Extract velocity data
  
   vr                  = vel.data
   platform_lat        = vel.radar_lat
   platform_lon        = vel.radar_lon
   platform_hgt        = vel.radar_hgt

# Use the volume mean time for the time of the volume
      
   dtime   = ncdf.num2date(ref.time['data'].mean(), ref.time['units'])
   days    = ncdf.date2num(dtime, units = "days since 1601-01-01 00:00:00")
   seconds = np.int(86400.*(days - np.floor(days)))  
 
# create the fileput filename and create new netCDF4 file

#filename = os.path.join(path, "%s_%s%s" % ("Inflation", DT.strftime("%Y-%m-%d_%H:%M:%S"), ".nc" ))

   print "\n -->  Writing %s as the radar file..." % (filename)
   
   rootgroup = ncdf.Dataset(filename, 'w', format='NETCDF4')
     
# Create dimensions

   shape = dbz.shape
 
   rootgroup.createDimension('nz',   shape[0])
   rootgroup.createDimension('ny',   shape[1])
   rootgroup.createDimension('nx',   shape[2])
   rootgroup.createDimension('stringlen', _stringlen)
   rootgroup.createDimension('datelen', _datelen)
   if zero_flag:
       rootgroup.createDimension('nz2',   2)
 
# Write some attributes

   rootgroup.time_units   = _time_units
   rootgroup.calendar     = _calendar
   rootgroup.stringlen    = "%d" % (_stringlen)
   rootgroup.datelen      = "%d" % (_datelen)
   rootgroup.platform_lat = platform_lat
   rootgroup.platform_lon = platform_lon
   rootgroup.platform_hgt = platform_hgt

# Create variables

   R_type  = rootgroup.createVariable('REF', 'f4', ('nz', 'ny', 'nx'), zlib=True, shuffle=True )    
   V_type  = rootgroup.createVariable('VEL', 'f4', ('nz', 'ny', 'nx'), zlib=True, shuffle=True )
 
   if zero_flag:
       R0_type   = rootgroup.createVariable('0REF',  'f4', ('nz2', 'ny', 'nx'), zlib=True, shuffle=True )    
       Z0_type   = rootgroup.createVariable('0HGTS', 'f4', ('nz2', 'ny', 'nx'), zlib=True, shuffle=True )
       CREF_type = rootgroup.createVariable('CREF', 'f4', ('ny', 'nx'), zlib=True, shuffle=True )
     
   V_dates = rootgroup.createVariable('date', 'S1', ('datelen'), zlib=True, shuffle=True)
   V_xc    = rootgroup.createVariable('XC', 'f4', ('nx'), zlib=True, shuffle=True)
   V_yc    = rootgroup.createVariable('YC', 'f4', ('ny'), zlib=True, shuffle=True)
   V_el    = rootgroup.createVariable('EL', 'f4', ('nz'), zlib=True, shuffle=True)

   V_lat   = rootgroup.createVariable('LATS', 'f4', ('ny'), zlib=True, shuffle=True)
   V_lon   = rootgroup.createVariable('LONS', 'f4', ('nx'), zlib=True, shuffle=True)
   V_hgt   = rootgroup.createVariable('HGTS', 'f4', ('nz', 'ny', 'nx'), zlib=True, shuffle=True)

# Write variables

   rootgroup.variables['date'][:] = ncdf.stringtoarr(dtime.strftime("%Y-%m-%d_%H:%M:%S"), _datelen)
 
   rootgroup.variables['REF'][:]  = dbz[:]
   rootgroup.variables['VEL'][:]  = vr[:]
   rootgroup.variables['XC'][:]   = ref.xg[:]
   rootgroup.variables['YC'][:]   = ref.yg[:]
   rootgroup.variables['EL'][:]   = elevations[:]
   rootgroup.variables['HGTS'][:] = ref.zg[:]
   rootgroup.variables['LATS'][:] = lats[:]
   rootgroup.variables['LONS'][:] = lons[:]
 
   if zero_flag:
      rootgroup.variables['0REF'][:]   = zero_data
      rootgroup.variables['0HGTS'][:]  = zero_hgts
      rootgroup.variables['CREF'][:]   = cref
 
   rootgroup.sync()
   rootgroup.close()
 
   return filename  
 
