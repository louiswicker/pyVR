#!/usr/bin/env python


import time
import logging
import os, sys
import datetime
import glob

#_VR_feed         = "/work/LDM/MRMS"
_VR_feed         = "/work/anthony.reinhart/VRtest/20170516/GriddedPoint3km"
#_VR_obs_seq      = "/work/wicker/REALTIME/"
_VR_obs_seq      = "/work/wicker/REALTIME/pyVr/3km"
_NEWSe_grid_info = "/scratch/wof/realtime/radar_files"
_prep_volume     = "/work/wicker/REALTIME/pyVr/prep_volume.py"
_prep_vr         = "/work/wicker/REALTIME/pyVr/prep_vr.py"

year       = 2017
mon        = 5
day        = [16, 16]   # start and stop days
hour       = [23, 23]    # start and stop hours
min        = [0, 10]

start_time = datetime.datetime(year, mon, day[0], hour[0], min[0], 0)
stop_time  = datetime.datetime(year, mon, day[1], hour[1], min[1], 05)
dtime      = datetime.timedelta(minutes=15)

obs_seq_out_dir = os.path.join(_VR_obs_seq, start_time.strftime("%Y%m%d"))

# create path for NEWSe radar file

radar_csh_file = os.path.join(_NEWSe_grid_info, ("radars.%s.csh" % start_time.strftime("%Y%m%d")))

# Parse center lat and lon out of the c-shell radar file - HARDCODED!
# If the file does not exist, then we exit out of this run

try:
    fhandle = open(radar_csh_file)
except:
    print("\n %s" % ("=" * 120))
    print("\n CANNOT OPEN radar CSH file, exiting MRMS processing:  %s" % radar_csh_file)
    print("\n %s" % ("=" * 120))
    sys.exit(1)

all_lines  = fhandle.readlines()
fhandle.close()

# Separate out radar string list

radars     = all_lines[6].split(" K")[1:]
radars[-1] = radars[-1].split(" )")[0]
radars     = list("K%s" % item for item in radars)

print("\n %s" % ("=" * 120))
print("\n Radars: %s" % ' '.join(radars))
print("\n %s" % ("=" * 120))

# Main catchup loop
#python prep_volume.py -d /work/wicker/Point/KDDC --realtime 201705162100 -o vr_files 

while start_time < stop_time:

#   MRMS_dir = os.path.join(_MRMS_feed, start_time.strftime("%Y/%m/%d"))
    
# Process individual radar directories

    for rad in radars:

       dir = os.path.join(_VR_feed, rad)

       print("\n Reading from VR directory:  %s\n" % dir)
    
       print("\n >>>>=======BEGIN===============================================================")
       cmd = "%s -d %s -o %s --realtime %s -p" % (_prep_volume, dir, obs_seq_out_dir, start_time.strftime("%Y%m%d%H%M"))

       print("\n Prep_VOLUME called at %s" % (time.strftime("%Y-%m-%d %H:%M:%S")))
       print(" Cmd: %s" % (cmd))
       ret = os.system("%s" % cmd)
       if ret != 0:
           print("\n ============================================================================")
           print("\n Prep_VOLUME cannot find a RF file between [-2,+1] min of %s" % start_time.strftime("%Y%m%d%H%M"))
           print("\n ============================================================================")
       print("\n <<<<<=======END================================================================")

# Combine all the radars together
    print("\n >>>>=======BEGIN===============================================================")
    cmd = "%s -d %s -o %s --realtime %s --write " % (_prep_vr, obs_seq_out_dir, obs_seq_out_dir, start_time.strftime("%Y%m%d%H%M"))

    print("\n Prep_VR called at %s" % (time.strftime("%Y-%m-%d %H:%M:%S")))
    print(" Cmd: %s" % (cmd))
    ret = os.system("%s" % cmd)
    if ret != 0:
        print("\n ============================================================================")
        print("\n Prep_VR cannot find a RF file between [-2,+1] min of %s" % start_time.strftime("%Y%m%d%H%M"))
        print("\n ============================================================================")
    print("\n <<<<<=======END================================================================")


    start_time = start_time + dtime

