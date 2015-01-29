# -*- coding: utf-8 -*-
"""
This script imports processed trajectory data into python. 

Created on Tue Jan 27 17:06:42 2015

@author: Sharri
"""

import scipy.io as io
import matplotlib.pyplot as plt

#TODO: convert lines 14 and 16 to os functions
data = io.loadmat('C:/Users/Sharri/Dropbox/Le grand dossier du Sharri/Data/Tracking Data/latestdata.mat')

store = data['rhon']   #extract the struct from the file

#For unprocessed data. You'll probably never use this.
#raw  = store['raw']     # 3d trajectory position of each file
#camxy = store['camxy']  # xy positions of trajectories in each camera
#
#a = raw[0][0]           #extracts the first trajectory from the file


#TODO: put this in a loop such all trajectories are extracted

a = store['KF']

tracks = a[0]

#TODO: loop this too (size of track = number of trajectories)
track_1 = track[0][0][0][0] #i agree, this is fucking ridiculous
# to cycle through loops, the first zero determines the trajectory number. 
#so track_2 would be: track_2 = track[1][0][0][0] because god hates you.



