# -*- coding: utf-8 -*-
"""
fetch_3D_temp_data

Input:
User input filename string
 
Output:
temperature data and the positional data as variables (without conditional names, as with the LH-oriented name scheme in the file now). 
Maybe this output can be a class? I'm not sure if that's useful here. Anyway, that output should get entered into the second script. 


Created on Fri Feb 13 11:19:31 2015
@author: Richard Decal, decal@uw.edu
"""

import scipy.io as io
import os.path

#==============================================================================
#  File I/O
#==============================================================================

mydir = os.path.dirname(__file__)
print """Current directory is: \n""", mydir, "\n \n"

#==============================================================================
# Loading temp data
#==============================================================================
print "Loading temperature data"
print """Enter temperature data file name. If none entered, default is %s """ % lhstore2_file, "\n \n"
Temp_data_file = raw_input("Input temperature data file dir: ")
if Temp_data_file == '':
    print "No input, using lhstore2.mat"
    Temp_data_file = os.path.join(mydir, "data", 'lhstore2.mat')
Temp_data = io.loadmat(Temp_data_file)


#pull out temp data 
T_raw = Temp_data['store2'].T
    
#==============================================================================
# Loading z file, then pull out elevation data (z)
#==============================================================================

z_file = os.path.join(mydir, "data", 'z-positions.mat')
z = io.loadmat(z_file)
zpos = z['y'][0]        

#load positional data file
lh50_file = os.path.join(mydir, 'data', 'final-lh50.mat')
#pull out x,y positional data
lh50_data = io.loadmat(lh50_file)

#==============================================================================
# lh50_data.keys() ==> ['p_in', 'p_mm', 'p', 's', 'store', '__header__', '__globals__',  '__version__']
# 'p_in' => x,y positions in inches
# 'p_mm' = > x,y positions in mm
# 'p' => x,y positions in grid
# 's' => time averaged temperature
# 'store' => raw temp data
# 'z' => elevation positions
#==============================================================================
