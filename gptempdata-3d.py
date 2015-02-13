# -*- coding: utf-8 -*-
"""
This script opens data files and extracts relevant data. Then, using a sklearn 
gaussian process package, fits a gaussian to the crosswind (1d) temperature
measurements and interpolates temparture values (relative, not absolute) at a
2 mm interval.

Created on Mon Dec 01 11:51:44 2014

@authors: Sharri and Richard
"""

from scipy.optimize import curve_fit
from sklearn import gaussian_process
import numpy as np
import matplotlib.pyplot as plt

#SECTION 1 (not sure if this should include these imports)
import scipy.io as io
import os

mydir = os.path.dirname(__file__)

#pull out tempearture data 
lhstore2_file = os.path.join(mydir, "data", 'lhstore2.mat')
lhstore2_data = io.loadmat(lhstore2_file)
T_raw = lhstore2_data['store2'].T

z_file = os.path.join(mydir, "data", 'z-positions.mat')
z = io.loadmat(z_file)
zpos = z['y'][0]        #pull out elevation (z) data

#pull out x,y positional data
lh50_file = os.path.join(mydir, 'data', 'final-lh50.mat')
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
##SECTION 2

#TODO: average these repeated points, instead of deleting them with lim
lim = 199      #limit of points used (to remove repeat positions or unwanted positions)

#Make array of observed locations (x,y)
xy_observed = np.zeros((2,lim),dtype = float)
observed_data = np.zeros((3,lim), dtype = float)

#temperatures_time_avg = lh50_data['s']      #time averaged temperature data
xy_observed[0,:] = lh50_data['p_mm'][:lim,0]          #x (crosswind) axis, observed data
xy_observed[1,:] = lh50_data['p_mm'][:lim,1] #Richard: is this the y axis?

#for 3d interpolation:
xyz_observed = np.zeros((3,len(xy_observed.T)),dtype = float)       #observed locations
xyz_observed[:2,:] = xy_observed
xyz_observed[2,:] = zpos[3]*np.ones((1,lim))
T_time_avg_3d = np.mean(T_raw[:lim,0,:],1)      #preallocate observed measurements variable
T_sd = np.std(T_raw[:lim,0,:],1)    #preallocate sd variable

for i in range(1,4):
    T_slice = np.mean(T_raw[:lim,i,:], 1)      #single height layer of temperature   
    T_slice_sd = np.std(T_raw[:lim,i,:],1)
    layer = len(xy_observed.T)      
    xy_slice = np.append(xy_observed,(np.ones((1,layer))*zpos[3-i]),axis = 0)     #single height layer of position data
    xyz_observed = np.append(xyz_observed, xy_slice, axis = 1)      #3d positions
    T_time_avg_3d = np.append(T_time_avg_3d, T_slice, axis = 1)       #resized tempearture data
    T_sd = np.append(T_sd,T_slice_sd,axis = 1)

#look for and remove nans () 
if np.any(np.isnan(T_time_avg_3d)):
    f = np.where(np.isnan(T_time_avg_3d))      #find the indices of Nans
    T_time_avg_3d = np.delete(T_time_avg_3d, f)
    xyz_observed = np.delete(xyz_observed, f, axis = 1)
    T_sd = np.delete(T_sd,f)

#fit all of the observed data into one array, for ease of use of fitting function
observed_data_3d = np.zeros((4,len(xyz_observed.T)), dtype = float)
observed_data_3d[:3,:] = xyz_observed
observed_data_3d[3,:] = T_time_avg_3d.T

#SECTION 3
#TODO: move everything below this to a function and/or separate script

#prediction locations, make 

x_predict = np.atleast_2d(np.linspace(0, 254, 25))       #2 mm prediction sites
y_predict = np.atleast_2d(np.linspace(100, 850, 15))
z_predict = np.atleast_2d(np.linspace(80, 280, 20))

x1,x2,x3 = np.meshgrid(x_predict, y_predict, z_predict)
xyz_predict = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size), x3.reshape(x3.size)]).T

#calculate noise (required)
nugget =  (T_sd/T_time_avg_3d)**2
nugget = nugget       #deletes repeated measurment locations
  
#TODO: make section into separate function
   
gp = gaussian_process.GaussianProcess(corr = 'absolute_exponential',
                                      theta0 = 1./25, 
                                      thetaL = 1e-1,
                                      thetaU = .3,
                                      normalize = True,
                                      nugget = nugget)
#when height = 1, thetaL = .1, thetaU = .3
#when height = 0, thetaL = 10e-2,thetaU = .3

gp.fit(xyz_observed.T, T_time_avg_3d)
 
#Target value error will come up with that last repeated row. It can't have 
#multiple measurements at the same location. Consider deleting that repeated
#last row of measurements or take a mean or stack the timeseries onto the 
#first measurement, which will effectivly average the values.

T_prediction, y_prediction_MSE = gp.predict(xyz_predict, eval_MSE = True)   #produce predicted y values
sigma = np.sqrt(y_prediction_MSE)   #get SD of fit at each x_predicted location (for confidence interval)

