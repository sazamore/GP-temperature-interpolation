# -*- coding: utf-8 -*-
"""
This script opens data files and extracts relevant data. Then, using a sklearn 
gaussian process package, fits a gaussian to the crosswind (1d) temperature
measurements and interpolates temparture values (relative, not absolute) at a
2 mm interval.

TODO: make gaussian process it's own script

Created on Mon Dec 01 11:51:44 2014

@authors: Sharri and Richard
"""

from scipy.optimize import curve_fit
from sklearn import gaussian_process
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io

import os
mydir = os.path.dirname(__file__)

#pull out tempearture data 
lhstore2_file = os.path.join(mydir, "data", 'lhstore2.mat')
lhstore2_data = io.loadmat(lhstore2_file)
T_raw = lhstore2_data['store2'].T
## lhstore2 is lh50's store with two channel error corrected
### temperatures_raw.shape() ==>  (215, 4, 20000)

#TODO: Allow for selection or incorporation of other heights (2nd dimension of temperatures_raw)
T_raw = T_raw[:210,0,:]       #subset of data to work with - one height, removed unnecessary points at end of wind tunnel

#pull out positional data
lh50_file = os.path.join(mydir, 'data', 'final-lh50.mat')
lh50_data = io.loadmat(lh50_file)
#==============================================================================
# lh50_data.keys() ==> ['p_in', 'p_mm', 'p', 's', 'store', '__header__', '__globals__',  '__version__']
# 'p_in' => pos in inches
# 'p_mm' = > positions in mm
# 'p' => pos in grid
# 's' => time averaged temperature
# 'store' => raw temp data
#==============================================================================

#Make array of observed locations (x,y)
xy_observed = np.zeros((2,210),dtype = float)
observed_data = np.zeros((3,210), dtype = float)

#temperatures_time_avg = lh50_data['s']      #time averaged temperature data
xy_observed[0,:] = lh50_data['p_mm'][:210,0]          #x (crosswind) axis, observed data
xy_observed[1,:] = lh50_data['p_mm'][:210,1]

#Get temparature data 
T_time_avg = np.mean(T_raw,1) 
T_sd = np.std(T_raw,1)

#fit all of the observed data into one array, for ease of use of fitting function
observed_data[:2,:] = xy_observed
observed_data[2,:] = T_time_avg

#TODO: move everything below this to a function and/or separate script

#prediction locations, make 

x_predicted = np.atleast_2d(np.linspace(0, 254, 15))       #2 mm prediction sites
y_predicted = np.atleast_2d(np.linspace(0, 850, 14))

x1,x2 = np.meshgrid(x_predicted, y_predicted)
xy_predicted = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size)]).T

#xy_predicted = track_1[:,:1]

#calculate noise (required)
nugget =  (T_sd/T_time_avg)**2
nugget = nugget[:181]       #deletes repeated measurment locations
  
#TODO: make section into separate function
   
gp = gaussian_process.GaussianProcess(corr = 'absolute_exponential',
                                      theta0 = 1./25, 
                                      thetaL = 1e-2,
                                      thetaU = 1,
                                      normalize = True,
                                      random_start = 100,
                                      nugget = nugget)

gp.fit(xy_observed.T[:181,:], T_time_avg[:181])
#gp.fit(xy_observed.T[:181,:], T_raw[:181,:])  #with time variants, this fits each time step...non ideal.
 
#Target value error will come up with that last repeated row. It can't have 
#multiple measurements at the same location. Consider deleting that repeated
#last row of measurements or take a mean or stack the timeseries onto the 
#first measurement, which will effectivly average the values.

T_prediction, y_prediction_MSE = gp.predict(xy_predicted, eval_MSE = True)   #produce predicted y values
sigma = np.sqrt(y_prediction_MSE)   #get SD of fit at each x_predicted location (for confidence interval)

