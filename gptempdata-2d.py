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
    
def lin_gaus_2d(xy_observed, A1, mu_x1, sigma_x1, mu_y1, sigma_y1, A2, mu_x2, sigma_x2, mu_y2, sigma_y2, m, b):
    """sum of two 2-dimensional gaussians + y-dimension linear slope
    """
    return A1 * np.exp(-((xy_observed[0]-mu_x1)**2/(2.*sigma_x1**2) + 
    (xy_observed[1]-mu_y1)**2/(2.*sigma_y1**2))) + A2 * np.exp(-((xy_observed[0]-mu_x2)**2/(2.*sigma_x2**2)+
    (xy_observed[1]-mu_y2)**2/(2.*sigma_y2**2))) + (xy_observed[1]*m + b)
    
#initial paramater values for fit function
p0 = [1, 69, 1, 500,5,
         1.5, 100, 1, 500, 5,
         1/800, 0]

coeff, cov = curve_fit(lin_gaus_2d, xy_observed, observed_data[2,:]-np.mean(observed_data[2,:]), p0 = p0)
T_fit = lin_gaus_2d(xy_observed, *coeff)

#convert such that this stupid gp can actually understand what's going on (dimension matching)
T_fit_2d = np.zeros([2,len(T_fit.T)])
T_fit_2d[0,:] = T_fit
T_fit_2d[1,:] = T_fit

#plot to check fit
#plt.plot(x_observed,temp_observed_mean,'ro',label='Test data'), plt.plot(x_observed,histemp_fit,label='Fitted data')

#TODO: move everything below this to a function and/or separate script

#prediction locations, make 

x_predicted = np.atleast_2d(np.linspace(0, 254, 15))       #2 mm prediction sites
y_predicted = np.atleast_2d(np.linspace(0, 850, 14))

x1,x2 = np.meshgrid(x_predicted, y_predicted)
xy_predicted = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size)]).T

#calculate noise (required)
nugget =  T_sd/T_time_avg)**2
nugget = nugget[:181]       #deletes repeated measurment locations
  
#TODO: make section into separate function
   
#TODO: add optimizer, resize nugget
gp = gaussian_process.GaussianProcess(corr = 'absolute_exponential',
                                      theta0 = 1./25, 
                                      thetaL = None,
                                      thetaU = None,
                                      nugget = nugget)
#TODO: nugget array size error
#nugget has to be at least 2d


gp.fit(xy_observed[:181,:], T_time_avg[:181])
   
#Target value error will come up with that last repeated row. It can't have 
#multiple measurements at the same location. Consider deleting that repeated
#last row of measurements or take a mean or stack the timeseries onto the 
#first measurement, which will effectivly average the values.

#y_expected_fit = gaus(x_observed,*coeff)     #single gaussian expected y values
#T_expected_fit = lin_gaus_2d(xy_observed, *coeff) #coeff)     #expected y values with double-gaussian-fit
T_prediction, y_prediction_MSE = gp.predict(xy_predicted, eval_MSE = True)   #produce predicted y values
sigma = np.sqrt(y_prediction_MSE)   #get SD of fit at each x_predicted location (for confidence interval)

