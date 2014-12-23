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
temperatures_raw = lhstore2_data['store2'].T
## lhstore2 is lh50's store with two channel error corrected
### temperatures_raw.shape() ==>  (215, 4, 20000)

#TODO: Allow for selection or incorporation of other heights (2nd dimension of temperatures_raw)
temperatures_raw = temperatures_raw[:210,0,:]       #subset of data to work with - one height, removed unnecessary points at end of wind tunnel

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
T_time_avg = np.mean(temperatures_raw,1) 
T_sd = np.std(T_observed,2)

#fit all of the observed data into one array, for ease of use of fitting function
observed_data[:2,:] = xy_observed
observed_data[2,:] = T_time_avg

#get distribution of temperatures from samples
#bins = np.linspace(15, 30, 100) #histogram bins
#h,b = np.historgram(temperature_adjusted, bins)
#centers = (b[:-1]+b[1,:])/2
#
#h2 = np.float(np.sum(h,0))  
#dist = h/h2   #convert histogram to probability
    
def gaus2d(xy_observed, A, mu_x, sigma_x, mu_y, sigma_y):
    """2-dimensional gaussian
    """
    return A * np.exp(-((xy_observed[0]-mu_x)**2/(2.*sigma_x**2)+
    (xy_observed[1]-mu_y)**2/(2.*sigma_y**2)))

def lin_gaus_2d(xy_observed, m, b, A, mu_x, sigma_x, mu_y, sigma_y):
    """ 2d gaussian + linear slope. should be good approximation for this data set
    """
    return (xy_observed[1]*m+b)+A * np.exp(-((xy_observed[0]-mu_x)**2/(2.*sigma_x**2)+
    (xy_observed[1]-mu_y)**2/(2.*sigma_y**2)))
    
#guess for coefficients
guess = [1/700, 0,
      0.13, 90, 15,
      450, 1]

coeff, cov = curve_fit(lin_gaus_2d,observed_data[:2,:],observed_data[3,:],p0=guess)
T_fit = lin_gaus_2d(xy_observed,*coeff)

#plot to check fit
#plt.plot(x_observed,temp_observed_mean,'ro',label='Test data'), plt.plot(x_observed,histemp_fit,label='Fitted data')

#TODO: fix all below such that it works.
#TODO: move everything below this to a function and/or separate script

#prediction locations
#x_predicted = np.atleast_2d(np.random.rand(100))*coeff(1)   #random data, around mean
x_predicted = np.atleast_2d(np.linspace(0, 254, 50))       #2 mm prediction sites
y_predicted = np.atleast_2d(np.linspace(0, 850,100))


T_observed_mean = np.atleast_2d(T_observed_mean)    #make 2d for gaussian process fit
T_fit_x = np.atleast_2d(T_fit_x)
T_fit_y = np.atleast_2d(T_fit_y)

#nugget =  (T_sd/T_observed_mean)**2 
  
#TODO: make section into separate function
   
#TODO: add optimizer, resize nugget
gp = gaussian_process.GaussianProcess(corr = 'absolute_exponential',
                                      theta0 = 1./25, 
                                      thetaL = None,
                                      thetaU = None)
                                      #nugget = nugget)

gp.fit(x_observed.T, T_fit.T)

#y_expected_fit = gaus(x_observed,*coeff)     #single gaussian expected y values
T_expected_fit = lin_gaus_2d(xy_observed, *coeff) #coeff)     #expected y values with double-gaussian-fit
T_prediction, y_prediction_MSE = gp.predict(x_predicted.T, eval_MSE = True)   #produce predicted y values
sigma = np.sqrt(y_prediction_MSE)   #get SD of fit at each x_predicted location (for confidence interval)