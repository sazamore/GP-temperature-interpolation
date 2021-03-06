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

#temperatures_time_avg = lh50_data['s']      #time averaged temperature data
x_raw = lh50_data['p_mm'][:15,0]          #x (crosswind) axis, observed data
y_raw = np.unique(lh50_data['p_mm'][:,1])
y_raw[13] = y_raw[1]      #last row repeats 

#IN PROGRESS: grid shape to data (x, y,temperature), to calculate std at each location

x_observed, y_observed= np.meshgrid(x_raw, y_raw)
x_observed = x_observed[0,:]
y_observed = y_observed[:,0]
T_observed =  np.reshape(temperatures_raw,(14,15,20000))  #reshape T for easier expansion of interpolation into 1+ dimensions

T_time_avg = np.mean(T_observed,2)
T_observed_y = np.mean(T_time_avg, 1) - np.min(np.mean(T_time_avg,1))    
T_observed_x = np.mean(T_time_avg, 0) - np.min(np.mean(T_time_avg,0))    
T_sd = np.std(T_observed,2)

#get distribution of temperatures from samples
#bins = np.linspace(15, 30, 100) #histogram bins
#h,b = np.historgram(temperature_adjusted, bins)
#centers = (b[:-1]+b[1,:])/2
#
#h2 = np.float(np.sum(h,0))  
#dist = h/h2   #convert histogram to probability
    
def gaus(x, A, mu, sigma):
    """"gaussian function, for fitting to distribution    
    """
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))


def sum_gaus(x, A1, mu1, sigma1, A2, mu2, sigma2):
    """Sum of (2) gaussians, to fit crosswind distribution
    """
    return gaus(x, A1, mu1, sigma1) + gaus(x, A2, mu2, sigma2)
    
def lin_gaus(x,m,b,A,mu,sigma):
    """linear formula to fit upwind distribution
    """
    return m*x+b + gaus(x,A,mu,sigma)

    
#TODO: add a 2d fit, maybe it can replace all of this--2 2D gaussians and a linear (y) fit

#fit sum of gaussians to distribution
p0 = [1, 90, 15 ,
      0.12, 50, 2]
coeff_x, cov = curve_fit(sum_gaus,x_observed,T_observed_x,p0=p0)
T_fit_x = sum_gaus(x_observed,*coeff_x)

#y-dimension, linear fit
p0 = [1/800, 0, 1.2, 450, 5]
coeff_y, cov = curve_fit(lin_gaus, y_observed, T_observed_y, p0=p0)
T_fit_y = lin_gaus(y_observed, *coeff_y)

#plot to check fit
#plt.plot(x_observed,temp_observed_mean,'ro',label='Test data'), plt.plot(x_observed,histemp_fit,label='Fitted data')

#merge the two dimensions together
def lin_gaus_2d(x, y, A, mu_x, sigma_x, mu_y, sigma_y, m, b):
    """2-dimensional gaussian + linear offset in y dimension
    """
    return A * np.exp(-((x-mu_x)**2/(2.*sigma_x**2)+(y-mu_y)**2/(2.*sigma_y**2))) + y*m + b

#fit_2d = lin_gaus_2d(x_observed, y_observed, A,)

#TODO: move everything below this to a function and/or separate script

#prediction locations
#x_predicted = np.atleast_2d(np.random.rand(100))*coeff(1)   #random data, around mean
x_predicted = np.atleast_2d(np.linspace(0, 254, 50))       #2 mm prediction sites
x_observed = np.atleast_2d(x_observed)    #make 2d for gaussian process fit. TODO: figure out what atleast_2d does
y_predicted = np.atleast_2d(np.linspace(0, 850,100))
y_observed = np.atleast_2d(y_observed)

T_time_avg = np.atleast_2d(T_time_avg)    #make 2d for gaussian process fit
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
T_expected_fit = sum_gaus(x_observed, *coeff) #coeff)     #expected y values with double-gaussian-fit
T_prediction, y_prediction_MSE = gp.predict(x_predicted.T, eval_MSE = True)   #produce predicted y values
sigma = np.sqrt(y_prediction_MSE)   #get SD of fit at each x_predicted location (for confidence interval)