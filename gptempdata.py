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
temperatures_raw = temperatures_raw[:210,1,:]       #subset of data to work with - one height, removed unnecessary points at end of wind tunnel

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
temperatures_time_avg = lh50_data['s']      #time averaged temperature data
x_observed = lh50_data['p_mm'][:15,0]          #x (crosswind) axis, observed data
y_observed = np.unique(lh50_data['p_mm'][:,1])
y_observed[13] = y_observed[1]      #last row repeats 

#IN PROGRESS: grid shape to data (x, y,temperature), to calculate std at each location

grid_x, grid_y= np.meshgrid(x_observed, y_observed)
temperatures_raw_reshaped =  np.reshape(temperatures_raw,(14,15,20000))  #reshape T for easier expansion of interpolation into 1+ dimensions
T_observed = np.zeros([9,15])        #preallocate matrix. 

for i in range(1,10):
    T_observed[i-1,:] = temperatures_time_avg[1, i*15 - 15:i*15]   #get corresponding crosswind slice temperatures

T_observed_mean = np.mean(T_observed, 0) - np.min(np.mean(T_observed, 0))     #subtract offset, improves fit

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

    
def gaus2(x, A1, mu1, sigma1, A2, mu2, sigma2):
    return gaus(x, A1, mu1, sigma1) + gaus(x, A2, mu2, sigma2)

    
#fit gaussian to distribution
p0 = [1, 90, 15]   #start guess for fitting
coeff1, cov = curve_fit(gaus, x_observed, T_observed_mean, p0 = p0) #inputs can be: gaus,centers, dist,coeff_guess, if using temperature distribution data
hist_fit = gaus(x_observed, *coeff1)

T_observed_adjusted = T_observed_mean - hist_fit    #subtract out first gaussian, to fit second (if necessary)

#fit second gaussian, if necessary 
p0 = [0.12, 40, 5]
coeff2, cov2 = curve_fit(gaus,x_observed, np.abs(T_observed_adjusted), p0 = p0)   #not sure how I feel about abs val..
hist_fit2 = gaus(x_observed, *coeff2)

#create final coefficients and fits
coeff = np.concatenate((coeff1, coeff2), axis = 0)
T_fit = hist_fit + hist_fit2

#plot to check fit
#plt.plot(x_observed,T_observed_mean,'ro',label='Test data'), plt.plot(x_observed,hist_fit,label='Fitted data')

#prediction locations
#x_predicted = np.atleast_2d(np.random.rand(100))*coeff(1)   #random data, around mean
x_predicted = np.atleast_2d(np.linspace(0, 254, 50))       #2 mm prediction sites
x_observed = np.atleast_2d(x_observed)    #make 2d for gaussian process fit. TODO: figure out what atleast_2d does

T_observed = np.atleast_2d(T_observed)    #make 2d for gaussian process fit
T_observed_adjusted = np.atleast_2d(T_observed_adjusted)
T_fit = np.atleast_2d(T_fit)
   
#TODO: make section into separate function
   
#TODO: add optimizer, resize nugget
gp = gaussian_process.GaussianProcess(corr = 'absolute_exponential',
                                      theta0 = 1./25, 
                                      thetaL = None,
                                      thetaU = None)
#                                      nugget = np.std(T_observed,1)/T_observed_mean)

gp.fit(x_observed.T, T_fit.T)

#y_expected_fit = gaus(x_observed,*coeff)     #single gaussian expected y values
T_expected_fit = gaus2(x_observed, *coeff) #coeff)     #expected y values with double-gaussian-fit
T_prediction, y_prediction_MSE = gp.predict(x_predicted.T, eval_MSE = True)   #produce predicted y values
sigma = np.sqrt(y_prediction_MSE)   #get SD of fit at each x_predicted location (for confidence interval)