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
import scipy.io as io

data = io.loadmat('C:\Users\Sharri\Documents\MATLAB\Temp stuff\lhstore2.mat')   #open data file (.mat)

temperature_adjusted = data['store2']    #pull out tempearture data 

data = io.loadmat('C:\Users\Sharri\Documents\MATLAB\Temp stuff\lhdata.mat')

pos_mm = data['p_mm']   #pull out positional data
temperature_mean = data['s']      #time averaged temperature data

x_observed = pos_mm[:15,0]           #x (crosswind) axis, observed data

y_observed = np.zeros([9,15])        #preallocate matrix. TODO: check if this line is necessary

for i in range(1,10):
    y_observed[i-1,:] = temperature_mean[1,i*15-15:i*15]   #get corresponding crosswind slice temperatures

y_observed_mean = np.mean(y_observed, 0) - np.min(np.mean(y_observed,0))     #subtract offset, improves fit

#get distribution of temperatures from samples
#bins = np.linspace(15, 30, 100) #histogram bins
#h,b = np.historgram(temperature_adjusted, bins)
#centers = (b[:-1]+b[1,:])/2
#
#h2 = np.float(np.sum(h,0))  
#dist = h/h2   #convert histogram to probability

def gaus(x, *p):
    """"gaussian function, for fitting to distribution
    """
    A,mu,sigma = p
    return A * np.exp(-(x-mu) ** 2 / (2. * sigma ** 2))
    

def gaus2(x,*p):
    """add two gaussians
    """
    A1,mu1,sigma1,A2,mu2,sigma2 = p
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2))+A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))
    
def gaus2_better(x,p1,p2):
    return gaus(x,p1) + gaus(x,p2)
    
#fit gaussian to distribution
position_guess = [1,90,15]   #start guess for fitting
coeff, cov = curve_fit(gaus, x_observed, y_observed_mean, position_guess = position_guess) #inputs can be: gaus,centers, dist,position_guess, if using temperature distribution data
hist_fit = gaus(x_observed,*coeff)

y_observed_adjusted = y_observed_mean - hist_fit    #subtract out first gaussian, to fit second (if necessary)

#fit second gaussian, if necessary 
position_guess = [0.12,40,5]
coeff2, cov2 = curve_fit(gaus,x_observed, np.abs(y_observed_adjusted), position_guess = position_guess)   #not sure how I feel about abs val..
hist_fit2 = gaus(x_observed,*coeff2)

#create final coefficients and fits
coeff = np.concatenate(coeff,coeff2, axis=0)
y_fit = hist_fit + hist_fit2

#plot to check fit
#TODO: move to end; plot after calculations are out of the way
plt.plot(x_observed,y_observed_mean,'ro',label='Test data'), plt.plot(x_observed,hist_fit,label='Fitted data')

#prediction locations
#x_predicted = np.atleast_2d(np.random.rand(100))*coeff(1)   #random data, around mean
x_predicted = np.atleast_2d(np.linspace(0,254,50))       #1 mm prediction sites
x_observed = np.atleast_2d(x_observed)    #make 2d for gaussian process fit. TODO: figure out what atleast_2d does

y_observed = np.atleast_2d(y_observed)    #make 2d for gaussian process fit
y_observed_adjusted = np.atleast_2d(y_observed_adjusted)
y_fit = np.atleast_2d(y_fit)
   
##TODO: make section into separate function
   
#TODO: set up gaussian process function, with MSE start point (minimum and maximum can be added)
gp = gaussian_process.GaussianProcess(corr = 'absolute_exponential',theta0=1./25, thetaL=None,thetaU = None)
                                      #thetaL=none,
                                      #thetaU=none)
                                     # random_start = 100)
gp.fit(x_observed.T, y_fit.T)

#y = gaus(x_observed,*coeff)     #single gaussian expected y values
y_expected_interpolated = gaus2(x_observed,*coeff)     #double gaussian expected y values
y_predictions, y_predictions_MeanSqErr = gp.predict(x_predicted.T,eval_MSE=True)   #produce predicted y values

