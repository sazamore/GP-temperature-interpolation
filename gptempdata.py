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

adjusted_temp = data['store2']    #pull out tempearture data 

data = io.loadmat('C:\Users\Sharri\Documents\MATLAB\Temp stuff\lhdata.mat')

pos_mm = data['p_mm']   #pull out positional data
mean_T = data['s']      #time averaged temperature data

x_obs = pos_mm[:15,0]           #x (crosswind) axis, observed data

y_obs = np.zeros([9,15])        #preallocate matrix

for i in range(1,10):
    y_obs[i-1,:] = mean_T[1,i*15-15:i*15]   #get corresponding crosswind slice temperatures

mean_y_obs = np.mean(y_obs,0)-np.min(np.mean(y_obs,0))     #subtract offset, improves fit

#get distribution of temperatures from samples
#bins = np.linspace(15, 30, 100) #histogram bins
#h,b = np.historgram(adjusted_temp, bins)
#centers = (b[:-1]+b[1,:])/2
#
#h2 = np.float(np.sum(h,0))  
#dist = h/h2   #convert histogram to probability

def gaus(x, *p):
    """"gaussian function, for fitting to distribution
    """
    A,mu,sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    

def gaus2(x,*p):
    """add two gaussians
    """
    A1,mu1,sigma1,A2,mu2,sigma2 = p
    return A1*np.exp(-(x-mu1)**2/(2.*sigma1**2))+A2*np.exp(-(x-mu2)**2/(2.*sigma2**2))
    
def gaus2_better(x,p1,p2):
    return gaus(x,p1) + gaus(x,p2)
    
#fit gaussian to distribution
p0 = [1,90,15]   #start guess for fitting
coeff, cov = curve_fit(gaus, x_obs,mean_y_obs, p0=p0) #inputs can be: gaus,centers, dist,p0, if using temperature distribution data
hist_fit = gaus(x_obs,*coeff)

y_obs_adjusted = mean_y_obs-hist_fit    #subtract out first gaussian, to fit second (if necessary)

#fit second gaussian, if necessary 
p0 = [0.12,40,5]
coeff2,cov2 = curve_fit(gaus,x_obs, np.abs(y_obs_adjusted),p0=p0)   #not sure how I feel about abs val..
hist_fit2 = gaus(x_obs,*coeff2)

#create final coefficients and fits
coeff = np.concatenate(coeff,coeff2,axis=0)
y_fit = hist_fit + hist_fit2

#plot to check fit
plt.plot(x_obs,mean_y_obs,'ro',label='Test data'), plt.plot(x_obs,hist_fit,label='Fitted data')

#prediction locations
#X_pred = np.atleast_2d(np.random.rand(100))*coeff(1)   #random data, around mean
X_pred = np.atleast_2d(np.linspace(0,254,50))       #1 mm prediction sites
x_obs = np.atleast_2d(x_obs)    #make 2d for gaussian process fit
y_obs = np.atleast_2d(y_obs)    #make 2d for gaussian process fit
y_obs_adjusted = np.atleast_2d(y_obs_adjusted)
y_fit = np.atleast_2d(y_fit)
   
##TODO: make section into separate function
   
#TODO: set up gaussian process function, with MSE start point (minimum and maximum can be added)
gp = gaussian_process.GaussianProcess(corr = 'absolute_exponential',theta0=1./25, thetaL=None,thetaU = None)
                                      #thetaL=none,
                                      #thetaU=none)
                                     # random_start = 100)
gp.fit(x_obs.T,y_fit.T)

#y = gaus(x_obs,*coeff)     #single gaussian expected y values
y = gaus2(x_obs,*coeff)     #double gaussian expected y values
y_pred, MSE = gp.predict(X_pred.T,eval_MSE=True)   #produce predicted y values

