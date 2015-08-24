# -*- coding: utf-8 -*-
"""
This script opens data files and extracts relevant data. Then, using a sklearn 
gaussian process package, fits a gaussian to the crosswind (1d) temperature
measurements and interpolates temparture values (relative, not absolute) at a
2 mm interval.

Created on Mon Dec 01 11:51:44 2014

@authors: Sharri 
"""

from scipy.optimize import curve_fit
from sklearn import gaussian_process#,grid_search,svm
import numpy as np
import matplotlib.pyplot as plt

#SECTION 1 (not sure if this should include these imports)
import scipy.io as io
import glob

allfiles = glob.glob(r'C:\Users\Sharri\Dropbox\Le grand dossier du Sharri\Data\Temperature Data\python temp data\*csv')

datafiles = []
for i in range(len(allfiles)):
   datafiles.append(np.genfromtxt(allfiles[i],delimiter = ','))
   
   
#load data
xyz_observed = datafiles[5]#np.genfromtxt('lh_temp_data_complete.csv', delimiter = ' ')
T_raw = datafiles[7] #np.genfromtxt('lh50temp_raw_data.csv',delimiter = ' ')
T_sd = np.nanstd(T_raw,axis=0)

#move origin to corner, this may be causing praaaablems.
xyz_observed[:,1] = xyz_observed[:,1]+0.127; 

if np.any(np.isnan(xyz_observed[:,3])):
    NaN_locations = np.where(np.isnan(xyz_observed[:,3])==1)      #find the indices of Nans
   # T_time_avg_3d = np.delete(T_time_avg_3d, NaN_locations)
    xyz_observed = np.delete(xyz_observed, NaN_locations, axis = 0)
    T_raw = np.delete(T_raw, NaN_locations,axis = 1)
    T_sd = np.delete(T_sd, NaN_locations)

#SECTION 3
#TODO: move everything below this to a function and/or separate script

#calculate noise (required)
nugget = T_sd/xyz_observed[:,3]

sweep = xyz_observed;#data to train gridsearch
check = np.arange(100,200)
sweep = np.delete(sweep, check, axis = 0)   #remove some points to test values
nugget = np.delete(nugget, check)
 
 #TODO: make into function   
gp = gaussian_process.GaussianProcess(regr = 'linear',
                                      corr = 'squared_exponential',
                                      theta0 = 4e-2,
                                      thetaL = 1e-4,    #best = 1e-6
                                      thetaU = 1,    #best = 1
                                      normalize = True,
                                      nugget = nugget)
#when height = 1, thetaL = .1, thetaU = .3
#when height = 0, thetaL = 10e-2,thetaU = .3

#
#param_grid = [{'thetaL':[1e-10 1e-8 1e-]
#
#gp = grid_search.ParameterGrid()

 
#gp.fit(xyz_observed[:,:3], xyz_observed[:,3])
gp.fit(sweep[:,:3],sweep[:,3])

#make prediction location grid
#x_predict = np.atleast_2d(np.linspace(.20, .900, 20)) #(-0.127, 0.127, 25))       #2 mm prediction sites
#y_predict = np.atleast_2d(np.linspace(0., 0.254, 25))
#z_predict = np.atleast_2d(np.linspace(.1, .240, 25))

x_predict = np.atleast_2d(xyz_observed[check,0])
y_predict = np.atleast_2d(xyz_observed[check,1])
z_predict = np.atleast_2d(xyz_observed[check,2])

#x1,x2,x3 = np.meshgrid(x_predict, y_predict, z_predict)  #for proper ordering use z-y-x
#xyz_predict = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size), x3.reshape(x3.size)]).T


T_prediction, y_prediction_MSE = gp.predict(xyz_observed[check, :3], eval_MSE = True)   #produce predicted y values
sigma = np.sqrt(y_prediction_MSE)   #get SD of fit at each x_predicted location (for confidence interval)

plt.figure, plt.plot(xyz_observed[check,3], T_prediction,'.')
plt.plot(np.arange(19,25),np.arange(19,25),'k')
plt.xlabel('Observed Temp')
plt.ylabel('Predicted Temp')
#predicted_draw = np.random.normal(T_prediction, sigma)
