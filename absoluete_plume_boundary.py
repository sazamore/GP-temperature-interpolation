# -*- coding: utf-8 -*-
"""
Finds the parts of the trajectory that is within a plume, given some 
predetermined plume threshold. 

Created on Wed Apr 01 17:40:33 2015

@author: Sharri
"""

import scipy.io as io
#import matplotlib.pyplot as plt

data = io.loadmat('C:/Users/Sharri/Dropbox/Le grand dossier du Sharri/Data/Tracking Data/latestdata.mat')

trajdata = data['lhon'] 
KF = trajdata['KF']
tracks = KF[0]

plume_thresh = 20.25 #absolute temperature plume threshold, in C

traj_in_plume = []
time_in_plume = []

for i in range(1,len(tracks)):
    trajectory = tracks[i][0][0][0]
    if ~np.all(np.isnan(trajectory)):
        Traj_Temp, TT_pred_MSE = gp.predict(trajectory, eval_MSE = True)
        in_plume = np.where(Traj_Temp>=plume_thresh)
        traj_in_plume.append(in_plume[0])
        time_in_plume.append(float(len(in_plume)./len(trajectory)))
    else:
        traj_in_plume.append(0.)
        time_in_plume.append(nan)
        print('Trajectory number {} is empty'.format(i))
        
