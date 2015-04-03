# -*- coding: utf-8 -*-
"""
This script imports processed trajectory data into python. Runs them through
the GP interpolation and plots a 3d scatterplot ot trajectories with 
average probable temperature on the color axis (not normalized).

Created on Tue Jan 27 17:06:42 2015

@author: Sharri
"""

import scipy.io as io
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt


#TODO: convert lines 14 and 16 to os functions
data = io.loadmat('C:/Users/Sharri/Dropbox/Le grand dossier du Sharri/Data/Tracking Data/latestdata.mat')

store = data['lhon']   #extract the struct from the file

#For unprocessed data. You'll probably never use this.
#raw  = store['raw']     # 3d trajectory position of each file
#camxy = store['camxy']  # xy positions of trajectories in each camera
#
#a = raw[0][0]           #extracts the first trajectory from the file

#TODO: put this in a loop such all trajectories are extracted

a = store['KF']

tracks = a[0]

fig = plt.figure()
ax = Axes3D(fig)
#ax = fig.add_subplot(111, projection = '3d')

for i in range(1,len(tracks)):
    track_1 = tracks[i][0][0][0] #i agree, this is fucking ridiculous
    if ~np.any(isnan(track_1)) and len(track_1)>0 :
        T_prediction, y_prediction_MSE = gp.predict(track_1, eval_MSE = True)   
        ax.scatter(track_1[:,0],track_1[:,1],track_1[:,2],
                   c=np.atleast_2d(T_prediction.T), \
                   edgecolors = 'None', clim = (19, 22))
    
#TODO: fix caxis (clim doesn't work i think)
    
plt.axis('equal')
ax.set_xlabel('Crosswind position (mm)')
ax.set_ylabel('Upwind position (mm)')
ax.set_zlabel('Elevation (mm)')

       

# to cycle through loops, the first zero determines the trajectory number. 
#so track_2 would be: track_2 = track[1][0][0][0] because god hates you.



