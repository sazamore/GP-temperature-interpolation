import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(211, projection='3d')

breaks = np.linspace(19,27,10)

for i in breaks:
    bin_count = np.where(T_prediction<breaks[i])
    p = ax.scatter(xyz_predict[bin_count,0],xyz_predict[bin_count,1], xyz_predict[bin_count,2],
                   s = 20, c = np.atleast_2d(T_prediction[bin_count]).T, \ 
                   edgecolors = 'None', alpha = 0.01)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

