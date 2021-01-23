'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper
import findM2
from mpl_toolkits.mplot3d import Axes3D

# Loading and reading data
data = np.load('../data/some_corresp.npz')
sel_pts = np.load('../data/templeCoords.npz')
cam = np.load('../data/intrinsics.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

# Initialization
K1 = cam['K1']
K2 = cam['K2']
M1 = np.array([[1, 0, 0, 0], 
               [0, 1, 0, 0],
               [0, 0, 1, 0]])
M = max(im1.shape[0], im1.shape[1])

# Fundamental and Essential matrix
F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
E = sub.essentialMatrix(F8, K1, K2)

# Finding camera matrices
M2s = helper.camera2(E)
C1 = np.matmul(K1, M1)
C2s = []
for j in range(4):
	temp = np.matmul(K2, M2s[:,:,j])
	C2s.append(temp)
C2s = np.asarray(C2s)

# Finding selected points in first image
N = 288
x1 = sel_pts['x1']
y1 = sel_pts['y1']
pts1 = np.zeros([N,2])
for i in range(N):
	pts1[i, 0] = x1[i]
	pts1[i, 1] = y1[i]
	
# Finding corresponding points
N = len(x1)
x2 = []
y2 = []
for i in range(N):
	x_t, y_t = sub.epipolarCorrespondence(im1, im2, F8, int(x1[i]), int(y1[i]))
	x2.append(x_t)
	y2.append(y_t)
x2 = np.asarray(x2)
y2 = np.asarray(y2)

pts2 = np.zeros([N,2])
for i in range(N):
	pts2[i, 0] = x2[i]
	pts2[i, 1] = y2[i]

# Finding 3D points in space
M2, C2, P = findM2.find(M2s, pts1, pts2, C1, K2)
np.savez('../results/q4_2.npz', F = F8, M1 = M1, M2 = M2, C1 = C1, C2 = C2)

# Plotting the 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

plt.gca().set_aspect('auto', adjustable = 'box')
ax.scatter(P[:,0], P[:,1], P[:,2], color='green')

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

plt.show()
plt.close()
