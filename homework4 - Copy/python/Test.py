"""
Check the dimensions of function arguments
This is *not* a correctness check
Written by Chen Kong, 2018.
"""
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import helper
import findM2

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
K = np.load('../data/intrinsics.npz')



N = data['pts1'].shape[0]
print(N)
M = 640

# # 2.1
F = sub.eightpoint(data['pts1'], data['pts2'], M)
# assert F8.shape == (3, 3), 'eightpoint returns 3x3 matrix'
# print("F is : ", F8)
# # np.savez('../results/q2_1.npz', F = F8, M = M)

# # # 2.1 Write Up
# # # file = np.load('../python/q2_1test.npz')
# # # print("Uploaded F is : ", file['F'])
# # # helper.displayEpipolarF(im1, im2, F8)


# 3.1
# Write up 3.1
E = sub.essentialMatrix(F, K['K1'], K['K2'])
print("E is : ", E)
K1 = K['K1']
K2 =  K['K2']

# 3.2
# print("3.2")
# M1 = np.array([[1,0,0,0],
#                 [0,1,0,0],
#                 [0,0,1,0]])
# M2s = helper.camera2(E)
# print("M shapes: ", M1.shape, M2s.shape)

# C1 = np.dot(K['K1'], M1)
# C2 = []
# for j in range(4):
#     sec = np.dot(K['K2'], M2s[:,:,j])
#     C2.append(sec)
# print(C1, C2[0])
# print("C1 & C2: ", C1.shape, C2[0].shape)
# P, err = sub.triangulate(C1, data['pts1'], C2[0], data['pts2'])
# assert P.shape == (N, 3), 'triangulate returns Nx3 matrix P'
# assert np.isscalar(err), 'triangulate returns scalar err'

# 3.3
# M2, C2r, P = findM2.find(M2s, data['pts1'], data['pts2'], C1, K['K2'])
#print("M2, C2r, P are ", M2, C2r, P)
# np.savez('../results/q3_3.npz', M2 = M2, C2 = C2r, P = P)

# 4.1
# temp1 = np.array([[64,131],[122,206],[446,226],[517,228],[480,91],[223,214],[299,279],[356,155],[91,246]])
# temp2 = np.array([[64,118],[122,176],[444,197],[515,198],[471,91],[222,191],[300,274],[352,159],[90,237]])
# print("4.1 finally ", F8, temp1, temp2)

# helper.epipolarMatchGUI(im1, im2, F8)
# np.savez('../results/q4_1.npz', F = F8, pts1 = temp1, pts2 = temp2)
# print("done")

# # 5.1
data2 = np.load('../data/some_corresp_noisy.npz')
# Fn = sub.eightpoint(data2['pts1'], data2['pts2'], M)
# helper.displayEpipolarF(im1, im2, Fn)
# a
# F, binary_inl = sub.ransacF(data2['pts1'], data2['pts2'], M)
# helper.displayEpipolarF(im1, im2, F)

# a
# # Q5.2
# r = np.ones([3,1])
# R = sub.rodrigues(r)

F, inlier_1, inlier_2 = sub.ransacF(data2['pts1'], data2['pts2'], M)
# r = sub.invRodrigues(R)
# print(r.shape, r)
# a
M1 = np.array([[1, 0, 0,0], [0, 1, 0,0],[0,0,1,0]])
M2s = helper.camera2(E)
C1 = np.matmul(K1, M1)
C2s = []
for j in range(4):
	temp = np.matmul(K2, M2s[:,:,j])
	C2s.append(temp)
E = sub.essentialMatrix(F, K1, K2)


# Good reprojection error and plot
M2, C2, P = findM2.find(M2s, C1, inlier_1, inlier_2, K2)
P, err = sub.triangulate(C1, inlier_1, C2, inlier_2)
print ('Error after bundle adjustment = ', err)
# plot P


# Bad reprojection error and plot
M2, P = sub.bundleAdjustment(K1, M1, inlier_1, K2, M2, inlier_2, P)
C2 = np.matmul(K2, M2)
P, err = sub.triangulate(C1, inlier_1, C2, inlier_2)
print ("error after bundle adjustment = ", err)
# plot


# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

x = P[:,0]
y = P[:,1]
z = P[:,2]
plt.gca().set_aspect('equal',adjustable = 'box')
ax.scatter(x,y,z, color='blue')

for i in range(P.shape[0]):
	ax.scatter(P[i][0], P[i][1], P[i][2], c='r', marker='o')

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')

plt.show()
plt.close()
