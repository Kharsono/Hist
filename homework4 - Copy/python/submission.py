"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here
import numpy as np
import helper
import util
import scipy.ndimage.filters as filt
from scipy.stats import skew
import cv2 as cv
import scipy.optimize

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    
    # Normalization
    pts1 = pts1 / M
    pts2 = pts2 / M
    
    # U Matrix
    N = pts1.shape[0]
    U = np.zeros((N, 9))
    for i in range(N):
       U[i,0] = pts2[i][0]*pts1[i][0]
       U[i,1] = pts2[i][0]*pts1[i][1]
       U[i,2] = pts2[i][0]
       U[i,3] = pts2[i][1]*pts1[i][0]
       U[i,4] = pts2[i][1]*pts1[i][1]
       U[i,5] = pts2[i][1]
       U[i,6] = pts1[i][0]
       U[i,7] = pts1[i][1]
       U[i,8] = 1
    
    # SVD for evector for smallest evalue
    u, s, vhT = np.linalg.svd(U)
    vh = np.transpose(vhT)
    evect = vh[:, -1]
    # print(vh.shape, vhT.shape, evect)
    
    # F Matrix
    F = evect.reshape(3,3)
    #print(np.linalg.matrix_rank(F), F.shape)
    #F[2,2] = 0
    F = util._singularize(F)
    F = util.refineF(F, pts1, pts2)
    #print(np.linalg.matrix_rank(F), F.shape)
    
    # Unnormalizing
    T = np.array([[1/M,0,0],[0,1/M,0],[0,0,1]])
    TT = np.transpose(T)
    F = np.dot(np.dot(TT, F), T)
    #print(F, M)
    
    # Saving F & M
    #np.savez('q2_1', F=F, M=M)
    print("F Shape is: ", F.shape)
    
    return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    
    K2T = np.transpose(K2)
    E = np.dot(np.dot(K2T, F), K1)
    
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    
    print("TRIANGULATE pts", pts1.shape, pts2.shape)
    print("TRI Cs", C1.shape, C2.shape)
    ones = np.ones([pts1.shape[0], 1])
    pts1 = np.concatenate((pts1, ones), axis = 1)
    pts2 = np.concatenate((pts2, ones), axis = 1)
    print(pts1.shape)
    W = np.zeros([len(pts1), 4])
    err = 0
   
    # Calculate 3D & its error for each point pair
    for i in range(len(pts1)):
        
        # Remove affects of intrinsic matrices
        x1 = pts1[i,0] * C1[2,:] - C1[0,:] 
        y1 = pts1[i,1] * C1[2,:] - C1[1,:]
        x2 = pts2[i,0] * C2[2,:] - C2[0,:]
        y2 = pts2[i,1] * C2[2,:] - C2[1,:]
        
        # SVD for 3D points
        A = np.array([x1, y1, x2, y2])
        u, s, vhT = np.linalg.svd(A)
        temp = vhT[-1, :]
        temp = temp/temp[3]
        # print("vh is ", temp)
        W[i, :] = temp
        # print(temp.shape)
        
        # Compute back to projections for error
        vh = np.transpose(temp)
        pts1_p = np.matmul(C1, vh); pts2_p = np.matmul(C2, vh)
        pts1_p = pts1_p/pts1_p[2]; pts2_p = pts2_p/pts2_p[2]
        # print(pts1_p.shape)
        err1 = pts1[i, :] - pts1_p; err2 = pts2[i, :] - pts2_p
        e = np.linalg.norm(err1) + np.linalg.norm(err2)
        err += e
        # print(err)

    W = W[:, 0:3] 
    
    print("W shape is: ", W.shape)
    
    return W, err
    


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''

def epipolarCorrespondence(im1, im2, F, x1, y1):

    # Mask around x1 and  y1
    max_error = np.inf
    inp = 10
    comp = inp * 2
    sig = 5

    epts = np.array([[x1],[y1],[1]], float)
    ep_line = np.dot(F, epts)
    # Line --> ax + by + c = 0
    a = ep_line[0]
    b = ep_line[1]
    c = ep_line[2]

    # TEdge Case of image 1
    if (y1 < 474 and y1 > 6):
        mask1 = im1[int(y1-inp) : int(y1+inp), int(x1-inp) : int(x1+inp)]
    
    arr = np.zeros((comp, comp))
    arr[comp//2, comp//2] = 1
    gauss = filt.gaussian_filter(arr, sig)
    gaussd = np.dstack((gauss, gauss, gauss))

    error_list = []
    
    # Gaussian weighting --> Assuming object has not moved much between 2 images
    for y2 in range(y1-30,y1+30): 
        
        x2 = int((-c - b*y2)/a) #from line eqn

        if (x2-inp > 0 and (x2+inp < im2.shape[1])) and (y2-inp > 0 and (y2+inp <= im2.shape[0])):
            mask2 = im2[int(y2-inp) : int(y2+inp), int(x2-inp) : int(x2+inp)]

            dist = mask1 - mask2
            distw = np.multiply(gaussd, dist)
            
            err = np.linalg.norm(distw)
            error_list.append(err)
            
            # Finding the min error and updating it
            if err < max_error:
                max_error = err
                x2f = x2
                y2f = y2

    return x2f, y2f

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    
    nIters = 500
    tol = 0.001
    print("nIters & tol are ", nIters, tol)
    max_in= 0
    N = pts1.shape[0]
    print("N or sets of points: ", N)
    
    ones = np.ones([N,1])
    pts1_3=  np.concatenate((pts1, ones), axis=1)
    pts2_3=  np.concatenate((pts2, ones), axis=1)

    for k in range(nIters):
        print(k)
        
        # Find random indices (8 total)
        pts1test= []
        pts2test= []
        pt_i= [np.random.randint(0, N-1) for p in range(8)]
        # print(pt_i)
        for i in range(8):
            pts1test.append(pts1[pt_i[i],:])
            pts2test.append(pts2[pt_i[i],:])
        # print("Pts1 :", pts1test)
        pts1trial = np.vstack(pts1test)
        pts2trial = np.vstack(pts2test)
        # print("Pts 1 m : ", pts1trial)
        
        # Compute for F8 from the 8 points
        F8 = eightpoint(pts1trial, pts2trial, M)
        
        # Check error & inliers
        ptinlier = []
        inliers= 0
        for j in range(pts1_3.shape[0]):
            error_x= np.abs(np.matmul(np.matmul(np.transpose(pts2_3[j]), F8), pts1_3[j])) 
            if (error_x < tol):
                inliers = inliers + 1
                ptinlier.append([1])
            else:
                ptinlier.append([0])
        if(inliers > max_in):
            max_in = inliers
            final = F8
                
    print (max_in, ptinlier)
    return final, ptinlier

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    
    print(cv.Rodrigues(r)[0]) # To scheck if the function is working correctly
    theta = np.linalg.norm(r)
    r = r / theta

    K = np.array([[0, -r[2], r[1]],
                  [r[2], 0, -r[0]],
                  [-r[1], r[0], 0]])

    rot = np.eye(3,3) + (1 - np.cos(theta)) * (np.dot(K,K)) + np.sin(theta) * K
    rot = np.ndarray.flatten(rot)
    R = np.zeros((3,3))
    print(R)
    for i in range(3):
        for j in range(3):
            R[i, j] = rot[(i*3)+j]

    print("Rot now", R)
    return R

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    
    A = (R - R.T)/2
    rs = np.asarray([A[2,1], A[0,2], A[1,0]]).T
    
    s = np.linalg.norm(rs, 2)
    eu = (R[0,0] + R[1,1] + R[2,2] - 1)/2 #Euclidean
    theta = np.arctan2(s, eu) #Rotation angle
    
    
    arr = np.array([(R[2,1] - R[1,2]), (R[0,2] - R[2,0]), (R[1,0] - R[0,1])])
    w = arr / (2 * np.sin(theta))
    r = w * theta
    
    return r


'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    
    
    # Extracting from flattening
    pc = x[:-6]
    P = pc.reshape(p1.shape[0], 3)
    t = x[-3:].reshape(3,1)
    r = x[-6:-3]
    
    # Find rotational
    R = rodrigues(r)
    M2 = np.hstack(R, t)

    C1 = np.matmul(K1, M1)
    C2 = np.matmul(K2, M2)

    pt1_p = np.matmul(C1, P.T)
    p1_hat = pt1_p/pt1_p[2]
    pt2_p = np.matmul(C2, P.T) 
    p2_hat = pt2_p/pt2_p[2] #nX2

    residuals = np.concatenate([(p1-p1_hat).reshape([-1]) , (p2-p2_hat).reshape([-1])])

    return residuals

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Breakdown M2
    R = M2_init[:,0:3]
    t = M2_init[:,3:]
    
    # Find the 3d vect
    r = invRodrigues(R)
    x = P_init.flatten()
    x = np.hstack((x, r.flatten()))
    x = np.hstack((x, t.flatten()))
    # print (x.shape)

    func = lambda x: ((rodriguesResidual(K1, M1,p1, K2, p2, x))**2).sum()

    z = scipy.optimize.minimize(func, x)
    xnew = z.x

    t = x[-3:].reshape(3,1)
    r = x[-6:-3]
    R = rodrigues(r)
    M2 = np.hstack(R, t)

    C1 = np.matmul(K1, M1)
    C2 = np.matmul(K2, M2)
    
    w, err = triangulate(C1, p1, C2, p2)
    
    return M2, w
