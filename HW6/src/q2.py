# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
from matplotlib import pyplot as plt


def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """

    B = None
    L = None

    U, S, Vh = np.linalg.svd(I, full_matrices=False)
    
    print("SVD RESULTS ", U.shape, S.shape, Vh.shape)
    
    Ua = np.zeros((3, 7))
    Va = np.zeros((3, Vh.shape[1]))
    Sa = np.zeros((3,3))
    Ua = U[0:3, :]
    Va = Vh[0:3, :]
    Sa[0,0] = S[0]; Sa[1,1] = S[1]; Sa[2,2] = S[2]
    
    print("Edited", Ua.shape, Sa.shape, Va.shape)
    
    L = np.dot((Sa**0.5), Ua)
    B = Va.T
    B = B.T
    print(L.shape , B.shape)

    return B, L

if __name__ == "__main__":

    path = '../data/'
    I, L, s, images = loadData(path)    

    B, L = estimatePseudonormalsUncalibrated(I)
    # G = np.array([[1,0,0],[0,1,0],[0,0,1000]])
    # B = np.dot(np.linalg.pinv(G.T), B)
    
    B = enforceIntegrability(B, s, sig = 3)
    albedos, normals = estimateAlbedosNormals(B)
    
    
    
    # f
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    
    # i
    surface = estimateShape(normals, s)
    plotSurface(surface)

    pass
