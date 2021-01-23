# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# Dec 2020
# ##################################################################### #

# Imports
import numpy as np
from skimage.io import imread
from skimage.color import rgb2xyz
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from utils import integrateFrankot

import scipy.sparse.linalg as sp

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centered on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the sphere in an array of size (3,)

    rad : float
        The radius of the sphere

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the sphere
    """

    image = None
    
    image = np.zeros((res[0], res[1]))
    rad_pix = rad / pxSize
    rad_pix = int(rad_pix)
    print(rad_pix, res)
    cx = int(res[0]/2); cy = int(res[1]/2)
    
    # Normal Sphere
    #x^2 + y^2 + z^2 = 0.75^2 #sphere equation
    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi, 100)
    
    
    # Surface Normal
    cx=0; cy=0; cz=0 # Center
    for i in range(res[0]): #x
        for j in range(res[1]): #y
            nx=i-cx; ny=j-cy; nz = 0
            #z = np.sqrt(0.75**2 - i**2 - j**2)
            
            #eqn 1
            #a = nx + nz * (zx - z)
            #eqn 2
            #b = nx + nz * (zy - z)
            
            
    # STEPS NOT DONE: # PARTIAL CREDIT
        # 1. draw pixel array of circle. 0's for non circle and 1's for circle
        # 2. For each pixel in the circle, compute the n-dot-l for intensity
        # 3. Plot the image
    
    
    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """

    I = None
    L = None
    s = None 
    images = None
    
    for i in range(7):
        j = i+1
        temp = imread(path + 'input_' + str(j) + '.tif')
        temp = rgb2xyz(temp)
        fors = np.copy(temp)
        temp = temp[:,:,1] #Just take luminance (Y)
        ipyn = np.copy(temp)
        print(ipyn.shape)
        temp = np.reshape(temp, (temp.shape[0]*temp.shape[1]))
        
        
        if i == 0:
            I = np.copy(temp)
            images = np.copy(ipyn)
        else:
            I = np.vstack((I, temp))
            images = np.vstack((images, ipyn))
    
    sources = np.load(path + 'sources.npy')
    L = np.copy(sources)
    L = L.T
    
    # s = (431, 369, 3)
    s = (fors.shape[0], fors.shape[1])
    
    print(L.shape, temp.shape, I.shape, s)
    
    return I, L, s, images


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = None

    
    Linv = np.linalg.pinv(L.T)
    print("THE Y = AX", I, Linv)
    I = I[:,:]
    B = np.dot(np.linalg.pinv(L.T), I)


    # A matrix & Y vector initialization
    # y = np.copy(I)
    # y = y.flatten()
    # A = np.zeros((y.shape[0], y.shape[0]))
    # # for i in range(7):
    # #     A[i, i] = L[0, i]
    # #     A[i, i+1] = L[1, i]
    # #     A[i, i+2] = L[2, i]
    # B = sp.lsqr(A, y)
    

    
    print("B HERE", B.shape)
    
    return B


def estimateAlbedosNormals(B): #3xP

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = None # will be 1xP
    normals = None # will still be 3xP
    
    # print("B IS", B.shape, B)
    albedos = np.zeros((B.shape[1]))
    normals = np.copy(B)
    
    albedos = np.linalg.norm(B, axis=0)
    
    for i in range(B.shape[1]):
        for j in range(B.shape[0]):
            normals[j, i] = normals[j, i] / albedos[i]
            
            
    # print(albedos.shape, normals.shape)
    
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = None
    normalIm = None
    
    albedoIm = np.copy(albedos)
    albedoIm = np.reshape(albedoIm, (s[0], s[1]))
    
    normalIm = np.copy(normals)
    normalIm = normalIm.T
    normalIm = np.reshape(normalIm, (s[0], s[1], 3))
    
    
    # print(albedoIm)
    # print("F shapes", albedoIm.shape, normalIm.shape)
    
    plt.subplot(1,2,1)
    plt.imshow(albedoIm, cmap = 'gray')
    plt.subplot(1,2,2)
    plt.imshow(normalIm, cmap = 'rainbow')
    plt.show()
    
    return albedoIm, normalIm


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    surface = None
    surface = np.zeros(normals[0].shape)
    
    zx = normals[0,:]/normals[2,:]
    zy = normals[1,:]/normals[2,:]
    zx = np.reshape(zx, (s)); zy = np.reshape(zy, (s))
    # print("GRADIENTS ", zx, zy)
    # print(zx.shape, zy.shape, normals.shape)
    pad = 512
    
    surface = integrateFrankot(zx, zy, pad)
    # print(surface.shape)
    
    return surface


def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    
    z = surface
    m, n = z.shape
    x, y = np.mgrid[0:m, 0:n]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot_surface(x, y, z, cmap='coolwarm')
    
    # rotate the axes and update
    # ax.view_init(60, 0)
    # plt.draw()
    # plt.pause(5)
    
    plt.show()
    
    pass


if __name__ == '__main__':

    # Put your main code here
    
    # a
    center = np.array([0,0,0])
    rad = 0.75 / 2 # in cm
    light1 = np.array([1,1,1] / np.sqrt(3))
    light2 = np.array([1,-1,1] / np.sqrt(3))
    light3 = np.array([-1,-1,1] / np.sqrt(3))
    light = np.array([light1, light2, light3])
    print(light.shape)
    pxSize = 0.0007 # in cm
    res = np.array([3840, 2160])
    
    sphere = renderNDotLSphere(center, rad, light, pxSize, res)
    
    
    # c
    path = '../data/'
    I, L, s, images = loadData(path)

    
    # d
    _, sing, _ = np.linalg.svd(I, full_matrices=False)
    r = np.linalg.matrix_rank(I)
    print("Rank of the singular values : ", sing, r)
    
    # e
    B = estimatePseudonormalsCalibrated(I, L)
    albedos, normals = estimateAlbedosNormals(B)
    
    # f
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    
    # i
    surface = estimateShape(normals, s)
    plotSurface(surface)
    
    print("done.")
    
    
    pass



















