import numpy as np
from LucasKanadeAffine import *
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import cv2
import time
from InverseCompositionAffine import*


def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """

    # put your implementation here
    mask = np.zeros(image1.shape, dtype=bool)
    print("Substract Dom Called")
    
    start = time.time()
    
    # Find M
    #   Regular Affine - comment out one segment (reg or inv)
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    # #   Inverse Affine
    # M = InverseCompositionAffine(image1, image2, threshold, num_iters) 
    
    M = M[:-1,:]
    end = time.time()
    titot = end - start
    print("Time for LKAff: ", titot)
    
    # Warp image using M
    img_w = cv2.warpAffine(image2, M, (image1.shape[1], image1.shape[0]))
    
    # Adjust image for operations
    img_w = binary_erosion(img_w)
    img_w = binary_dilation(img_w)
    diff = image1 - img_w
    
    # Build binary mask
    diff = np.abs(diff)
    mask = (diff > tolerance)
   
    return mask
