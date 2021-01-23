import numpy as np
import cv2
#Import necessary functions
from planarH import computeH_ransac
from planarH import compositeH
from matchPics import matchPics
import skimage.color
from opts import get_opts
from loadVid import loadVid
import matplotlib.pyplot as plt
import imageio
#import cpselect.cpselect as cpselect
import Panorama as this



#Write script for Q4.2x
def SIFT(image):
    #SIFT descriptor to output keypoints & descriptor
    descriptor = cv2.xfeatures2d.SIFT_create()
    (k, desc) = descriptor.detectAndCompute(image, None)
    
    #Convert indices to match coordinates
    k = np.float32([ind.pt for ind in k]) 
	
    return (k, desc)



opts = get_opts()
left = cv2.imread('../data/pano_left.jpg')
right = cv2.imread('../data/pano_right.jpg')
print("Image shapes are ", left.shape, right.shape)

# Features & keypoints
#KL, DescL = SIFT(left)
#KR, DescR = SIFT(right)
matches, locs1, locs2 = matchPics(left, right, opts)
KL = locs1[matches[:,0]]; KR = locs2[matches[:,1]]
print("K shape", KL.shape, KR.shape)

# Descriptor matching --> No need
#dmatch = cv2.DescriptorMatcher_create("BruteForce") #bruteforce matches each one
#mat = dmatch.knnMatch(DescL, DescR, 2) #top 2 match of feature vectors in descriptors
#match = np.asarray(mat)
#print("Match shape", match.shape)

# Homography
H, mas = cv2.findHomography(KL, KR, cv2.RANSAC, opts.inlier_tol)

# Stitching
stit = cv2.warpPerspective(right, H, (left.shape[1] + right.shape[1], left.shape[0])) #combination width & same height
stit[0:right.shape[0], 0:right.shape[1]] = left

cv2.imshow('Panorama', stit)
cv2.waitKey(0)