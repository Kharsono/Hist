import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from SubtractDominantMotion import *

parser = argparse.ArgumentParser()
# parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
# parser.add_argument('--threshold', type=float, default=0.5, help='dp threshold of Lucas-Kanade for terminating optimization')
# parser.add_argument('--tolerance', type=float, default=0.9, help='binary threshold of intensity difference when computing the mask')
# FOR INVERSE
parser.add_argument('--num_iters', type=int, default=100, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=0.80, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.73, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')
print(seq.shape[2])

for i in range(seq.shape[2] - 1):
    print(i)
    
    img1 = seq[:,:,i]; img2 = seq[:,:,i+1]
    mask = SubtractDominantMotion(img1, img2, threshold, num_iters, tolerance)
    color_mask = np.ma.masked_where(mask == False, mask)
    
    if (i == 1 or i == 30 or i == 60 or i == 90 or i == 120):   
        print("here")
        plt.figure()
        plt.imshow(img1, cmap = 'gray')
        plt.imshow(color_mask, cmap = 'spring')
        plt.show()