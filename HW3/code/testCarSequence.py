import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold


seq = np.load("../data/carseq.npy")
rect = np.asarray([59, 116, 145, 151], dtype=float)
print(seq.shape[2])
comp = np.empty((seq.shape[2]-1, 4), dtype=float)

from LucasKanade import LucasKanade

for i in range(seq.shape[2] - 1):
    It = seq[:,:,i]
    It1 = seq[:,:,i+1]
    p = LucasKanade(It, It1, rect, threshold, num_iters)

    rect[0] += p[0]; rect[1] += p[1]; rect[2] += p[0]; rect[3] += p[1]


    # Plotting
    if i==0 or i==99 or i==199 or i==299 or i==399: 
        fig, axs = plt.subplots(1)
        axs.imshow(seq[:,:,i])
        red_rect = patches.Rectangle((rect[0],rect[1]), 
                                     rect[2] - rect[0], rect[3] - rect[1], 
                                     linewidth = 1, edgecolor = 'r', facecolor = 'none')
        axs.add_patch(red_rect)
        plt.show()
        
    comp[i,:] = rect[:]
        
    if i == (seq.shape[2]-2):
        print(comp.shape)
        print(comp)
        np.save('carseqrects', comp)