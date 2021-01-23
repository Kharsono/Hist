import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=300, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=0.68, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = args.num_iters
threshold = args.threshold
template_threshold = 5

seq = np.load("../data/carseq.npy")
rect = np.asarray([59, 116, 145, 151], dtype=float)
ori = np.asarray([59, 116, 145, 151], dtype=float)
stay = np.asarray([59, 116, 145, 151], dtype=float)

from LucasKanade import LucasKanade

qn = np.zeros(2)
pt = np.zeros(2)
It = seq[:,:,0]
comp = np.empty((seq.shape[2] - 1, 4), dtype=float)

for i in range(seq.shape[2] - 1):        
    print("frame ", i)
    
    It1 = seq[:,:,i+1]
    p = LucasKanade(It, It1, rect, threshold, num_iters, pt) # Calculate movements between curr (template) & next frame
    
    temp = rect[:2] + p - stay[:2] # Find the difference from the original frame but reference to rect
    qn = LucasKanade(seq[:,:,0], It1, stay, threshold, num_iters, temp[:2]) # Find p star
    qn = qn + stay[:2]- rect[:2] # Use the same reference of rect & stay
    delp = qn - p; check = (delp[0]**2 + delp[1]**2)
    print("check ", check)
    if (check < template_threshold):
        print("Act conservatively")
        rect[0] += qn[0]; rect[1] += qn[1]; rect[2] += qn[0]; rect[3] += qn[1]
        It = np.copy(It1)
        pt = np.copy(p)
        
    else:
        print("Else")
        rect[0] += p[0]; rect[1] += p[1]; rect[2] += p[0]; rect[3] += p[1]
        pt = np.copy(p)

    rect2 = np.copy(rect)

    Et = seq[:,:,i]
    Et1 = seq[:,:,i+1]
    real_p = LucasKanade(Et, Et1, ori, threshold, num_iters)
    ori[0] += real_p[0]; ori[1] += real_p[1]; ori[2] += real_p[0]; ori[3] += real_p[1]
    
    # Plotting
    if i==0 or i==99 or i==199 or i==299 or i==399:
        fig, axs = plt.subplots(1)
        axs.imshow(seq[:,:,i])
        red_rect = patches.Rectangle((rect2[0],rect2[1]), 
                                     rect2[2] - rect2[0], rect2[3] - rect2[1], 
                                     linewidth = 1, edgecolor = 'r', facecolor = 'none')
        blue_rect = patches.Rectangle((ori[0],ori[1]), 
                                      ori[2] - ori[0], ori[3] - ori[1], 
                                      linewidth = 1, edgecolor = 'b', facecolor = 'none')
        axs.add_patch(blue_rect)
        axs.add_patch(red_rect)
        plt.show()
        
    comp[i,:] = rect[:]
        
    if i == (seq.shape[2]-2):
        print(comp.shape)
        print(comp)
        np.save('carseqrects-wcrt', comp)
    
    