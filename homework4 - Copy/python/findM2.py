'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

import numpy as np
import helper
import util
import submission as sub

def find(M2s, pts1, pts2, C1, K2):
    
    C2 = []
    
    for i in range(M2s.shape[2]):
        M2 = M2s[:,:,i]
        C2.append(np.dot(K2, M2s[:,:,i]))
        print("FIND M2", C1.shape, pts1.shape, C2[i].shape, pts2.shape)
        W, err = sub.triangulate(C1, pts1, C2[i], pts2)
        
        if (W[:,-1] >= 0.0).all(): #every last element needs to be pos
            break
        
    return M2, C2[i], W       