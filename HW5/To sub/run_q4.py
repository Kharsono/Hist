import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()
    
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    
    # Compare vertical positions of the centers of images
    center = []
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        cent = ((minr + maxr) // 2, (minc + maxc) // 2)
        center.append((cent, bbox))
    center = sorted(center, key = lambda x : x[0]) #sort per row
    
    # find rows + remove unaligned images
    rows = []
    for point in center:
        isFind = False
        cent, bbox = point #for each point
        for row in rows:
            avg_h = 0
            avg_cent = 0
            for i in row:
                avg_h += (i[1][2] - i[1][0])
                avg_cent += (i[0][0])
            avg_h = avg_h / float(len(row)) #find average character height
            avg_cent = avg_cent / float(len(row)) #find average center pos
            if avg_cent - avg_h < cent[0] < avg_cent + avg_h:
                row.append(point)
                isFind = True
                break
        if not isFind:
            rows.append([point])
    for j in range(len(rows)):
        rows[j] = sorted(rows[j], key = lambda x : x[0][1])
    
    ##########################


    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    
    dataset = []
    for row in rows:
        datarow = []
        for point in row:
            center, bbox = point
            minr, minc, maxr, maxc = bbox
            img = bw[minr : maxr + 1, minc : maxc + 1]
            
            #Square crop
            H, W = img.shape
            if H > W: #image is taller than wider
                right = (H - W) // 2 #pad the sides
                left = H - W - right
                img = np.pad(img, ((H//40, H//40), (left+H//40, right+H//40)), "constant", constant_values = (1,1))
            else: #image is wider than taller
                top = (W - H) // 2
                bot = W - H - top
                img = np.pad(img, ((top+W//40, bot+W//40), (W//40, W//40)), "constant", constant_values = (1,1))
    
            img = skimage.transform.resize(img, (32,32))
            datarow.append(np.transpose(img).flatten())
        dataset.append(np.array(datarow))
        
    ##########################
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    
    ##########################
    for data in dataset:
        sig = forward(data, params, "layer1")
        pred = forward(sig, params, "output", softmax)
        predy = np.argmax(pred, axis=1)
        text = ""
        for y in predy:
            text += letters[y]
        print(text)
        

    ##########################
    