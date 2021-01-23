import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import skimage.color
#import tempfile
#import random
#from sklearn.cluster import KMeans


def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    # ----- TODO -----

    # Check if image is floating point
    
    # Check if image has 3 channels
    if img.ndim == 2:
        img = np.array([img, img, img])

    if img.shape[0] == 3:
        first=img.shape[0]; second=img.shape[1]; third=img.shape[2]
        img = np.reshape(img, (second,third,first))
    
    # Convert to Lab color space & initialize final array
    img = skimage.color.rgb2lab(img)
    filter_responses = np.empty((np.size(img,0), np.size(img,1), 0)) #check filled. Align with utility

    # Apply filter bank  
    for z in range(len(filter_scales)): #run all filters first before moving on to next scale
        temp = np.empty_like (img)
        for x in range(4): #run for 4 image filters
            for y in range(3): #run for 3 rgb channels (separated)
                if x == 0:
                    temp[:,:,y] = scipy.ndimage.gaussian_filter(img[:,:,y], filter_scales[z])
                if x == 1:
                    temp[:,:,y] = scipy.ndimage.gaussian_laplace(img[:,:,y], filter_scales[z])
                if x == 2:
                    temp[:,:,y] = scipy.ndimage.gaussian_filter(img[:,:,y], filter_scales[z], order=[0,1]) #gaussian x-dir
                if x == 3:
                    temp[:,:,y] = scipy.ndimage.gaussian_filter(img[:,:,y], filter_scales[z], order=[1,0]) #gaussian y-dir
            comb = np.dstack((temp[:,:,0],temp[:,:,1],temp[:,:,2])) #combining the separated 3 rgb channels
            filter_responses = np.concatenate((filter_responses, comb),2) #combining filter_responses for the whole image
    
    return filter_responses

    pass


def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary

    '''

    # ----- TODO -----
    import random

    print("COMPUTE DICT 1 IM CALLED")
    al = 25
    filter_responses = np.empty((0, args.shape[2]))
    
    ##args is array of the current image file
    print(args.shape)
    
    for i in range(al): # 25 random pixels
        h = random.randrange(0, np.size(args,0), 1); v = random.randrange(0, np.size(args,1), 1)
        filter_responses = np.append(filter_responses, [args[h,v]], axis=0)
     
    return filter_responses
    
    pass


def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----

    # Initialization
    import sklearn.cluster
    import random
    T = len(train_files)
    al = opts.alpha
    #print(train_files); print(T)

    # Find all the filter_reponses by calling compute_dict_one for every image
    for x in range(T):
        path = join(opts.data_dir, train_files[x])
        img = Image.open(path)
        img = np.array(img).astype(np.float32)/255
        img = extract_filter_responses(opts, img) # become: (x,y, 3*F*filter_scales)
        print(x)
        
        filter_responses = compute_dictionary_one_image(img)
        '''
        filter_responses = np.empty((0, args.shape[2]))
        for i in range(al): # 25 random pixels
            h = random.randrange(0, np.size(args,0), 1); v = random.randrange(0, np.size(args,1), 1)
            filter_responses = np.append(filter_responses, [args[h,v]], axis=0)
        '''
        filter_responses = np.append(filter_responses, filter_responses, axis=0)
    

    # Compute kmeans & clusters, then export dictionary file
    print(filter_responses.shape)
    print(K)
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    print(dictionary.shape)
    np.save('dictionary.npy', dictionary)
    
    pass

    ## example code snippet to save the dictionary
    # np.save(join(out_dir, 'dictionary.npy'), dictionary)

def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----

    import scipy.spatial #conda doesn't show in my computer (although installed) -> default to python 3.8 and it requires import

    img = extract_filter_responses(opts, img) # become: (x,y, 3*F(4)*filter_scales)

    '''
    # For loops for each column, row & K words to compare the distance
    wordmap = np.empty([img.shape[0], img.shape[1]])
    product = img.shape[0] * img.shape[1]
    img = np.reshape(img, (-1,3))
    diff = np.empty([product, dictionary.shape[0]])
    diff = scipy.spatial.distance.cdist(img, dictionary)

    for x in range(product):
        wordmap[x] = np.argmin(diff[x,:])

    wordmap = np.reshape(wordmap, (img.shape[0], img.shape[1]), 'C')
    '''
    
    # For loops for each column, row & K words to compare the distance
    wordmap = np.empty([img.shape[0], img.shape[1]])
    for x in range(img.shape[0]): #repeat all for the number of columns
        diff = scipy.spatial.distance.cdist(img[x,:,:], dictionary) #only accepts 2d matrix so compare 1 whole row with 10 k words (all in rgb -> 3 components)
        for y in range(img.shape[1]): #find the min distance for each row
            wordmap[x,y] = 0 #wordmap is just indexing of dictionary
            temp = diff[y,0]
            for z in range(dictionary.shape[0]): #there's 10 distance differences to compare
                if (diff[y,z] < temp):#comparing 1 < 0 till 10 < 9 ==> 9 steps
                    wordmap[x,y] = z
                    temp = diff[y,z]
    
    
    #import pdb; pdb.set_trace()
    return wordmap
    pass























