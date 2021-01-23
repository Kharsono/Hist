import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K  #K is number of words
    # ----- TODO -----

    #Calculate occurences
    hist, bin_e = np.histogram(wordmap, bins = K, density = False)
    bin_w = bin_e[2] - bin_e[1]


    #Check sum of count elements -> should be # of pixels
    pix = wordmap.shape[0] * wordmap.shape[1]
    nhist = hist/pix

    '''
    #Plotting
    import matplotlib.pyplot as plt

    plt.figure()
    plt.bar(bin_e[:-1], nhist, label = "heights nhist")
    plt.xlabel("Index Dict")
    plt.ylabel("Percentage of showing up")
    plt.grid()
    plt.legend()
    plt.title("Test Hist")
    plt.show()
    '''

    return nhist
    pass


def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K #Number of histog columns
    L = opts.L #Number of layers
    # ----- TODO -----
    
    # Initialization weightage of layer
    wght = np.empty(L)
    '''
    for x in range(L):
        if (x == 0 or x == 1):
            wght[x] = 1/(pow(2, L))
        else:
            wght[x] = pow(2, (x-L))
    '''

    # Calculate number of smaller boxes
    size = int(K*(pow(4,L)-1)/3)
    histall = np.empty([size]) #contains all histogram of all layers
    for x in range(L, 0, -1): #iterate layers going from most fine
        if (L == 1 or L == 2):
            wght[x-1] = (1/pow(2,(L-1)))
        else:
            wght[x-1] = (1/pow(2,(-x+L+1)))
            if (x == 1):
                wght[x-1] = wght[x+1]
        szb = pow(2, (x-1))
        leng = int(wordmap.shape[0]/szb)
        hei = int(wordmap.shape[1]/szb)
        chunk = np.empty(K)             #this is how many boxes & keeps szb*szb amount of histograms
        content = np.empty([leng, hei]) #attempt on copying image in each splitted box
        histmini = np.empty(szb*szb*K)  #result of first concatenate aka per layer
        for a in range(szb):
            for b in range(szb):        #a & b identify which chunk is it in
                for y in range(leng):       #do this for # column IN each chunk/box
                    for z in range(hei):    #do this for # row IN each chunk/box
                        content[y,z] = wordmap[(a*leng + y), (b*hei + z)]
                chunk = get_feature_from_wordmap(opts, content)     #has each histogram

                #1st concatenating within each layer
                if a==0 and b==0 :
                    histmini = chunk  
                else:
                    histmini = np.concatenate((histmini, chunk), axis = None)    
                

        #2nd concatenate + adding weighting scheme
        if(x == L):
            histall = histmini * wght[x-1]
        else:
            histall = np.concatenate((histall, (histmini * wght[x-1])), axis = None)

    #print(L, histall.shape)
        
    return histall
    pass


    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''

    # ----- TODO -----
    print(" get feat called")
    img = Image.open(img_path)      #uploading image and convert into array
    img = np.array(img).astype(np.float32)/255
    wordmap = visual_words.get_visual_words(opts, img, dictionary)  #grab the wordmap for each image
    feat = get_feature_from_wordmap_SPM(opts, wordmap)  #convert the wordmap into histogram with super long bins (depending on the L)
    #print(opts.L, feat.shape)
    
    return feat
    
    pass

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----

    #import pdb; pdb.set_trace()
    
    #Initializing. Purpose: Find features and labels
    K = opts.K; L = opts.L
    size = int(K*(pow(4,L)-1)/3)
    T = len(train_files)
    feat = np.empty([T,size]); lab = np.empty(T)

    #Finding features & labels for T number files
    for x in range(T):
        print(x)
        path = join(opts.data_dir, train_files[x])
        feat[x,:] = get_image_feature(opts, path, dictionary)
        lab[x] = train_labels[x]

    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=feat,
        labels=lab,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )
    
    pass



    ## example code snippet to save the learned system
    # np.savez_compressed(join(out_dir, 'trained_system.npz'),
    #     features=features,
    #     labels=train_labels,
    #     dictionary=dictionary,
    #     SPM_layer_num=SPM_layer_num,
    # )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    
    hor = histograms.shape[0]; ver = histograms.shape[1]
    sim = np.empty(hor)
    s = np.zeros((hor, ver))
    expand = word_hist + s
    #print(expand)
    #print(histograms)
    score = np.minimum(expand, histograms)
    #print(score)
    sim = np.sum(score, axis = 1)
    sim = 1-sim
    '''
    sim = np.empty(histograms.shape[0])
    for i in range(histograms.shape[0]):                # Number of T or N => Numb of features
        score = np.minimum(word_hist, histograms[i])    # find all the min values and array it
        total = 0
        for j in range(histograms.shape[1]): # Number of bins or minimum vals to add to
            total = total + score[j]         # summing all the min bin values
        sim[i] = total                       # Most minim value should be 1 and most max val should be 2 (we normalized)
    sim = 1 - sim
    '''
    #print(sim)
    return sim
    pass    

    #broadcast array function -> check this out
    #duplicate and directly see instead of for loop
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----

    # Initializing variables    
    N = len(test_files)
    histograms = trained_system['features']
    trained_labels = trained_system['labels']
    size = int(test_opts.K*(pow(4,opts.L)-1)/3)
    feat = np.empty([N, size])  # features for each image in test files
    minloc = 0                  # index for finding label in trained files
    calc_labels = np.empty(N)   # labels for test_files (program)

    # Compare test labels from program and true value
    accuracy = 0
    conf = np.empty([8,8])
    for i in range(8):
        for j in range(8):
            conf[i,j] = 0
    
    # Finding the labels for the test_files through program
    mix = None
    for i in range(N):
        print(i)
        path = join(opts.data_dir, test_files[i])
        feat[i,:] = get_image_feature(test_opts, path, dictionary)
        dist = distance_to_set(feat[i], histograms)
        minloc = np.argmin(dist)                    #find the index of lowest distance => most similar
        calc_labels[i] = trained_labels[minloc]     #find labels from program
        conf[ int(test_labels[i]) , int(calc_labels[i]) ] += 1
        if (int(test_labels[i]) == 4 and int(calc_labels[i]) == 3):
            print(path, test_labels[i], calc_labels[i])
             
    accuracy = np.trace(conf)/N

    print(calc_labels)
    return conf, accuracy
            
    pass





















