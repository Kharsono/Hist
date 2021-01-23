import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    
    # estimate noise -> denoise -> greyscale -> threshold
    denoise = skimage.restoration.denoise_bilateral(image, multichannel = True)
    greyscale = skimage.color.rgb2gray(image)
    threshold = skimage.filters.threshold_otsu(greyscale)
    
    # morphology -> label -> skip small boxes
    bw = greyscale < threshold
    bw = skimage.morphology.closing(bw, skimage.morphology.square(5)) #Dilate after erosion (remove dark spots)
    label = skimage.morphology.label(bw, connectivity = 2)
    
    # Skip small boxes
    properties = skimage.measure.regionprops(label) #Measure properties
    mean = 0
    for prop in properties:
        mean += prop.area
    mean = mean / len(properties)
    
    bboxes = [prop.bbox for prop in properties if prop.area > mean / 4]
    bw = (~bw).astype(np.float)
    
    ##########################

    return bboxes, bw