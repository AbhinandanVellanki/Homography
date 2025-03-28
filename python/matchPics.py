import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

# Q2.1.4

def matchPics(I1, I2, opts):
    """
    Match features across images

    Input
    -----
    I1, I2: Source images
    opts: Command line args

    Returns
    -------
    matches: List of indices of matched features across I1, I2 [p x 2]
    locs1, locs2: Pixel coordinates of matches [N x 2]
    """

    ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'

    # Convert Images to GrayScale
    gI1 = skimage.color.rgb2gray(I1)
    gI2 = skimage.color.rgb2gray(I2)

    # Detect Features in Both Images
    corners_gI1 = corner_detection(img=gI1, sigma=sigma)
    corners_gI2 = corner_detection(img=gI2, sigma=sigma)

    # Obtain descriptors for the computed feature locations
    descriptors_gI1, locs1 = computeBrief(img=gI1, locs=corners_gI1)
    descriptors_gI2, locs2 = computeBrief(img=gI2, locs=corners_gI2)

    # Match features using the descriptors
    matches = briefMatch(desc1=descriptors_gI1, desc2=descriptors_gI2, ratio=ratio)

    return matches, locs1, locs2
