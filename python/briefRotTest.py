import matplotlib.pyplot
import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts
import matplotlib.pyplot as plt
import scipy

# Q2.1.6


def rotTest(opts):

    # TODO: Read the image and convert to grayscale, if necessary
    img = cv2.imread("../data/cv_cover.jpg")

    matches_count = []
    rotations = []
    step = 0

    for i in range(36):
        step += 1

        # TODO: Rotate Image
        rot_img = scipy.ndimage.rotate(img, 10.0 * (i))

        # TODO: Compute features, descriptors and Match features
        matches, locs1, locs2 = matchPics(I1=img, I2=rot_img, opts=opts)

        # TODO: Update histogram
        matches_count.append(len(matches))
        rotations.append(10.0 * i)

        # Display matches every 12 steps
        if step == 12:
            plotMatches(im1=img, im2=rot_img, matches=matches, locs1=locs1, locs2=locs2)
            step = 0

    # TODO: Display histogram
    plt.bar(x=rotations, height=matches_count, width=5.0)

    # add labels to axes and add title
    plt.xlabel("Rotation(Degrees)")
    plt.ylabel("Matches")
    plt.title("BRIEF Descriptor Rotation Test")

    # display plot
    plt.show()

if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
