import numpy as np
import cv2
import skimage.io
import skimage.color
from opts import get_opts
from matchPics import matchPics
from displayMatch import plotMatches
from planarH import computeH_ransac, compositeH

# Import necessary functions

# Q2.2.4


def warpImage(opts):
    img_1 = np.array(cv2.imread("../data/cv_cover.jpg"))
    img_2 = np.array(cv2.imread("../data/cv_desk.png"))
    hp_cover = np.array(cv2.imread("../data/hp_cover.jpg"))

    # Resizing the template to the same size as the source image
    hp_cover = cv2.resize(
        src=hp_cover,
        dsize=(img_1.shape[1], img_1.shape[0]),
    )

    # compute homography between img_1 and img_2
    matches, locs1, locs2 = matchPics(I1=img_1, I2=img_2, opts=opts)

    # display matches
    #plotMatches(img_1, img_2, matches, locs1, locs2)

    # filter out unmatched interest points from locs1 and locs2
    locs1 = locs1[matches[:, 0]]
    locs2 = locs2[matches[:, 1]]

    # swap columns of locs outputs of matchPics
    locs1[:, [0, 1]] = locs1[:, [1, 0]]
    locs2[:, [0, 1]] = locs2[:, [1, 0]]

    H2to1, inliers = computeH_ransac(locs1=locs1, locs2=locs2, opts=opts)
    H1to2 = np.linalg.inv(H2to1)

    composite_img = compositeH(img=img_2, template=hp_cover, H2to1=H1to2)

    cv2.imshow("Harry Potterized Book", composite_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)
