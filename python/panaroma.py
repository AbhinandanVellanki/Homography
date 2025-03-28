import numpy as np
import cv2

# Import necessary functions
from planarH import compositeH, computeH_ransac
from helper import loadVid
from matchPics import matchPics
from displayMatch import plotMatches
from opts import get_opts


# Q4
def main(opts):
    # Load images
    img1 = cv2.imread("../data/pano_left.jpg")
    img2 = cv2.imread("../data/pano_right.jpg")

    # Taking img2 as the destination image, pad zeros in all directions of width equal to img1

    img2 = cv2.copyMakeBorder(
        img2,
        img1.shape[0],
        img1.shape[0],
        img1.shape[1],
        img1.shape[1],
        cv2.BORDER_CONSTANT,
        None,
        value=0,
    )

    # Find matches
    matches, locs1, locs2 = matchPics(I1=img1, I2=img2, opts=opts)
    # plotMatches(im1=img1, im2=img2, matches=matches, locs1=locs1, locs2=locs2)

    # filter out unmatched interest points from locs1 and locs2
    locs1 = locs1[matches[:, 0]]
    locs2 = locs2[matches[:, 1]]

    # swap columns of locs outputs of matchPics
    locs1[:, [0, 1]] = locs1[:, [1, 0]]
    locs2[:, [0, 1]] = locs2[:, [1, 0]]

    # compute Homography
    HLtoR, inliers = computeH_ransac(locs1=locs2, locs2=locs1, opts=opts)

    # warp and paste left image into right image coordinates
    panorama = compositeH(template=img1, H2to1=HLtoR, img=img2)

    # Remove borders from panorama
    _, thresh = cv2.threshold(
        cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY
    )
    contours, h = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    panorama = panorama[y : y + h, x : x + w]

    cv2.imwrite("../result/mypano1.jpeg", panorama)

    return


if __name__ == "__main__":

    opts = get_opts()
    main(opts)
