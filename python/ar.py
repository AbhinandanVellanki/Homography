import numpy as np
import cv2

# Import necessary functions
from planarH import compositeH, computeH_ransac
from helper import loadVid
from matchPics import matchPics
from displayMatch import plotMatches
from opts import get_opts


def crop_middle_resize(element, frame, ratio):
    # To extract the centre part of the frame by a ratio and resize it to the same size as the element

    # Get width of centre part from height and ratio
    crop_width = ratio * frame.shape[0]
    crop_height = frame.shape[0]

    # Crop out centre part
    mid_x, mid_y = int(frame.shape[1] / 2), int(frame.shape[0] / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = frame[mid_y - ch2 : mid_y + ch2, mid_x - cw2 : mid_x + cw2]

    # Remove zeros from the top and bottom
    crop_img = crop_img[
        43:-43, :, :
    ]  # these dimensions were found for the specific video

    # Resize the cropped part to the element dimensions
    crop_img = cv2.resize(src=crop_img, dsize=(element.shape[1], element.shape[0]))

    return crop_img


def main(opts):

    # load video
    source_video_frames = loadVid(path="../data/ar_source.mov")
    destination_video_frames = loadVid(path="../data/book.mov")

    # load element for calculating homography
    element = cv2.imread("../data/cv_cover.jpg")

    # determine ratio
    element_ratio = element.shape[1] / element.shape[0]  # width / height

    # video writer
    output = cv2.VideoWriter(
        "../result/ar_video.avi",
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        20,
        (destination_video_frames[0].shape[1], destination_video_frames[0].shape[0]),
    )

    for source_frame, destination_img in zip(
        source_video_frames, destination_video_frames
    ):

        # Get the template to be pasted onto the destination frame
        template = crop_middle_resize(
            element=element, frame=source_frame, ratio=element_ratio
        )

        # Get matching points between element and destination frame
        matches, locs1, locs2 = matchPics(I1=element, I2=destination_img, opts=opts)

        # filter out unmatched interest points from locs1 and locs2
        locs1 = locs1[matches[:, 0]]
        locs2 = locs2[matches[:, 1]]

        # swap columns of locs outputs of matchPics
        locs1[:, [0, 1]] = locs1[:, [1, 0]]
        locs2[:, [0, 1]] = locs2[:, [1, 0]]

        HEtoD, inliers = computeH_ransac(locs1=locs1, locs2=locs2, opts=opts)
        HEtoD = np.linalg.inv(HEtoD)

        new_frame = compositeH(img=destination_img, template=template, H2to1=HEtoD)

        output.write(new_frame)

    output.release()

if __name__ == "__main__":

    opts = get_opts()
    main(opts=opts)


# Write script for Q3.1
