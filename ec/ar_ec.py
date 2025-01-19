import numpy as np
import cv2
import sys

# Import necessary functions
from planarH import compositeH, computeH_ransac
from loadVid import loadVid
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

    # initialise ORB detector
    orb = cv2.ORB_create(nfeatures=1000)

    # declare FLANN matcher parameters
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1)
    flann = cv2.FlannBasedMatcher(index_params, {})

    for source_frame, destination_img in zip(
        source_video_frames, destination_video_frames
    ):
        # Get the template to be pasted onto the destination frame
        template = crop_middle_resize(
            element=element, frame=source_frame, ratio=element_ratio
        )

        # detect features and descriptors
        feat_template, desc_template = orb.detectAndCompute(template, None)
        feat_destination, desc_destination = orb.detectAndCompute(destination_img, None)

        # Find matches
        matches = flann.knnMatch(desc_template, desc_destination, k=2)

        # Filter out good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 1.0 * n.distance:
                good_matches.append(m)

        # Get point pairs of matches
        locs_template = np.float32([feat_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        locs_destination = np.float32([feat_destination[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # Compute homography
        HEtoD, inliers = computeH_ransac(locs1=locs_destination, locs2=locs_template, opts=opts)

        new_frame = compositeH(img=destination_img, template=template, H2to1=HEtoD)

        cv2.imshow("Real-Time AR", new_frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    return

if __name__ == "__main__":

    opts = get_opts()
    main(opts=opts)


# Q3.2
