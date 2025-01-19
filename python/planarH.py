import numpy as np
import cv2


def computeH(x1, x2):
    # Q2.2.1
    # TODO: Compute the homography between two sets of points

    if x1.shape[0] == x2.shape[0]:
        N = x1.shape[0]
        if N < 4:
            return None
    else:
        return None

    # declare A matrix
    A = []

    # populate A
    for i in range(N):
        row1 = [
            x2[i][0],
            x2[i][1],
            1,
            0,
            0,
            0,
            -x1[i][0] * x2[i][0],
            -x1[i][0] * x2[i][1],
            -x1[i][0],
        ]
        row2 = [
            0,
            0,
            0,
            x2[i][0],
            x2[i][1],
            1,
            -x1[i][1] * x2[i][0],
            -x1[i][1] * x2[i][1],
            -x1[i][1],
        ]
        A.append(row1)
        A.append(row2)

    A = np.array(A)

    # svd method:
    U, s, V = np.linalg.svd(A, full_matrices=True)
    H2to1 = V[-1].reshape((3, 3))

    return H2to1


def compute_centroid(x):
    # compute centroid by taking mean along both axes
    length = x.shape[0]
    sum_x = np.sum(x[:, 0])
    sum_y = np.sum(x[:, 1])
    return (sum_x // length, sum_y // length)


def shift_points(points, origin):
    # shift the origin of points to the new origin
    N = points.shape[0]
    for i in range(N):
        points[i][0] = points[i][0] - origin[0]
        points[i][1] = points[i][1] - origin[1]
    return points


def normalize(points):
    max_distance = -1
    for i in range(points.shape[0]):
        dist = np.linalg.norm(points[i])
        if dist > max_distance:
            max_distance = dist
    scale = (2**0.5) / max_distance
    points = scale * points
    return points, scale


def computeH_norm(x1, x2):
    # Q2.2.2
    # TODO: Compute the centroid of the points

    centroid_x1 = compute_centroid(x1)
    centroid_x2 = compute_centroid(x2)

    # TODO: Shift the origin of the points to the centroid
    x1_shifted = shift_points(x1, centroid_x1)
    x2_shifted = shift_points(x2, centroid_x2)

    # TODO: Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    x1_normalized, x1_scale = normalize(x1_shifted)
    x2_normalized, x2_scale = normalize(x2_shifted)

    # TODO: Similarity transform 1
    T_1 = np.array(
        [
            [1, 0, -centroid_x1[0]],
            [0, 1, -centroid_x1[1]],
            [0, 0, 1 / x1_scale],
        ]
    )
    T_1 = x1_scale * T_1

    # TODO: Similarity transform 2
    T_2 = np.array(
        [
            [1, 0, -centroid_x2[0]],
            [0, 1, -centroid_x2[1]],
            [0, 0, 1 / x2_scale],
        ]
    )
    T_2 = x2_scale * T_2

    # TODO: Compute homography
    norm_H2to1 = computeH(x1=x1_normalized, x2=x2_normalized)

    # TODO: Denormalization - convert norm_H2to1 to H2to1
    T_1_inverse = np.linalg.inv(T_1)
    H2to1 = T_1_inverse @ norm_H2to1 @ T_2

    return H2to1


def computeH_ransac(locs1, locs2, opts):
    # Q2.2.3 Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = (
        opts.inlier_tol
    )  # the tolerance value for considering a point to be an inlier

    if locs1.shape == locs2.shape:
        N = locs1.shape[0]
    else:
        return None

    iterations = 0
    inliers = []
    bestH2to1 = None

    while iterations < max_iters:
        # select four matching pairs randomly
        chosen_points_idx = np.random.choice(len(locs1), 4, replace=False)
        x1 = locs1[chosen_points_idx]
        x2 = locs2[chosen_points_idx]

        # compute homography H
        this_H2to1 = computeH_norm(x1=x1, x2=x2)

        # transform all source points using H, compare with actual points and count inliers
        this_inliers = []

        for i in range(N):
            # transform source points
            point = np.array([[locs2[i][0]], [locs2[i][1]], [1]])
            transformed_point = this_H2to1 @ point
            transformed_point = 1 / transformed_point[2][0] * transformed_point
            actual_point = np.array([[locs1[i][0]], [locs1[i][1]], [1]])

            # calculate error
            error = np.linalg.norm(transformed_point - actual_point)
            print(error)

            # update best inliers and best H
            if error < inlier_tol:
                this_inliers.append(1)
            else:
                this_inliers.append(0)

        if np.count_nonzero(this_inliers) > np.count_nonzero(inliers):
            bestH2to1 = this_H2to1
            inliers = this_inliers

        iterations += 1
        
    # Re compute H with all the inliers
    best_inliers_1 = []
    best_inliers_2 = []
    for i in range(N):
        if inliers[i] == 1:
            best_inliers_1.append(locs1[i])
            best_inliers_2.append(locs2[i])

    bestH2to1 = computeH_norm(x1=np.array(best_inliers_1), x2=np.array(best_inliers_2))

    return bestH2to1, inliers


def compositeH(H2to1, template, img):

    # Create a composite image after warping the template image on top
    # of the image using the homography

    # Note that the homography we compute is from the image to the template;
    # x_template = H2to1*x_photo
    # For warping the template to the image, we need to invert it.

    # TODO: Create mask of same size as template
    mask = np.ones(template[:, :, 0].shape, dtype="uint8")
    mask = 255 * mask

    # TODO: Warp mask by appropriate homography
    mask = cv2.warpPerspective(src=mask, M=H2to1, dsize=(img.shape[1], img.shape[0]))

    # TODO: Warp template by appropriate homography
    template = cv2.warpPerspective(
        src=template, M=H2to1, dsize=(img.shape[1], img.shape[0])
    )

    # TODO: Use mask to combine the warped template and the image
    composite_img = cv2.bitwise_and(src1=img, src2=img, mask=cv2.bitwise_not(mask))
    composite_img += template

    return composite_img
