import cv2
import numpy as np


def face_warp(img_src, img_dest, landmarks, landmarks_dest):
    original_warp_dict = face_warping(img_src, img_dest, landmarks, landmarks_dest, True)
    fake_groundtruth_warp_dict = face_warping(img_src, img_dest, landmarks, landmarks_dest, False)
    return {
        "original_warp": original_warp_dict["warp"],
        "dest_warp": fake_groundtruth_warp_dict["blend"],
    }


def get_landmark_index(landmarks_arr, point):
    try:
        return np.where((landmarks_arr == point).all(axis=1))[0][0]
    except Exception:
        raise ValueError(f"the point {point} is missing from the landmarks")


def face_warping(img_src_rgb, img_dest_rgb, landmarks, landmarks_dest, add_background_landmarks=False):
    """

    :param img_src_rgb:
    :param img_dest_rgb:
    :param landmarks:
    :param landmarks_dest:
    :param add_background_landmarks: If True, apply delaunay triangulation on whole image by adding 8 arbitrary border
            points. Otherwise, only do triangulation for the face
    :return:
    """
    img_src_arr = cv2.cvtColor(img_src_rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros(img_src_rgb.shape[:2], dtype=np.uint8)
    img_dest_arr = cv2.cvtColor(img_dest_rgb, cv2.COLOR_RGB2BGR)

    dest_height, dest_width, channels = img_dest_arr.shape
    img_dest_new_face = np.zeros((dest_height, dest_width, channels), np.uint8)

    src_height, src_width = img_src_rgb.shape[:2]
    if add_background_landmarks:
        landmarks = add_background_points(landmarks, src_height, src_width)
    convexhull_src = cv2.convexHull(landmarks)
    cv2.fillConvexPoly(mask, convexhull_src, 255)

    # Delaunay triangulation
    rect = cv2.boundingRect(convexhull_src)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert([*landmarks])
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    # map the mesh's triangles vertices to the landmarks
    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = get_landmark_index(landmarks, pt1)
        index_pt2 = get_landmark_index(landmarks, pt2)
        index_pt3 = get_landmark_index(landmarks, pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    if add_background_landmarks:
        landmarks_dest = add_background_points(landmarks_dest, dest_height, dest_width)
    convexhull_dest = cv2.convexHull(landmarks_dest)

    for triangle_index in indexes_triangles:
        tr1_pt1 = landmarks[triangle_index[0]]
        tr1_pt2 = landmarks[triangle_index[1]]
        tr1_pt3 = landmarks[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        x, y, w, h, right, bottom = fix_bounding_rect(x, y, w, h, dest_width, dest_height)

        cropped_triangle = img_src_arr[y: bottom, x: right]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)

        points = np.array(
            [[tr1_pt1[0] - x, tr1_pt1[1] - y],
             [tr1_pt2[0] - x, tr1_pt2[1] - y],
             [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32
        )

        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

        # Triangulation of second face
        tr2_pt1 = landmarks_dest[triangle_index[0]]
        tr2_pt2 = landmarks_dest[triangle_index[1]]
        tr2_pt3 = landmarks_dest[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2
        x, y, w, h, right, bottom = fix_bounding_rect(x, y, w, h, dest_width, dest_height)

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array(
            [[tr2_pt1[0] - x, tr2_pt1[1] - y],
             [tr2_pt2[0] - x, tr2_pt2[1] - y],
             [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32
        )

        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        # Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        rotation_matrix = cv2.getAffineTransform(points, points2)
        try:
            warped_triangle = cv2.warpAffine(cropped_triangle, rotation_matrix, (w, h))
        except:
            # invalid shape, one of the dimension is 0, it can just be ignored
            continue

        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # Reconstructing destination face
        img_dest_new_face_rect_area = img_dest_new_face[y: bottom, x: right]
        img_dest_new_face_rect_area_gray = cv2.cvtColor(img_dest_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img_dest_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img_dest_new_face_rect_area = cv2.add(img_dest_new_face_rect_area, warped_triangle)
        img_dest_new_face[y: bottom, x: right] = img_dest_new_face_rect_area

    # Face swapped (putting 1st face into 2nd face)
    img_dest_face_mask = np.zeros(img_dest_rgb.shape[:2], dtype=np.uint8)
    img_dest_head_mask = cv2.fillConvexPoly(img_dest_face_mask, convexhull_dest, 255)
    img_dest_face_mask = cv2.bitwise_not(img_dest_head_mask)

    img_dest_head_noface = cv2.bitwise_and(img_dest_arr, img_dest_arr, mask=img_dest_face_mask)
    result = cv2.add(img_dest_head_noface, img_dest_new_face)

    (x, y, w, h) = cv2.boundingRect(convexhull_dest)
    x, y, w, h, right, bottom = fix_bounding_rect(x, y, w, h, dest_width, dest_height)
    center_face_dest = ((x + right) // 2, (y + bottom) // 2)

    try:
        seamlessclone = cv2.seamlessClone(result, img_dest_arr, img_dest_head_mask, center_face_dest, cv2.NORMAL_CLONE)
    except Exception as e:
        raise e
    return {
        "warp": cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
        "blend": cv2.cvtColor(seamlessclone, cv2.COLOR_BGR2RGB),
    }


def fix_bounding_rect(x, y, b_w, b_h, width, height):
    right = np.clip(x + b_w, 0, width)
    bottom = np.clip(y + b_h, 0, height)
    x = np.clip(x, 0, width)
    y = np.clip(y, 0, height)
    return x, y, right - x, bottom - y, right, bottom


def add_background_points(landmarks_arr, height, width):
    landmarks_arr = np.append(
        landmarks_arr,
        [
            [0, 0],
            [width - 1, 0],
            [(width - 1) // 2, 0],
            [0, height - 1],
            [0, (height - 1) // 2],
            [(width - 1) // 2, height - 1],
            [width - 1, height - 1],
            [(width - 1), (height - 1) // 2]
        ],
        axis=0,
    )

    return landmarks_arr
