import numpy as np


def convert_mediapipe_landmarks(mediapipe_landmarks,
                                img_width,
                                img_height,
                                scale_wh=True,
                                three_dims=False):
    if scale_wh:
        scale_w = img_width
        scale_h = img_height
    else:
        scale_w = 1
        scale_h = 1

    if three_dims:
        return np.array([
            [
                landmark.x * scale_w,
                landmark.y * scale_h,
                landmark.z,
            ] for landmark in mediapipe_landmarks.landmark
        ])
    else:
        return np.array([
            [
                landmark.x * scale_w,
                landmark.y * scale_h,
            ] for landmark in mediapipe_landmarks.landmark
        ])


def get_enclosing_rectangle(points: np.array):
    left = points[:, 0].min()
    top = points[:, 1].min()
    right = points[:, 0].max()
    bottom = points[:, 1].max()

    return left, top, right, bottom


def rescale_landmarks(landmarks, width_ratio: float, height_ratio: float):
    """

    :param landmarks: [N, 2]-shape XY format
    :param width_ratio:
    :param height_ratio:
    :return: rescaled landmarks
    """

    return (landmarks * np.array([width_ratio, height_ratio])).astype(np.int32)
