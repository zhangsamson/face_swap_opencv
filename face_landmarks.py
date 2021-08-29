from __future__ import annotations

from abc import abstractmethod
from typing import Type

import numpy as np

from landmarks import Landmarks

DLIB_FACE_LANDMARKS_PARAMS = {
    "right_eye": list(range(42, 48)),
    "right_eyebrow": list(range(22, 27)),
    "left_eye": list(range(36, 42)),
    "left_eyebrow": list(range(17, 22)),
    "mouth": list(range(48, 68)),
    "nose": list(range(27, 36)),
    "contour": list(range(0, 17)),
    "face": list(range(0, 68)),
}

MEDIAPIPE_FACE_LANDMARKS_PARAMS = {
    "right_eye": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
    "right_eye_iris": [
        # 473,  # middle
        474,  # left
        475,  # upper
        476,  # right
        477  # lower
    ],
    "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "left_eye_iris": [
        # 468,  # middle
        471,  # left
        470,  # up
        469,  # right
        472,  # down
    ],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              409, 270, 269, 267, 0, 37, 39, 40, 185,
              78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
              415, 310, 311, 312, 13, 82, 81, 80, 191,
              ],
    "contour": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
                149, 150,
                136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
    "nose": [2, 326, 460, 294, 331, 279, 360, 456, 351, 168, 122, 236, 131, 49, 102, 64, 240, 97]
}


def rotate_landmarks(rotation_matrix, landmarks):
    landmarks_one = np.concatenate([landmarks, np.ones(landmarks.shape[0]).reshape(-1, 1)], axis=1)
    landmarks_manual = np.matmul(rotation_matrix, landmarks_one.T).T
    return landmarks_manual


class FaceLandmarks(Landmarks):
    def __init__(self, landmarks_params, landmarks, dtype=np.int32):
        super(FaceLandmarks, self).__init__(landmarks_params, landmarks, dtype)

    @abstractmethod
    def rotate_landmarks(self, rotation_matrix) -> Type[FaceLandmarks]:
        raise NotImplementedError()

    def get_contour(self):
        return self.get_landmarks_from_part_name("contour")

    def get_left_eye(self):
        return self.get_landmarks_from_part_name("left_eye")

    def get_right_eye(self):
        return self.get_landmarks_from_part_name("right_eye")

    def get_left_eyebrow(self):
        return self.get_landmarks_from_part_name("left_eyebrow")

    def get_right_eyebrow(self):
        return self.get_landmarks_from_part_name("right_eyebrow")

    def get_mouth(self):
        return self.get_landmarks_from_part_name("mouth")

    def get_nose(self):
        return self.get_landmarks_from_part_name("nose")

    def get_face(self):
        return self.get_landmarks_from_part_name("face")


class DlibFaceLandmarks(FaceLandmarks):
    def __init__(self, landmarks, dtype=np.int32):
        super(DlibFaceLandmarks, self).__init__(DLIB_FACE_LANDMARKS_PARAMS, landmarks, dtype=dtype)

    def rotate_landmarks(self, rotation_matrix) -> DlibFaceLandmarks:
        rotated_landmarks = rotate_landmarks(rotation_matrix, self.landmarks)
        return DlibFaceLandmarks(rotated_landmarks)


class MediapipeFaceLandmarks(FaceLandmarks):
    def __init__(self, landmarks, dtype=np.int32):
        super(MediapipeFaceLandmarks, self).__init__(MEDIAPIPE_FACE_LANDMARKS_PARAMS, landmarks, dtype=dtype)

    def rotate_landmarks(self, rotation_matrix) -> MediapipeFaceLandmarks:
        rotated_landmarks = rotate_landmarks(rotation_matrix, self.landmarks)
        return MediapipeFaceLandmarks(rotated_landmarks)

    def get_right_eye_iris(self):
        assert len(self.landmarks) > 468, "There are no iris landmarks. Please use mediapipe iris"
        return super().get_landmarks_from_part_name("right_eye_iris")

    def get_left_eye_iris(self):
        assert len(self.landmarks) > 468, "There are no iris landmarks. Please use mediapipe iris"
        return super().get_landmarks_from_part_name("left_eye_iris")
