import numpy as np


class Landmarks:

    def __init__(self, landmarks_params, landmarks, dtype=np.int32):
        """

        :param landmarks_params:
        :param landmarks: face landmarks in (width, height) format
        """

        self.landmarks_params = landmarks_params
        self.landmarks = np.array(landmarks, dtype=dtype) if not issubclass(type(landmarks), Landmarks) else np.array(
            landmarks.landmarks, dtype=dtype
        )

    def get_all_landmarks(self) -> np.array:
        return self.landmarks

    def __array__(self) -> np.array:
        return self.get_all_landmarks()

    def get_landmarks_from_part_name(self, part_name):
        indexes = self.landmarks_params.get(part_name)
        if indexes is not None:
            return self.landmarks[indexes, :]

        raise NotImplementedError()
