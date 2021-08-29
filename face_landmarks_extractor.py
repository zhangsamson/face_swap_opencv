"""
Wrapper for face landmarks models
"""

import os
from abc import abstractmethod
from typing import Dict

import mediapipe as mp
import numpy as np
from face_landmarks import MediapipeFaceLandmarks, DlibFaceLandmarks
from utils import convert_mediapipe_landmarks, get_enclosing_rectangle

try:
    import dlib
except:
    print("Please install dlib if you want to use dlib")


class LandmarksExtractor:
    @abstractmethod
    def __call__(self, img) -> Dict:
        """

        :param img: input img
        :return: dict with at least "face_landmarks" keys with list of instances of FaceLandmarks subtype
        """
        raise NotImplementedError()


class DlibLandmarksExtractor(LandmarksExtractor):
    def __init__(
            self,
            cnn_face_detector=None,
            face_landmarks_predictor=None,
            device="cpu",
    ):
        """

        :param cnn_face_detector:
        :param face_landmarks_predictor:
        :param device: fallbacks to "cpu" if cuda not available
        """

        if cnn_face_detector is None:
            # cnn_face_detector = FasterRCNNFaceDetector(device=device)
            cnn_face_detector = dlib.cnn_face_detection_model_v1("./models/mmod_human_face_detector.dat")
        self.cnn_face_detector = cnn_face_detector
        if face_landmarks_predictor is None:
            weights_fp = os.path.join(os.path.dirname(__file__), "./models/shape_predictor_68_face_landmarks.dat")
            face_landmarks_predictor = dlib.shape_predictor(weights_fp)
        if "cuda" in str(device) and dlib.cuda.get_num_devices():
            dlib.cuda.set_device(0)
        self.face_landmarks_predictor = face_landmarks_predictor

    def __call__(self, image):
        boxes = self.cnn_face_detector(image, 0)

        face_landmarks_list = []
        bboxes = []
        for box in boxes:
            box = box.rect
            bboxes.append((box.left, box.top, box.right, box.bottom))
            face_landmarks = self.face_landmarks_predictor(image, box)

            face_landmarks = np.array(
                [[face_landmarks.part(i).x, face_landmarks.part(i).y] for i in range(68)]
            ).astype(
                np.int32
            )

            face_landmarks_list.append(DlibFaceLandmarks(face_landmarks))

        return {
            "face_landmarks": face_landmarks_list,
            "bounding_boxes": bboxes,
        }


class MediapipeFaceLandmarksExtractor:
    def __init__(self, max_num_faces=10, min_detection_confidence=0.5, scale_wh=True, three_dims=False, iris=False):
        """

        :param max_num_faces: only works in non-iris mode
        :param min_detection_confidence:
        :param scale_wh:
        :param three_dims:
        :param iris: use mediapipe iris landmarks model which is equivalent to face_mesh (468 points) model and
            10 additional landmarks are added for both eyes' irises
            if iris mode is used, the max_num_faces argument is ignored since it only works on 1 face
        """
        self.max_num_faces = max_num_faces
        self.min_detection_confidence = min_detection_confidence
        self.scale_wh = scale_wh
        self.three_dims = three_dims
        self.iris = iris

    def __call__(self, image):
        h, w = image.shape[:2]

        face_landmarks_list = []
        bboxes = []

        # need iris python bindings in mediapipe, might need to compile from source manually
        if self.iris:
            face_mesh = mp.solutions.iris.Iris(
                static_image_mode=True,
                min_detection_confidence=self.min_detection_confidence,
            )
        else:
            face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=self.max_num_faces,
                min_detection_confidence=self.min_detection_confidence
            )

        results = face_mesh.process(image)

        results_landmarks = [results.face_landmarks_with_iris] if self.iris else results.multi_face_landmarks

        if results_landmarks:
            for landmarks in results_landmarks:
                landmarks_np = convert_mediapipe_landmarks(
                    landmarks, w, h, three_dims=self.three_dims,
                    scale_wh=self.scale_wh
                )
                face_landmarks_list.append(MediapipeFaceLandmarks(landmarks_np))
                bboxes.append(get_enclosing_rectangle(landmarks_np))

        face_mesh.close()

        return {
            "face_landmarks": face_landmarks_list,
            "bounding_boxes": bboxes,
        }
