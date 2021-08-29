import argparse

import cv2

from face_landmarks_extractor import MediapipeFaceLandmarksExtractor, DlibLandmarksExtractor
from face_warper import face_warp

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-src", "--src_face", action="store", type=str, required=True,
                    help="Source face to swap from")
    ap.add_argument("-dest", "--dest_face", action="store", type=str, required=True,
                    help="Dest face to swap the source face to")
    ap.add_argument("-o", "--output", action="store", type=str, default="out.jpg",
                    help="dir containing background images used for augmentation")
    ap.add_argument("-l", "--landmarks", action="store", type=str, default="mediapipe",
                    choices=["mediapipe", "dlib"],
                    help="dir containing background images used for augmentation")
    args = ap.parse_args()

    img_src = cv2.cvtColor(cv2.imread(args.src_face), cv2.COLOR_BGR2RGB)
    img_dest = cv2.cvtColor(cv2.imread(args.dest_face), cv2.COLOR_BGR2RGB)

    if args.landmarks == "dlib":
        landmarks_extractor = DlibLandmarksExtractor(device="cuda")
    else:
        landmarks_extractor = MediapipeFaceLandmarksExtractor(max_num_faces=1)

    landmarks_src = landmarks_extractor(img_src)["face_landmarks"][0].landmarks
    landmarks_dest = landmarks_extractor(img_dest)["face_landmarks"][0].landmarks

    warp_res = face_warp(
        img_src,
        img_dest,
        landmarks_src,
        landmarks_dest,
    )

    cv2.imwrite(args.output, cv2.cvtColor(warp_res["dest_warp"], cv2.COLOR_RGB2BGR))
