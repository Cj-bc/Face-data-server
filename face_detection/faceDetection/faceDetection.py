# module faceDetection
# (faceCalibration, facemark, LANDMARK_NUM) where
import os
import cv2
import dlib
from typing import List
from Types import (CalibrationData, Cv2Image, CapHasClosedError)
from functools import reduce


# -- Variables

# Those values are defined based on this site image:
#   https://qiita.com/kekeho/items/0b2d4ed5192a4c90a0ac
# ignore
LANDMARK_NUM = {"TIN_CENTER": 19
               , "MOUSE_R": 58
               , "MOUSE_L": 71
               , "MOUSE_TOP": 65
               , "MOUSE_BOTTOM": 79
               , "LEFT_EYE_R": 114
               , "LEFT_EYE_L": 124
               , "LEFT_EYE_TOP": 120
               , "LEFT_EYE_BOTTOM": 129
               , "RIGHT_EYE_R": 145
               , "RIGHT_EYE_L": 135
               , "RIGHT_EYE_TOP": 140
               , "RIGHT_EYE_BOTTOM": 149
               , "LEFT_EYEBROW_R": 154
               , "RIGHT_EYEBROW_L": 174
               }

# Cascade files directory path
_CASCADE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/../haarcascades/"
_LEARNED_MODEL_PATH = os.path.dirname(
    os.path.abspath(__file__)) + "/../learned-models/"

predictor = dlib.shape_predictor(
    _LEARNED_MODEL_PATH + 'helen-dataset.dat')
face_cascade = cv2.CascadeClassifier(
    _CASCADE_PATH + 'haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()


# faceCalibration(cap: cv2.VideoCapture) -> CalibrationData {{{
def faceCalibration(cap: cv2.VideoCapture) -> CalibrationData:
    """Calibrate individuals' differences.

    What this function does are:
        1. Get distance between eyes
    """
    print("=========== Face calibration ===========")
    input("Please face front and press enter:")
    frame = waitUntilFaceDetect(cap)
    print("got your face... wait for a second...")
    landmarks = facemark(frame)

    return {}
# }}}


# waitUntilFaceDetect(cap: cv2.VideoCapture) -> Cv2Image {{{
def waitUntilFaceDetect(cap: cv2.VideoCapture) -> Cv2Image:
    """Wait until face(s) is detected. Return frame if it contains faces.

        Raise Exception:
            CapHasClosedError : this exception might be raised
                                when 'cap' has been closed by some reason
    """
    while cap.isOpened():
        _, frame = cap.read()
        if isFaceExist(frame):
            return frame

    raise CapHasClosedError()
# }}}


# isFaceExist(gray_img: Cv2Image) -> bool {{{
def isFaceExist(gray_img: Cv2Image) -> bool:
    """True if faces are exist in image
    """
    faces = face_cascade.detectMultiScale(gray_img, minSize=(100, 100))
    return faces.size() != 0

# }}}


# facemark(gray_img: Cv2Image) -> List[Landmark] {{{
def facemark(gray_img: Cv2Image) -> List[dlib.point]:
    """Recoginize face landmark position by i-bug 300-w dataset
    Return:
        randmarks = [
        [x, y],
        [x, y],
        ...
        ]
        [0~40]: chin
        [41~57]: nose
        [58~85]: outside of lips
        [86-113]: inside of lips
        [114-133]: right eye
        [134-153]: left eye
        [154-173]: right eyebrows
        [174-193]: left eyebrows
    """
    rects: dlib.rectangles = detector(gray_img, 1)

    for rect in rects:
        parts: dlib.points = predictor(gray_img, rect).parts()
        wholeFace: List[dlib.point] = reduce(lambda l, n: l + [n], parts, [])
    return _normalization(wholeFace)
# }}}


# _normalization(face_landmarks: List[dlib.point]) -> List[dlib.point] {{{
def _normalization(face_landmarks: List[dlib.point]) -> List[dlib.point]:
    """Normalize facemark result. FOR INTERNAL USE
        Please refer to [this image]() [WIP]

        This code was written by @kekeho(Qiita), refer to:
            https://qiita.com/kekeho/items/0b2d4ed5192a4c90a0ac

    """
    return_list = []
    for facemark in face_landmarks:
        # nose
        nose = list(range(130, 147 + 1))
        nose.remove(139)

        # right eyebrows
        right_eyebrow = list(range(62, 83 + 1))
        right_eyebrow.remove(68)
        right_eyebrow.remove(79)

        # left eyebrows
        left_eyebrow = list(range(84, 105 + 1))
        left_eyebrow.remove(90)
        left_eyebrow.remove(101)

        # right eye
        right_eye = list(range(18, 39 + 1))
        right_eye.remove(24)
        right_eye.remove(35)

        # left_eye
        left_eye = list(range(40, 61 + 1))
        left_eye.remove(46)
        left_eye.remove(57)

        # outside lips
        outside_lips = list(range(148, 178 + 1))
        outside_lips.remove(150)
        outside_lips.remove(161)
        outside_lips.remove(172)

        # inside lips
        inside_lips = list(range(3, 17 + 1))
        inside_lips.remove(13)
        add_list_lips = list(range(179, 193 + 1))
        add_list_lips.remove(183)
        inside_lips += add_list_lips

        # chin
        chin = [0, 1, 106, 117, 128, 139, 150, 161, 172,
                183, 2, 13, 24, 35, 46, 57, 68, 79, 90, 101]
        add_list = list(range(107, 129 + 1))
        add_list.remove(117)
        add_list.remove(128)
        chin += add_list

        for nose_i, fm_i in enumerate(nose):
            nose[nose_i] = facemark[fm_i]

        for reb_i, fm_i in enumerate(right_eyebrow):
            right_eyebrow[reb_i] = facemark[fm_i]

        for leb_i, fm_i in enumerate(left_eyebrow):
            left_eyebrow[leb_i] = facemark[fm_i]

        for re_i, fm_i in enumerate(right_eye):
            right_eye[re_i] = facemark[fm_i]

        for le_i, fm_i in enumerate(left_eye):
            left_eye[le_i] = facemark[fm_i]

        for ol_i, fm_i in enumerate(outside_lips):
            outside_lips[ol_i] = facemark[fm_i]

        for il_i, fm_i in enumerate(inside_lips):
            inside_lips[il_i] = facemark[fm_i]

        for chin_i, fm_i in enumerate(chin):
            chin[chin_i] = facemark[fm_i]

        return_list.append(chin + nose + outside_lips + inside_lips +
                           right_eye + left_eye + right_eyebrow + left_eyebrow)

    return return_list
# }}}
