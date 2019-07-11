import os
import cv2
import dlib
from typing import List, Union, Tuple
from Types import (CalibrationData, Landmark, Cv2Image, CapHasClosedError)
import numpy


# -- Variables

# Those values are defined based on this site image:
#   https://qiita.com/kekeho/items/0b2d4ed5192a4c90a0ac
# ignore
LANDMARK_NUM =  { "MOUSE_R": 58
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
                }

# Cascade files directory path
CASCADE_PATH = os.path.dirname(os.path.abspath(__file__)) + "/haarcascades/"
LEARNED_MODEL_PATH = os.path.dirname(
    os.path.abspath(__file__)) + "/learned-models/"
predictor = dlib.shape_predictor(
    LEARNED_MODEL_PATH + 'helen-dataset.dat')
face_cascade = cv2.CascadeClassifier(
    CASCADE_PATH + 'haarcascade_frontalface_default.xml')


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


# waitUntilFaceDetect(cap: cv2.VideoCapture) -> numpy.ndarray {{{
def waitUntilFaceDetect(cap: cv2.VideoCapture) -> numpy.ndarray:
    """Wait until face(s) is detected. Return detected face(s).

        Raise Exception:
            CapHasClosedError : this exception might be raised
                                when 'cap' has been closed by some reason
    """
    while cap.isOpened():
        _, frame = cap.read()
        faces_roi = face_position(frame)
        if faces_roi != ():
            return faces_roi

    raise CapHasClosedError()
# }}}

# face_position(gray_img) {{{
def face_position(gray_img: numpy.ndarray) -> Union[numpy.ndarray, Tuple]:
    """Detect faces position
    Return:
        faces: faces position list (x, y, w, h)
    """
    faces = face_cascade.detectMultiScale(gray_img, minSize=(100, 100))
    return faces
# }}}


# facemark(gray_img): {{{
def facemark(gray_img: numpy.ndarray) -> List[Landmark]:
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
    faces_roi = face_position(gray_img)
    landmarks = []

    for face in faces_roi:
        x, y, w, h = face
        face_img = gray_img[y: y + h, x: x + w];

        detector = dlib.get_frontal_face_detector()
        rects = detector(gray_img, 1)

        landmarks = []
        for rect in rects:
            landmarks.append(
                numpy.array(
                    [(p.x, p.y) for p in predictor(gray_img, rect).parts()])
            )
    return _normalization(landmarks)
# }}}


# _normalization(face_landmarks): {{{
def _normalization(face_landmarks: List[Landmark]) -> List[Landmark]:
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
