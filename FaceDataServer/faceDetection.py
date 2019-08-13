# module faceDetection
# (faceCalibration, facemark) where
import os
import cv2
import dlib
from typing import List, Optional
from .Types import (Cv2Image, CapHasClosedError,
                    RawFaceData, Face)
from functools import reduce


# -- Variables

# Cascade files directory path
_SRC_PATH = os.path.dirname(os.path.abspath(__file__)) + "/../src"
_predictor = dlib.shape_predictor(_SRC_PATH + '/helen-dataset.dat')
_detector = dlib.get_frontal_face_detector()


# faceCalibration(cap: cv2.VideoCapture) -> RawFaceData {{{
def faceCalibration(cap: cv2.VideoCapture) -> RawFaceData:
    """Calibrate individuals' differences.

    What this function does are:
        1. Get distance between eyes
    """
    print("=========== Face calibration ===========")
    input("Please face front and press enter:")
    frame = _waitUntilFaceDetect(cap)
    print("got your face... wait for a second...")
    face: Face = Face.fromDPoints(facemark(frame))
    print("done :)")
    return RawFaceData.get(face)
# }}}


# facemark(gray_img: Cv2Image) -> Optional[dlib.dpoints] {{{
def facemark(gray_img: Cv2Image) -> Optional[dlib.dpoints]:
    """Recoginize face landmark position by i-bug 300-w dataset
        This will return biggest face from recognized faces list

        If no faces are found, it'll return None

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
    rects: dlib.rectangles = _detector(gray_img, 1)
    wholeFace: List[dlib.dpoints] = []
    for rect in rects:
        parts: dlib.dpoints =\
            _points2dpoints(_predictor(gray_img, rect).parts())
        wholeFace.append(parts)

    if len(wholeFace) == 0:
        return None

    absolute_coord = _normalization(_getBiggestFace(wholeFace))
    center = absolute_coord[49]
    return _toRelative(absolute_coord, center)
# }}}


# _waitUntilFaceDetect(cap: cv2.VideoCapture) -> Cv2Image {{{
def _waitUntilFaceDetect(cap: cv2.VideoCapture) -> Cv2Image:
    """Wait until face(s) is detected. Return frame if it contains faces.

        Raise Exception:
            CapHasClosedError : this exception might be raised
                                when 'cap' has been closed by some reason
    """
    while cap.isOpened():
        _, frame = cap.read()
        if _isFaceExist(frame):
            return frame

    raise CapHasClosedError()
# }}}


# _isFaceExist(gray_img: Cv2Image) -> bool {{{
def _isFaceExist(gray_img: Cv2Image) -> bool:
    """True if faces are exist in image """
    faces: dlib.rectangles = _detector(gray_img, 1)
    return len(faces) != 0

# }}}


# _getBiggestFace(faces: List[dlib.dpoints]) -> dlib.dpoints: {{{
def _getBiggestFace(faces: List[dlib.dpoints]) -> dlib.dpoints:
    """ Return biggest face in given list
       'biggest face' is one that has biggest width

    """
    def ln(p: dlib.dpoints) -> float:
        return abs((p[40] - p[0]).x)

    return reduce(lambda p, q: p if ln(p) > ln(q) else q,
                  faces, dlib.dpoints(194))
# }}}


# _normalization(face: dlib.dpoints) -> dlib.dpoints {{{
def _normalization(face: dlib.dpoints) -> dlib.dpoints:
    """Normalize facemark result. FOR INTERNAL USE
        Please refer to [this image]() [WIP]

        This code was written by @kekeho(Qiita), refer to:
            https://qiita.com/kekeho/items/0b2d4ed5192a4c90a0ac

    """
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
        nose[nose_i] = face[fm_i]

    for reb_i, fm_i in enumerate(right_eyebrow):
        right_eyebrow[reb_i] = face[fm_i]

    for leb_i, fm_i in enumerate(left_eyebrow):
        left_eyebrow[leb_i] = face[fm_i]

    for re_i, fm_i in enumerate(right_eye):
        right_eye[re_i] = face[fm_i]

    for le_i, fm_i in enumerate(left_eye):
        left_eye[le_i] = face[fm_i]

    for ol_i, fm_i in enumerate(outside_lips):
        outside_lips[ol_i] = face[fm_i]

    for il_i, fm_i in enumerate(inside_lips):
        inside_lips[il_i] = face[fm_i]

    for chin_i, fm_i in enumerate(chin):
        chin[chin_i] = face[fm_i]

    return dlib.dpoints(chin + nose + outside_lips + inside_lips
                        + right_eye + left_eye
                        + right_eyebrow + left_eyebrow)
# }}}


# _points2dpoints(ps: dlib.points) -> dlib.dpoints: {{{
def _points2dpoints(ps: dlib.points) -> dlib.dpoints:
    """convert dlib.points object to dlib.dpoints object.
        All points() are should be converted to dpoints,
        as we use float values
    """
    ret = dlib.dpoints()
    for p in ps:
        ret.append(dlib.dpoint(float(p.x), float(p.y)))

    return ret
# }}}



# _toRelative(target: dlib.dpoints, center: dlib.dpoint) -> dlib.dpoints: {{{
def _toRelative(target: dlib.dpoints, center: dlib.dpoint) -> dlib.dpoints:
    converted = list(map(lambda p: p - center, target))
    return dlib.dpoints(converted)
# }}}
