# module faceDetection
# (faceCalibration, facemark) where
import os
import cv2
import dlib
from typing import List, Optional
from .Types import (Cv2Image, CapHasClosedError,
                    RawFaceData, Face, ExitCode, Tuple)
from functools import reduce


# -- Variables

# Cascade files directory path
_SRC_PATH = os.path.dirname(os.path.abspath(__file__)) + "/../src"
_predictor = dlib.shape_predictor(_SRC_PATH + '/helen-dataset.dat')
_detector = dlib.get_frontal_face_detector()


# faceCalibration(cap: cv2.VideoCapture) -> RawFaceData {{{
def faceCalibration(cap: cv2.VideoCapture) -> Tuple[RawFaceData, float]:
    """Calibrate individuals' differences.

    What this function does are:
        1. Get distance between eyes
    """
    print("=========== Face calibration ===========")
    input("Please face front and press enter:")
    frame = _waitUntilFaceDetect(cap)
    print("got your face... wait for a second...")
    face, ratio = Face.fromDPointsWithRatio(facemark(frame))
    print("done :)")
    return (RawFaceData.get(face), ratio)
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
    # Find biggest face from image
    rects: dlib.rectangles = _detector(gray_img, 1)

    wholeFace: List[dlib.dpoints] = []
    for rect in rects:
        parts: dlib.dpoints =\
            _points2dpoints(_predictor(gray_img, rect).parts())
        wholeFace.append(parts)

    if len(wholeFace) == 0:
        return None

    absolute_coord = _getBiggestFace(wholeFace)
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

    raise CapHasClosedError(ExitCode.FILE_FACEDETECTION)
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


# _points2dpoints(ps: dlib.points) -> dlib.dpoints: {{{
def _points2dpoints(ps: dlib.points) -> dlib.dpoints:
    """convert dlib.points object to dlib.dpoints object.
        All points() are should be converted to dpoints,
        as we use float values
    """
    ret: dlib.dpoints = dlib.dpoints()
    for p in ps:
        ret.append(_point2dpoint(p))

    return ret
# }}}


# def _point2dpoint(p: dlib.point) -> dlib.dpoint: {{{
def _point2dpoint(p: dlib.point) -> dlib.dpoint:
    """convert dlib.point object to dlib.dpoint object.
    """
    return dlib.dpoint(float(p.x), float(p.y))
# }}}


# _toRelative(target: dlib.dpoints, center: dlib.dpoint) -> dlib.dpoints: {{{
def _toRelative(target: dlib.dpoints, center: dlib.dpoint) -> dlib.dpoints:
    """ convert target into relative coordinates
        Only Center will be left as is.
    """
    # convert all points to relative
    converted = list(map(lambda p: p - center, target))
    # center position holds absolute coordinate
    converted[49] = center
    return dlib.dpoints(converted)
# }}}
