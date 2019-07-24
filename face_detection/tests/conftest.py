import sys
import os
import numpy
import cv2
import dlib
from typing import List

sys.path.append(os.getcwd())

from Types import Cv2Image

faceFrame: Cv2Image = cv2.imread('tests/src/face.jpg')
noFaceFrame: Cv2Image = cv2.imread('tests/src/noface.png')


def _constructLandmark(tin_center, left_eye_r, left_eye_bottom,
                       right_eye_l, right_eye_bottom):
    ls = [(0, 0)] * 19 + [tin_center] + [(0, 0)] * 94 + [left_eye_r] +\
         [(0, 0)] * 14 + [left_eye_bottom] + [(0, 0)] * 15 +\
         [right_eye_l] + [(0, 0)] * 13 + [right_eye_bottom] +\
         [(0, 0)] * 44
    l_points = list(map(lambda n: dlib.point(n[0], n[1]), ls))
    return constructPoints(l_points)


points_front = _constructLandmark((0, -50), (5, 50), (10, 20), (-5, 50),
                                  (-10, 20))
points_right = _constructLandmark((0, -50), (0, 50), (5, 20), (-5, 50),
                                  (-10, 20))
points_left = _constructLandmark((0, -50), (5, 50), (10, 20), (0, 50),
                                 (-5, 20))
points_upside = _constructLandmark((0, -40), (5, 30), (10, 10), (-5, 30),
                                   (-10, 10))
points_bottom = _constructLandmark((0, -60), (5, 30), (10, 0), (-5, 30),
                                   (-10, 0))
# points_lean_leftUp
# points_lean_rightUp


class MockedCap():
    opened: bool = True
    frame: Cv2Image = noFaceFrame

    def __init__(self, opened: bool, frame: Cv2Image) -> None:
        self.opened = opened
        self.frame = frame

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> Cv2Image:
        ret = True if self.frame.all() == faceFrame.all() else False
        return (ret, self.frame)


def constructPoints(ps: List[dlib.point]) -> dlib.points:
    """ helper function.
        Construct dlib.points from list of dlib.point
    """
    ret = dlib.points()
    for p in ps:
        ret.append(p)

    return ret
