import sys
import os
from pathlib import Path
import numpy
import cv2
import dlib
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

import main
from faceDetection.Types import Cv2Image

faceFrame: Cv2Image = cv2.imread('tests/src/face.jpg')
noFaceFrame: Cv2Image = cv2.imread('tests/src/noface.png')


def constructPoints(ps: List[dlib.point]) -> dlib.points:
    """ helper function.
        Construct dlib.points from list of dlib.point
    """
    ret = dlib.points()
    for p in ps:
        ret.append(p)

    return ret


def _constructLandmark(side_right, tin_center, side_left, left_eye_r,
                       left_eye_bottom, right_eye_l, right_eye_bottom,
                       eyebrow_left_r, eyebrow_right_l):
    ls = [side_right] + [(0, 0)] * 18 + [tin_center] + [(0, 0)] * 20 +\
         [side_left] + [(0, 0)] * 73 + [left_eye_r] +\
         [(0, 0)] * 14 + [left_eye_bottom] + [(0, 0)] * 5 +\
         [right_eye_l] + [(0, 0)] * 13 + [right_eye_bottom] +\
         [(0, 0)] * 4 + [eyebrow_left_r] + [(0, 0)] * 19 +\
         [eyebrow_right_l] + [(0, 0)] * 21
    l_points = list(map(lambda n: dlib.point(n[0], n[1]), ls))
    return constructPoints(l_points)


points_front = _constructLandmark((-25, 30), (0, -50), (25, 30), (3, 25),
                                  (10, 20), (-3, 25), (-10, 20),
                                  (5, 50), (-5, 50))
points_right = _constructLandmark((-25, 30), (0, -50), (20, 30), (-2, 25),
                                  (5, 20), (-3, 25), (-10, 20),
                                  (0, 50), (-5, 50))
points_left = _constructLandmark((-20, 30), (0, -50), (25, 30), (3, 25),
                                 (10, 20), (2, 25), (-5, 20),
                                 (5, 50), (0, 50))
points_upside = _constructLandmark((-25, 30), (0, -40), (25, 30), (3, 30),
                                   (10, 10), (-3, 30), (-10, 10),
                                   (5, 50), (-5, 50))
points_bottom = _constructLandmark((-25, 30), (0, -60), (25, 30), (3, 10),
                                   (10, 0), (-3, 10), (-10, 0),
                                   (5, 30), (-5, 30))
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


