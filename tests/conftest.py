import sys
from pathlib import Path
import cv2
import dlib
import math
import hypothesis.strategies as st
from typing import List
sys.path.insert(0, str(Path(__file__).parent.parent))
from FaceDataServer.Types import (Cv2Image, Coord, Part, AbsoluteCoord, RelativeCoord  # noqa: E402
                                 , Face, Eye, Mouth, Nose, EyeBrow)

faceFrame: Cv2Image = cv2.imread('tests/src/face.jpg')
noFaceFrame: Cv2Image = cv2.imread('tests/src/noface.png')


finiteFloatCallable = st.floats(allow_infinity=False, allow_nan=False
                               , min_value=0.0, max_value=1.0e4, width=16)

CoordStrategies = st.builds(Coord, finiteFloatCallable, finiteFloatCallable)
AbsoluteCoordStrategies = st.builds(AbsoluteCoord, finiteFloatCallable, finiteFloatCallable)
RelativeCoordStrategies = st.builds(RelativeCoord, finiteFloatCallable, finiteFloatCallable)
PartStrategies = st.builds(Part, CoordStrategies, CoordStrategies
                          , CoordStrategies, CoordStrategies)


def constructPoints(ps: List[dlib.dpoint]) -> dlib.dpoints:
    """ helper function.
        Construct dlib.points from list of dlib.point
    """
    ret = dlib.dpoints()
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
    l_points = list(map(lambda n: dlib.dpoint(n[0], n[1]), ls))
    return constructPoints(l_points)


def _leanFace(dpoints: dlib.dpoints, angle: float) -> dlib.dpoints:
    """rotate given 'dpoints' for 'angle' degree.
    """
    def _leanOnePoint(p: dlib.dpoint) -> dlib.dpoint:
        # As X/Y axis are swapped, sin/cos are also swapped
        return dlib.dpoint(p.x * math.cos(angle),
                           p.y * math.sin(angle))

    _leanedDPointsL = list(map(_leanOnePoint, dpoints))
    return dlib.dpoints(_leanedDPointsL)


points_front = Face(AbsoluteCoord(0, 0), RelativeCoord(25, 30)
                   , RelativeCoord(-25, 30), RelativeCoord(0, -50)
                   , Eye(Coord(10, 20), Coord(None, None), Coord(-3, 25), Coord(3, 25))
                   , Eye(Coord(5, 50), Coord(None, None), Coord(-10, 20), Coord(-5, 50))
                   , Mouth(Coord(None, None), Coord(None, None), Coord(None, None), Coord(None, None))
                   , Nose(Coord(0, 0), Coord(None, None), Coord(None, None))
                   , EyeBrow(Coord(None, None), Coord(None, None), Coord(None, None), Coord(5, 50))
                   , EyeBrow(Coord(None, None), Coord(None, None), Coord(-5, 50), Coord(None, None)))
points_right = Face(AbsoluteCoord(0, 0), RelativeCoord(20, 30)
                   , RelativeCoord(-25, 30), RelativeCoord(0, -50)
                   , Eye(Coord(5, 20), Coord(None, None), Coord(None, None), Coord(2, 25))
                   , Eye(Coord(None, None), Coord(None, None), Coord(-3, 25), Coord(None, None))
                   , Mouth(Coord(None, None), Coord(None, None), Coord(None, None), Coord(None, None))
                   , Nose(Coord(None, None), Coord(None, None), Coord(None, None))
                   , EyeBrow(Coord(None, None), Coord(None, None), Coord(None, None), Coord(0, 50))
                   , EyeBrow(Coord(None, None), Coord(None, None), Coord(-5, 50), Coord(None, None)))
points_left = Face(AbsoluteCoord(0, 0), RelativeCoord(25, 30)
                  , RelativeCoord(-20, 30), RelativeCoord(0, -50)
                  , Eye(Coord(10, 20), Coord(None, None), Coord(None, None), Coord(3, 25))
                  , Eye(Coord(-5, 20), Coord(None, None), Coord(2, 25), Coord(None, None))
                  , Mouth(Coord(None, None), Coord(None, None), Coord(None, None), Coord(None, None))
                  , Nose(Coord(None, None), Coord(None, None), Coord(None, None))
                  , EyeBrow(Coord(None, None), Coord(None, None), Coord(None, None), Coord(5, 50))
                  , EyeBrow(Coord(None, None), Coord(None, None), Coord(0, 50), Coord(None, None)))
points_upside = Face(AbsoluteCoord(0, 0), RelativeCoord(25, 30)
                    , RelativeCoord(-25, 30), RelativeCoord(0, -40)
                    , Eye(Coord(10, 10), Coord(None, None), Coord(None, None), Coord(3, 30))
                    , Eye(Coord(-10, 10), Coord(None, None), Coord(-3, 30), Coord(None, None))
                    , Mouth(Coord(None, None), Coord(None, None), Coord(None, None), Coord(None, None))
                    , Nose(Coord(None, None), Coord(None, None), Coord(None, None))
                    , EyeBrow(Coord(None, None), Coord(None, None), Coord(None, None), Coord(5, 50))
                    , EyeBrow(Coord(None, None), Coord(None, None), Coord(-5, 50), Coord(None, None)))
points_bottom = Face(AbsoluteCoord(0, 0), RelativeCoord(25, 30)
                    , RelativeCoord(-25, 30), RelativeCoord(0, -60)
                    , Eye(Coord(10, 0), Coord(None, None), Coord(None, None), Coord(3, 10))
                    , Eye(Coord(-10, 0), Coord(None, None), Coord(-3, 10), Coord(None, None))
                    , Mouth(Coord(None, None), Coord(None, None), Coord(None, None), Coord(None, None))
                    , Nose(Coord(None, None), Coord(None, None), Coord(None, None))
                    , EyeBrow(Coord(None, None), Coord(None, None), Coord(None, None), Coord(5, 30))
                    , EyeBrow(Coord(None, None), Coord(None, None), Coord(-5, 30), Coord(None, None)))

points_lean_left = _leanFace(points_front, math.pi / 4)
points_lean_right = _leanFace(points_front, -(math.pi / 4))


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
