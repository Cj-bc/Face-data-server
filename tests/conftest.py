import sys
from pathlib import Path
import cv2
import math
import numpy as np
import hypothesis.strategies as st
sys.path.insert(0, str(Path(__file__).parent.parent))
from FaceDataServer.Types import (Cv2Image, Coord, Part
                                 , AbsoluteCoord, RelativeCoord  # noqa: E402
                                 , Face, Eye, Mouth, Nose, EyeBrow)

faceFrame: Cv2Image = cv2.imread('tests/src/face.jpg')
noFaceFrame: Cv2Image = cv2.imread('tests/src/noface.png')


finiteFloatCallable = st.floats(allow_infinity=False, allow_nan=False
                               , min_value=0.0, max_value=1.0e4, width=16)

CoordStrategies = st.builds(Coord, finiteFloatCallable, finiteFloatCallable)
AbsoluteCoordStrategies = st.builds(AbsoluteCoord
                                   , finiteFloatCallable, finiteFloatCallable)
RelativeCoordStrategies = st.builds(RelativeCoord
                                   , finiteFloatCallable, finiteFloatCallable)
PartStrategies = st.builds(Part, CoordStrategies, CoordStrategies
                          , CoordStrategies, CoordStrategies)
NoseStrategies = st.builds(Nose, CoordStrategies
                          , CoordStrategies, CoordStrategies)
FaceStrategies = st.builds(Face, AbsoluteCoordStrategies
                          , RelativeCoordStrategies
                          , RelativeCoordStrategies, RelativeCoordStrategies
                          , st.builds(Eye, CoordStrategies, CoordStrategies
                                     , CoordStrategies, CoordStrategies)
                          , st.builds(Eye, CoordStrategies, CoordStrategies
                                     , CoordStrategies, CoordStrategies)
                          , st.builds(Mouth, CoordStrategies, CoordStrategies
                                     , CoordStrategies, CoordStrategies)
                          , st.builds(Nose, CoordStrategies, CoordStrategies
                                     , CoordStrategies)
                          , st.builds(EyeBrow, CoordStrategies, CoordStrategies
                                     , CoordStrategies, CoordStrategies)
                          , st.builds(EyeBrow, CoordStrategies, CoordStrategies
                                     , CoordStrategies, CoordStrategies))


face_front = Face(AbsoluteCoord(0, 0), RelativeCoord(-1, 0)
                   , RelativeCoord(1, 0), RelativeCoord(0, -1)
                   , Eye(Coord(0.4, 0.6), Coord(0, 0)
                        , Coord(0, 0), Coord(0.12, 0.83))
                   , Eye(Coord(-0.4, 0.6), Coord(0, 0)
                        , Coord(-0.12, 0.83), Coord(0, 0))
                   , Mouth(Coord(0, 0), Coord(0, 0), Coord(0, 0), Coord(0, 0))
                   , Nose(Coord(0, 0), Coord(0, 0), Coord(0, 0))
                   , EyeBrow(Coord(0, 0), Coord(0, 0)
                            , Coord(0, 0), Coord(0.2, 1))
                   , EyeBrow(Coord(0, 0), Coord(0, 0)
                            , Coord(-0.2, 1), Coord(0, 0)))
face_right = Face(AbsoluteCoord(-3, 0), RelativeCoord(1, 1)
                   , RelativeCoord(-1, 1), RelativeCoord(-1, -1)
                   , Eye(Coord(0.25, 0.6), Coord(0, 0)
                        , Coord(0, 0), Coord(0.1, 0.83))
                   , Eye(Coord(-0.24, 0.6), Coord(0, 0)
                        , Coord(-0.12, 0.83), Coord(0, 0))
                   , Mouth(Coord(0, 0), Coord(0, 0), Coord(0, 0), Coord(0, 0))
                   , Nose(Coord(0, 0), Coord(0, 0), Coord(0, 0))
                   , EyeBrow(Coord(0, 0), Coord(0, 0)
                            , Coord(0, 0), Coord(0, 1))
                   , EyeBrow(Coord(0, 0), Coord(0, 0)
                            , Coord(-0.25, 1), Coord(0, 0)))
face_left = Face(AbsoluteCoord(3, 0), RelativeCoord(1, 1)
                  , RelativeCoord(-1, 1), RelativeCoord(0.1, -1)
                  , Eye(Coord(0.4, 0.6), Coord(0, 0)
                       , Coord(0, 0), Coord(0.12, 0.83))
                  , Eye(Coord(-0.2, 0.6), Coord(0, 0)
                       , Coord(-0.1, 0.83), Coord(0, 0))
                  , Mouth(Coord(0, 0), Coord(0, 0), Coord(0, 0), Coord(0, 0))
                  , Nose(Coord(0, 0), Coord(0, 0), Coord(0, 0))
                  , EyeBrow(Coord(0, 0), Coord(0, 0)
                           , Coord(0, 0), Coord(0.2, 1))
                  , EyeBrow(Coord(0, 0), Coord(0, 0)
                           , Coord(0, 1), Coord(0, 0)))
face_upside = Face(AbsoluteCoord(0, 5), RelativeCoord(1, 1)
                    , RelativeCoord(-1, 1), RelativeCoord(0, -1)
                    , Eye(Coord(0.4, 0.3), Coord(0, 0)
                         , Coord(0, 0), Coord(0.12, 1))
                    , Eye(Coord(-0.4, 0.3), Coord(0, 0)
                         , Coord(-0.12, 1), Coord(0, 0))
                    , Mouth(Coord(0, 0), Coord(0, 0), Coord(0, 0), Coord(0, 0))
                    , Nose(Coord(0, 0), Coord(0, 0), Coord(0, 0))
                    , EyeBrow(Coord(0, 0), Coord(0, 0)
                             , Coord(0, 0), Coord(0.2, 1))
                    , EyeBrow(Coord(0, 0), Coord(0, 0)
                             , Coord(-0.2, 1), Coord(0, 0)))
face_bottom = Face(AbsoluteCoord(0, -10), RelativeCoord(1, 1)
                    , RelativeCoord(-1, 1), RelativeCoord(0, -1)
                    , Eye(Coord(0.4, 0), Coord(0, 0)
                         , Coord(0, 0), Coord(0.12, 0.3))
                    , Eye(Coord(-0.4, 0), Coord(0, 0)
                         , Coord(-0.12, 0.3), Coord(0, 0))
                    , Mouth(Coord(0, 0), Coord(0, 0), Coord(0, 0), Coord(0, 0))
                    , Nose(Coord(0, 0), Coord(0, 0), Coord(0, 0))
                    , EyeBrow(Coord(0, 0), Coord(0, 0)
                             , Coord(0, 0), Coord(0.2, 1))
                    , EyeBrow(Coord(0, 0), Coord(0, 0)
                             , Coord(-0.2, 1), Coord(0, 0)))


def _rotateCoord(c, theta):
    """ rotate Coord for theta
        just a tiny helper function to generate leaned faces
    """
    rotateMatrix = np.matrix([[np.cos(theta), np.sin(-theta)]
                             , [np.sin(theta), np.cos(theta)]])
    ans = rotateMatrix * np.matrix([[c.x], [c.y]])
    return c.__class__(ans.tolist()[0][0], ans.tolist()[1][0])


face_lean_left = Face(AbsoluteCoord(0, 0)
                       , _rotateCoord(RelativeCoord(1, 1), math.pi / 4)
                       , _rotateCoord(RelativeCoord(-1, 1), math.pi / 4)
                       , _rotateCoord(RelativeCoord(0, -1), math.pi / 4)
                       , Eye(_rotateCoord(Coord(0.4, 0.6), math.pi / 4)
                            , Coord(0, 0)
                            , Coord(0, 0)
                            , _rotateCoord(Coord(0.12, 0.83), math.pi / 4))
                       , Eye(_rotateCoord(Coord(-0.4, 0.6), math.pi / 4)
                            , Coord(0, 0)
                            , _rotateCoord(Coord(-0.12, 0.83), math.pi / 4)
                            , Coord(0, 0))
                       , Mouth(Coord(0, 0) , Coord(0, 0)
                              , Coord(0, 0) , Coord(0, 0))
                       , Nose(Coord(0, 0) , Coord(0, 0) , Coord(0, 0))
                       , EyeBrow(Coord(0, 0)
                            , Coord(0, 0)
                            , Coord(0, 0)
                            , _rotateCoord(Coord(0.2, 1), math.pi / 4))
                       , EyeBrow(Coord(0, 0)
                            , Coord(0, 0)
                            , _rotateCoord(Coord(-0.2, 1), math.pi / 4)
                            , Coord(0, 0)))
face_lean_right = Face(AbsoluteCoord(0, 0)
                       , _rotateCoord(RelativeCoord(1, 1), -(math.pi / 4))
                       , _rotateCoord(RelativeCoord(-1, 1), -(math.pi / 4))
                       , _rotateCoord(RelativeCoord(0, -1), -(math.pi / 4))
                       , Eye(_rotateCoord(Coord(0.4, 0.6), -(math.pi / 4))
                            , Coord(0, 0)
                            , Coord(0, 0)
                            , _rotateCoord(Coord(0.12, 0.83), -(math.pi / 4)))
                       , Eye(_rotateCoord(Coord(-0.4, 0.6), -(math.pi / 4))
                            , Coord(0, 0)
                            , _rotateCoord(Coord(-0.12, 0.83), -(math.pi / 4))
                            , Coord(0, 0))
                       , Mouth(Coord(0, 0) , Coord(0, 0)
                              , Coord(0, 0) , Coord(0, 0))
                       , Nose(Coord(0, 0) , Coord(0, 0) , Coord(0, 0))
                       , EyeBrow(Coord(0, 0)
                            , Coord(0, 0)
                            , Coord(0, 0)
                            , _rotateCoord(Coord(0.2, 1), -(math.pi / 4)))
                       , EyeBrow(Coord(0, 0)
                            , Coord(0, 0)
                            , _rotateCoord(Coord(-0.2, 1), -(math.pi / 4))
                            , Coord(0, 0)))


def round_Coord(s: Coord) -> str:
    return format(s.x, '.12g') + " " + format(s.y, '.12g')


def round_Part(s: Part) -> str:
    return round_Coord(s.top) + " " \
        + round_Coord(s.bottom) + " " \
        + round_Coord(s.leftSide) + " " \
        + round_Coord(s.rightSide)


def round_Face(s: Face) -> str:
    return round_Coord(s.center) + " " \
        + round_Coord(s.leftTemple) + " " \
        + round_Coord(s.rightTemple) + " " \
        + round_Coord(s.chinCenter) + " " \
        + round_Part(s.leftEye) + " " \
        + round_Part(s.rightEye) + " " \
        + round_Part(s.leftEyeBrow) + " " \
        + round_Part(s.rightEyeBrow)


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

    def release(self) -> None:
        pass
