from typing import NewType, TypeVar, Union
import numpy
import dlib
import dataclasses
import math


# Those values are defined based on this site image:
#   https://qiita.com/kekeho/items/0b2d4ed5192a4c90a0ac
# ignore
LANDMARK_NUM = {"LEFT_TEMPLE": 0
               , "TIN_CENTER": 19
               , "RIGHT_TEMPLE": 40
               , "NOSE_R": 44
               , "NOSE_BOTTOM": 49
               , "NOSE_L": 54
               , "MOUSE_R": 58
               , "MOUSE_TOP": 65
               , "MOUSE_L": 71
               , "MOUSE_BOTTOM": 79
               , "LEFT_EYE_R": 114
               , "LEFT_EYE_TOP": 120
               , "LEFT_EYE_L": 124
               , "LEFT_EYE_BOTTOM": 129
               , "RIGHT_EYE_L": 135
               , "RIGHT_EYE_TOP": 140
               , "RIGHT_EYE_R": 145
               , "RIGHT_EYE_BOTTOM": 149
               , "EYEBROW_LEFT_R": 154
               , "EYEBROW_LEFT_TOP": 159
               , "EYEBROW_LEFT_L": 164
               , "EYEBROW_LEFT_BOTTOM": 169
               , "EYEBROW_RIGHT_L": 174
               , "EYEBROW_RIGHT_TOP": 179
               , "EYEBROW_RIGHT_R": 185
               , "EYEBROW_RIGHT_BOTTOM": 190
                }

# type aliases {{{

Error = NewType('Error', str)
Cv2Image = numpy.ndarray
S = TypeVar('S')
Num = Union[int, float]
# }}}


# Coordinates {{{
class Coord:
    """Base class to express Coordinates
        This is made to be converted from dlib.dpoint
    """
    x: float
    y: float

    def __repr__(self):
        return f"Coord({self.x}, {self.y})"

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self: S, other: S) -> S:
        return Coord(self.x - other.x, self.y - other.y)

    def __add__(self: S, other: S) -> S:
        return Coord(self.x + other.x, self.y + other.y)

    def __truediv__(self: S, other: Num) -> S:
        return Coord(self.x / other, self.y / other)

    @classmethod
    def default(cls):
        """return default coordinate."""
        return cls(0, 0)

    @classmethod
    def fromDPoint(cls: S, p: dlib.dpoint) -> S:
        return cls(p.x, p.y)


class AbsoluteCoord(Coord):
    def __sub__(self: S, other: S) -> S:
        return AbsoluteCoord(self.x - other.x, self.y - other.y)


class RelativeCoord(Coord):
    def __sub__(self: S, other: S) -> S:
        return RelativeCoord(self.x - other.x, self.y - other.y)

    def fromCoord(c: Coord) -> S:
        return RelativeCoord(c.x, c.y)

# }}}


# Each face parts {{{
class Part():
    bottom: Coord
    top: Coord
    leftSide: Coord
    rightSide: Coord

    def __repr__(self):
        return f"Part({self.bottom}, {self.top}, \
                 {self.leftSide}, {self.rightSide})"

    def __init__(self, b, t, l, r):
        def _coord(c):
            if type(c) == Coord:
                return c
            elif type(c) == dlib.dpoint:
                return Coord.fromDPoint(c)
            else:
                raise TypeError

        self.bottom = _coord(b)
        self.top = _coord(t)
        self.leftSide = _coord(l)
        self.rightSide = _coord(r)

    def __sub__(self: S, other: S) -> S:
        return Part(self.bottom - other.bottom
                   , self.top - other.top
                   , self.leftSide - other.leftSide
                   , self.rightSide - other.rightSide)

    def __add__(self: S, other: S) -> S:
        return Part(self.bottom + other.bottom
                   , self.top + other.top
                   , self.leftSide + other.leftSide
                   , self.rightSide + other.rightSide)

    def __truediv__(self: S, other: Num) -> S:
        return Part(self.bottom / other , self.top / other
                   , self.leftSide / other , self.rightSide / other)

    @staticmethod
    def default(cls):
        """return default coordinate."""
        return cls(0, 0, 0, 0)


class Eye(Part):
    pass


class Mouth(Part):
    pass


class Nose(Part):
    top = None

    def __init__(self: S, b: Coord, l: Coord, r: Coord) -> S:
        self.bottom = b
        self.leftSide = l
        self.rightSide = r

        return self


class EyeBrow(Part):
    pass
# }}}


class Face:
    center: AbsoluteCoord
    leftTemple: RelativeCoord
    rightTemple: RelativeCoord
    tinCenter: RelativeCoord
    leftEye: Eye
    rightEye: Eye
    mouth: Mouth
    nose: Nose
    leftEyeBrow: EyeBrow
    rightEyeBrow: EyeBrow

    @classmethod
    def default(cls):
        """return default coordinate."""
        return cls(AbsoluteCoord.default(), RelativeCoord.default()
                  , RelativeCoord.default(), RelativeCoord.default()
                  , Eye.default(), Eye.default()
                  , Mouth.default(), Nose.default(), EyeBrow.default()
                  , EyeBrow.default())

    @classmethod
    def fromDPoints(cls: S, points: dlib.dpoints) -> S:
        """return 'Face' object based on given 'facemark'"""
        _center = AbsoluteCoord.fromDPoint(points[LANDMARK_NUM["NOSE_BOTTOM"]])
        _ltemp  = RelativeCoord.fromDPoint(points[LANDMARK_NUM["TEMPLE_LEFT"]])
        _rtemp  = RelativeCoord.fromDPoint(points[LANDMARK_NUM["TEMPLE_RIGHT"]]) # noqa
        _tin    = RelativeCoord.fromDPoint(points[LANDMARK_NUM["TIN_CENTER"]])
        _leye   = Eye(points[LANDMARK_NUM["LEFT_EYE_BOTTOM"]]
                     , points[LANDMARK_NUM["LEFT_EYE_TOP"]]
                     , points[LANDMARK_NUM["LEFT_EYE_L"]]
                     , points[LANDMARK_NUM["LEFT_EYE_R"]])
        _reye   = Eye(points[LANDMARK_NUM["RIGHT_EYE_BOTTOM"]]
                     , points[LANDMARK_NUM["RIGHT_EYE_TOP"]]
                     , points[LANDMARK_NUM["RIGHT_EYE_L"]]
                     , points[LANDMARK_NUM["RIGHT_EYE_R"]])
        _mouth  = Mouth(points[LANDMARK_NUM["MOUSE_BOTTOM"]]
                       , points[LANDMARK_NUM["MOUSE_TOP"]]
                       , points[LANDMARK_NUM["MOUSE_L"]]
                       , points[LANDMARK_NUM["MOUSE_R"]])
        _nose   = Nose(points[LANDMARK_NUM["NOSE_BOTTOM"]]
                      , points[LANDMARK_NUM["NOSE_L"]]
                      , points[LANDMARK_NUM["NOSE_R"]])
        _leb    = EyeBrow(points[LANDMARK_NUM["EYEBROW_LEFT_BOTTOM"]]
                         , points[LANDMARK_NUM["EYEBROW_LEFT_TOP"]]
                         , points[LANDMARK_NUM["EYEBROW_LEFT_L"]]
                         , points[LANDMARK_NUM["EYEBROW_LEFT_R"]])
        _reb    = EyeBrow(points[LANDMARK_NUM["EYEBROW_RIGHT_BOTTOM"]]
                         , points[LANDMARK_NUM["EYEBROW_RIGHT_TOP"]]
                         , points[LANDMARK_NUM["EYEBROW_RIGHT_L"]]
                         , points[LANDMARK_NUM["EYEBROW_RIGHT_R"]])

        return cls(_center, _ltemp, _rtemp, _tin, _leye, _reye
                  , _mouth, _nose, _leb, _reb)


@dataclasses.dataclass(frozen=True)
class RawFaceData:
    eyeDistance: float
#    rightEyeSize: float
#    leftEyeSize: float
    faceHeigh: float
    faceCenter: dlib.dpoint

    @classmethod
    def get(cls: S, face: Face) -> S:
        """ Return RawFaceData from dlib.points
        """
        eyeDistance  = abs(face.rightEye.leftSide.x - face.leftEye.rightSide.x)

        _middleForehead = (face.leftEyeBrow.rightSide
                          + face.rightEyeBrow.leftSide) / 2
        _faceHeighVector  = _middleForehead\
                            - Coord.fromDPoint(face.tinCenter)
        faceHeigh = math.sqrt(_faceHeighVector.x ** 2
                              + _faceHeighVector.y ** 2)
        faceCenter = face.center
        return cls(eyeDistance, faceHeigh, faceCenter)

    def thresholded(self, t):
        """Force eyeDistance / faceHeigh to be smaller than threshold
        """
        eD = min(self.eyeDistance, t.eyeDistance)
        fH = min(self.faceHeigh, t.faceHeigh)
        return RawFaceData(eD, fH, self.faceCenter)


@dataclasses.dataclass(frozen=True)
class FaceRotations:
    x: float
    y: float
    z: float

    @classmethod
    def get(cls: S, face: Face, calib: RawFaceData) -> S:
        """ calculate face rotations from calibration data and landmark
        """
        eyeLineVector = face.rightEye.bottom - face.leftEye.bottom
        raw = RawFaceData.get(face).thresholded(calib)
        # those values are used in the near future.Just ignore this for linting
        leftEdge2Center  = face.leftTemple - raw.faceCenter # noqa
        rightEdge2Center = raw.faceCenter - face.rightTemple # noqa
        chin2Center = raw.faceCenter - Coord.fromDpoint(face.tinCenter) # noqa

        # TODO: how can I notice which side does face face to?
        #       I can't simply compare eyes sizes, 'cus sometimes
        #       user might wink. In that case, I can't recognize properly.
        degreeY = math.acos(raw.eyeDistance / calib.eyeDistance)
        degreeX = math.acos(raw.faceHeigh / calib.faceHeigh)
        degreeZ = math.atan(eyeLineVector.y / eyeLineVector.x)
        # TODO: ^ This some times got error 'Division by 0'

        rotateX = degreeX if raw.faceCenter.y > calib.faceCenter.y\
                            else -1 * degreeX
        rotateY = degreeY if raw.faceCenter.x > calib.faceCenter.x\
                            else -1 * degreeY
        # v Is this correct code? v
        rotateZ = degreeZ
        return cls(rotateX, rotateY, rotateZ)


class FaceDetectionError(Exception):
    """Base class for exceptions in this module"""
    pass


class CapHasClosedError(FaceDetectionError):
    """Exception raised for unexpected cv2.VideoCapture close"""
    def __str__(self):
        return "The camera connection has been closed. Please try again"
