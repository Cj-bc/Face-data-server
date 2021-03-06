from typing import NewType, TypeVar, Union, Tuple
import numpy
import dlib
import dataclasses
import math
from math import pi
import struct

majorVersionNum = 1
minorVersionNum = 0
defaultGroupAddr = "226.70.68.83"
defaultPortNumber = 5032


# Those values are defined based on this site image:
#   https://qiita.com/kekeho/items/0b2d4ed5192a4c90a0ac
# ignore
LANDMARK_NUM = {"TEMPLE_LEFT": 0
               , "CHIN_CENTER": 19
               , "TEMPLE_RIGHT": 40
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


# ExitCode {{{
class ExitCode():
    """
    Exit Codes are constructed with these pattern:
        0b00000000
          +==~~---
          || | |---- Number in that 'kind'
          || |------ Specify kind of error
          ||-------- Specify which file is that
          |--------- MUST BE 0, as exit code 126+ is treated as fatal error
    """
    Ok = 0
    FILE_MAIN          = 0b00100000
    FILE_TYPE          = 0b01000000
    FILE_FACEDETECTION = 0b01100000

    ERR_UNKNOWN = 0b00001000
    ERR_IO      = 0b00010000

    CameraNotFound = ERR_IO | 0b00000001
    ServerIsStillUsed = ERR_IO | 0b00000010
# }}}


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

    def __eq__(self: S, other: S) -> S:
        return self.x == other.x and self.y == other.y

    def __neg__(self: S) -> S:
        return self.__class__(-self.x, -self.y)

    def __add__(self: S, other: S) -> S:
        return self.__class__(self.x + other.x, self.y + other.y)

    def __sub__(self: S, other: S) -> S:
        return self + (-other)

    def __mul__(self: S, other: Num) -> S:
        return self.__class__(self.x * other, self.y * other)

    def __truediv__(self: S, other: Num) -> S:
        return self.__class__(self.x / other, self.y / other)
        # I don't know why but the expr below won't work correctly
        # return self * (1 / other)

    @classmethod
    def default(cls):
        """return default coordinate."""
        return cls(0, 0)

    @classmethod
    def fromDPoint(cls: S, p: dlib.dpoint) -> S:
        return cls(p.x, p.y)

    def toTuple(self):
        return (self.x, self.y)

    def map(self: S, f):
        self.x = f(self.x)
        self.y = f(self.y)
        return self


class AbsoluteCoord(Coord):
    def __repr__(self):
        return f"AbsoluteCoord({self.x}, {self.y})"

    def fromCoord(c: Coord) -> S:
        return AbsoluteCoord(c.x, c.y)


class RelativeCoord(Coord):
    def __repr__(self):
        return f"RelativeCoord({self.x}, {self.y})"

    def fromCoord(c: Coord) -> S:
        return RelativeCoord(c.x, c.y)
# }}}


# Each face parts {{{
class Part():
    bottom: Coord
    top: Coord
    leftSide: Coord
    rightSide: Coord

    def __eq__(self: S, other: S) -> bool:
        return self.bottom == other.bottom and \
            self.top == other.top and \
            self.leftSide == other.leftSide and \
            self.rightSide == other.rightSide

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

    def __neg__(self: S) -> S:
        return self.__class__(-self.bottom , -self.top
                   , -self.leftSide , -self.rightSide)

    def __add__(self: S, other: S) -> S:
        return self.__class__(self.bottom + other.bottom
                   , self.top + other.top
                   , self.leftSide + other.leftSide
                   , self.rightSide + other.rightSide)

    def __sub__(self: S, other: S) -> S:
        return self + (-other)

    def __mul__(self: S, other: Num) -> S:
        return self.__class__(self.bottom * other , self.top * other
                   , self.leftSide * other , self.rightSide * other)

    def __truediv__(self: S, other: Num) -> S:
        return self.__class__(self.bottom / other , self.top / other
                   , self.leftSide / other , self.rightSide / other)
        # I don't know why but the expr below won't work correctly
        # return self * (1 / other)

    def map(self: S, f):
        self.bottom    = self.bottom.map(f)
        self.top       = self.top.map(f)
        self.leftSide  = self.leftSide.map(f)
        self.rightSide = self.rightSide.map(f)
        return self

    @classmethod
    def default(cls):
        """return default coordinate."""
        return cls(Coord.default(), Coord.default()
                  , Coord.default(), Coord.default())


class Eye(Part):
    def __repr__(self):
        return f"Eye({self.bottom}, {self.top}"\
               f", {self.leftSide}, {self.rightSide})"


class Mouth(Part):
    def __repr__(self):
        return f"Mouth({self.bottom}, {self.top},"\
               f"{self.leftSide}, {self.rightSide})"


class Nose(Part):
    def __repr__(self):
        return f"Nose({self.bottom}, {self.top}"\
               f", {self.leftSide}, {self.rightSide})"

    def __init__(self, b, l, r):
        super().__init__(b, Coord.default(), l, r)

    def __neg__(self: S) -> S:
        return self.__class__(-self.bottom
                             , -self.leftSide , -self.rightSide)

    def __add__(self: S, other: S) -> S:
        return self.__class__(self.bottom + other.bottom
                             , self.leftSide + other.leftSide
                             , self.rightSide + other.rightSide)

    def __sub__(self: S, other: S) -> S:
        return self + (-other)

    def __mul__(self: S, other: Num) -> S:
        return self.__class__(self.bottom * other
                             , self.leftSide * other , self.rightSide * other)

    def __truediv__(self: S, other: Num) -> S:
        return self.__class__(self.bottom / other
                             , self.leftSide / other , self.rightSide / other)
        # I don't know why but the expr below won't work correctly
        # return self * (1 / other)

    @classmethod
    def default(cls):
        """return default coordinate."""
        return cls(Coord.default(), Coord.default(), Coord.default())

    def map(self: S, f):
        self.top       = self.top.map(f)
        self.leftSide  = self.leftSide.map(f)
        self.rightSide = self.rightSide.map(f)
        return self


class EyeBrow(Part):
    def __repr__(self):
        return f"EyeBrow({self.bottom}, {self.top}"\
               f", {self.leftSide}, {self.rightSide})"
# }}}


# Face {{{
class Face:
    center: AbsoluteCoord
    leftTemple: RelativeCoord
    rightTemple: RelativeCoord
    chinCenter: RelativeCoord
    leftEye: Eye
    rightEye: Eye
    mouth: Mouth
    nose: Nose
    leftEyeBrow: EyeBrow
    rightEyeBrow: EyeBrow

    def __init__(self: S, c: AbsoluteCoord, lt: RelativeCoord,
                 rt: RelativeCoord, tc: RelativeCoord, le: Eye,
                 re: Eye, m: Mouth, n: Nose, leb: EyeBrow,
                 reb: EyeBrow) -> None:
        self.center = c
        self.leftTemple = lt
        self.rightTemple = rt
        self.chinCenter = tc
        self.leftEye = le
        self.rightEye = re
        self.mouth = m
        self.nose = n
        self.leftEyeBrow = leb
        self.rightEyeBrow = reb

    def __repr__(self: S) -> str:
        return f"Face({self.center}, {self.leftTemple}"\
               f", {self.rightTemple}, {self.chinCenter}"\
               f", {self.leftEye}, {self.rightEye}"\
               f", {self.mouth}, {self.nose}"\
               f", {self.leftEyeBrow}, {self.rightEyeBrow})"

    def __eq__(self: S, other: S) -> bool:
        return self.center == other.center\
            and self.leftTemple == other.leftTemple \
            and self.rightTemple == other.rightTemple \
            and self.chinCenter == other.chinCenter \
            and self.leftEye == other.leftEye \
            and self.rightEye == other.rightEye \
            and self.mouth == other.mouth \
            and self.nose == other.nose \
            and self.leftEyeBrow == other.leftEyeBrow \
            and self.rightEyeBrow == other.rightEyeBrow

    def __mul__(s: S, o: Num) -> S:
        return Face(s.center * o, s.leftTemple * o, s.rightTemple * o
                   , s.chinCenter * o, s.leftEye * o, s.rightEye * o
                   , s.mouth * o, s.nose * o, s.leftEyeBrow * o
                   , s.rightEyeBrow * o)

    def __truediv__(s: S, o: Num) -> S:
        # I don't know why but the expr below won't work correctly
        # return s * (1 / o)
        return Face(s.center / o, s.leftTemple / o, s.rightTemple / o
                   , s.chinCenter / o, s.leftEye / o, s.rightEye / o
                   , s.mouth / o, s.nose / o, s.leftEyeBrow / o
                   , s.rightEyeBrow / o)

    @classmethod
    def defaultWithRatio(cls: S, defRatio: float) -> Tuple[S, float]:
        return (cls.default(), defRatio)

    @classmethod
    def default(cls):
        """return default coordinate."""
        return cls(AbsoluteCoord.default(), RelativeCoord.default()
                  , RelativeCoord.default(), RelativeCoord.default()
                  , Eye.default(), Eye.default()
                  , Mouth.default(), Nose.default(), EyeBrow.default()
                  , EyeBrow.default())

    @classmethod
    def fromDPointsWithRatio(cls: S, points: dlib.dpoints) -> Tuple[S, float]: # noqa
        """ Return '<Face y length> / <Face x length>' ratio
            along with 'fromDPoints' result
        """
        xLength = abs(points[LANDMARK_NUM["TEMPLE_LEFT"]].x
                        - points[LANDMARK_NUM["TEMPLE_RIGHT"]].x)
        yLength = abs(points[LANDMARK_NUM["EYEBROW_LEFT_TOP"]].x
                        - points[LANDMARK_NUM["CHIN_CENTER"]].x)
        ratio = yLength / xLength
        return (Face.fromDPoints(points), ratio)

    @classmethod
    def fromDPoints(cls: S, points: dlib.dpoints) -> S:
        """return 'Face' object based on given 'facemark'"""

        def _normalize(smallest: float, biggest: float, current: float) -> float: # noqa
            # move smallest to be 0
            movedCurrent = (-smallest) + current
            movedBiggest = (-smallest) + biggest
            between0to1  = movedCurrent / movedBiggest
            # make it between -100 to 100
            return (200 * between0to1) - 100

        def _normalizePoint(p: dlib.dpoint) -> dlib.dpoint:
            smallestX = points[LANDMARK_NUM["TEMPLE_LEFT"]].x
            biggestX  = points[LANDMARK_NUM["TEMPLE_RIGHT"]].x
            smallestY = points[LANDMARK_NUM["CHIN_CENTER"]].y
            biggestY  = points[LANDMARK_NUM["EYEBROW_LEFT_TOP"]].y
            return dlib.dpoint(_normalize(smallestX, biggestX, p.x)
                              , _normalize(smallestY, biggestY, p.y)
                               )

        def _point(name: str) -> dlib.dpoint:
            return _normalizePoint(points[LANDMARK_NUM[name]])

        _c     = AbsoluteCoord.fromDPoint(_point("NOSE_BOTTOM"))
        _ltmp  = RelativeCoord.fromDPoint(_point("TEMPLE_LEFT"))
        _rtmp  = RelativeCoord.fromDPoint(_point("TEMPLE_RIGHT"))
        _chin  = RelativeCoord.fromDPoint(_point("CHIN_CENTER"))
        _leye  = Eye(_point("LEFT_EYE_BOTTOM")
                    , _point("LEFT_EYE_TOP")
                    , _point("LEFT_EYE_L")
                    , _point("LEFT_EYE_R"))
        _reye  = Eye(_point("RIGHT_EYE_BOTTOM")
                    , _point("RIGHT_EYE_TOP")
                    , _point("RIGHT_EYE_L")
                    , _point("RIGHT_EYE_R"))
        _mouth = Mouth(_point("MOUSE_BOTTOM")
                      , _point("MOUSE_TOP")
                      , _point("MOUSE_L")
                      , _point("MOUSE_R"))
        _nose  = Nose(_point("NOSE_BOTTOM")
                     , _point("NOSE_L")
                     , _point("NOSE_R"))
        _leb   = EyeBrow(_point("EYEBROW_LEFT_BOTTOM")
                        , _point("EYEBROW_LEFT_TOP")
                        , _point("EYEBROW_LEFT_L")
                        , _point("EYEBROW_LEFT_R"))
        _reb   = EyeBrow(_point("EYEBROW_RIGHT_BOTTOM")
                        , _point("EYEBROW_RIGHT_TOP")
                        , _point("EYEBROW_RIGHT_L")
                        , _point("EYEBROW_RIGHT_R"))

        return cls(_c, _ltmp, _rtmp, _chin, _leye, _reye
                  , _mouth, _nose, _leb, _reb)

    def fixWithRatio(self: S, init: float, current: float):
        """ Fix Face values along with ratio

            init: initial ratio
            current: current ratio

            A 'ratio' is (faceHeigh / faceWidth)
            It usually be the same, but when we open mouth,
            it'll be different value as faceHeigh will be grater.
            This affects Parts percentages. So fix it with this.
            Related issue on Github: Cj-bc/Face-Data-Server #40
        """
        ratioMagnif = current / init
        surplus = lambda a: a * ratioMagnif # noqa
        self.center.y       /= ratioMagnif
        self.leftTemple.y   /= ratioMagnif
        self.rightTemple.y  /= ratioMagnif
        self.chinCenter.y   /= ratioMagnif
        self.leftEye        = self.leftEye.map(surplus)
        self.rightEye.y     = self.rightEye.map(surplus)
        self.mouth.y        = self.mouth.map(surplus)
        self.nose.y         = self.nose.map(surplus)
        self.leftEyeBrow.y  = self.leftEyeBrow.map(surplus)
        self.rightEyeBrow.y = self.rightEyeBrow.map(surplus)
# }}}


# RawFaceData {{{
@dataclasses.dataclass(frozen=True)
class RawFaceData:
    eyeDistance: float
    faceHeigh: float
    faceCenter: AbsoluteCoord
    mouthHeight: float
    mouthWidth: float
    leftEyeHeight: float
    rightEyeHeight: float

    @staticmethod
    def default() -> S:
        return RawFaceData(0.0, 0.0, AbsoluteCoord.default(), 0, 0, 0, 0)

    @classmethod
    def get(cls: S, face: Face) -> S:
        """ Return RawFaceData from dlib.points
        """
        _eyeVector  = face.leftEye.rightSide - face.rightEye.leftSide
        eyeDistance = round(math.sqrt(_eyeVector.x ** 2
                            + _eyeVector.y ** 2), 15)

        _middleForehead = (face.leftEyeBrow.rightSide
                          + face.rightEyeBrow.leftSide) / 2
        _faceHeighVector  = _middleForehead\
                            - face.chinCenter
        faceHeigh = round(math.sqrt(_faceHeighVector.x ** 2
                                   + _faceHeighVector.y ** 2)
                         , 15)

        return cls(eyeDistance
                  , faceHeigh
                  , face.center
                  , face.mouth.top.y - face.mouth.bottom.y
                  , abs(face.mouth.leftSide.x - face.mouth.rightSide.x)
                  , face.leftEye.top.y - face.leftEye.bottom.y
                  , face.rightEye.top.y - face.rightEye.bottom.y
                   )

    def thresholded(self, t):
        """Force eyeDistance / faceHeigh to be smaller than threshold
        """
        eD = min(self.eyeDistance, t.eyeDistance)
        fH = min(self.faceHeigh, t.faceHeigh)
        return RawFaceData(eD, fH, self.faceCenter
                          , self.mouthHeight
                          , self.mouthWidth
                          , self.leftEyeHeight
                          , self.rightEyeHeight
                           )

# }}}


# Exceptions {{{
class FaceDetectionError(Exception):
    """Base class for exceptions in this module"""
    exitCode = ExitCode.ERR_UNKNOWN

    def __init__(self, ex=None):
        """ Use ex to force the class to have that value as exit code
        """
        if ex is not None:
            self.exitCode = ex


class CapHasClosedError(FaceDetectionError):
    """Exception raised for unexpected cv2.VideoCapture close"""
    exitCode = ExitCode.CameraNotFound

    def __str__(self):
        return "The camera connection has been closed. Please try again"
# }}}


class FaceData:
    """ contains FaceData
    """
    face_x_radian: float
    face_y_radian: float
    face_z_radian: float
    mouth_height_percent: int
    mouth_width_percent: int
    left_eye_percent: int
    right_eye_percent: int

    def __init__(self, x, y, z, mh, mw, le, re):
        self.face_x_radian        = x
        self.face_y_radian        = y
        self.face_z_radian        = z
        self.mouth_height_percent = mh
        self.mouth_width_percent  = mw
        self.left_eye_percent     = le
        self.right_eye_percent    = re

    def default() -> S:
        return FaceData(0.0, 0.0, 0.0, 100, 100, 100, 100)

    @classmethod
    def get(cls: S, face: Face, calib: RawFaceData) -> S:
        """ calculate face rotations from calibration data and landmark
        """
        eyeLineVector = face.rightEye.bottom - face.leftEye.bottom
        raw = RawFaceData.get(face).thresholded(calib)
        # those values are used in the near future.Just ignore this for linting
        leftEdge2Center  = face.leftTemple - raw.faceCenter # noqa
        rightEdge2Center = raw.faceCenter - face.rightTemple # noqa
        chin2Center = raw.faceCenter - face.chinCenter # noqa

        # TODO: how can I notice which side does face face to?
        #       I can't simply compare eyes sizes, 'cus sometimes
        #       user might wink. In that case, I can't recognize properly.
        degreeY = math.acos(round(raw.eyeDistance / calib.eyeDistance, 15))
        degreeX = math.acos(round(raw.faceHeigh / calib.faceHeigh, 15))
        degreeZ = math.atan(round(eyeLineVector.y / eyeLineVector.x, 15))
        # TODO: ^ This some times got error 'Division by 0'

        rotateX = degreeX if raw.faceCenter.y > calib.faceCenter.y\
                            else -1 * degreeX
        rotateY = degreeY if raw.faceCenter.x > calib.faceCenter.x\
                            else -1 * degreeY
        # v Is this correct code? v
        rotateZ = degreeZ

        mouthHPercent = round((raw.mouthHeight / calib.mouthHeight) * 100)
        mouthWPercent = round((raw.mouthWidth / calib.mouthWidth) * 100)
        lEyePercent = round((raw.leftEyeHeight / calib.leftEyeHeight) * 100)
        rEyePercent = round((raw.rightEyeHeight / calib.rightEyeHeight) * 100)

        return cls(clamp(rotateX, -1 / pi, 1 / pi)
                  , clamp(rotateY, -1 / pi, 1 / pi)
                  , clamp(rotateZ, -1 / pi, 1 / pi)
                  , clamp(mouthHPercent, 0, 150)
                  , clamp(mouthWPercent, 0, 150)
                  , clamp(lEyePercent, 0, 150)
                  , clamp(rEyePercent, 0, 150)
                   )

    def toBinary(s):
        """ convert FaceData into binary format
        """
        return struct.pack('!BdddBBBB'
                          , (majorVersionNum << 4) + minorVersionNum
                          , s.face_x_radian
                          , s.face_y_radian
                          , s.face_z_radian
                          , s.mouth_height_percent
                          , s.mouth_width_percent
                          , s.left_eye_percent
                          , s.right_eye_percent
                           )


def clamp(a, _min, _max):
    if a <= _min:
        return _min
    elif _max <= a:
        return _max
    else:
        return a
