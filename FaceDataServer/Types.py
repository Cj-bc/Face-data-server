from typing import NewType, TypeVar
import numpy
import dlib
import dataclasses
import math


# Those values are defined based on this site image:
#   https://qiita.com/kekeho/items/0b2d4ed5192a4c90a0ac
# ignore
LANDMARK_NUM = {"TIN_CENTER": 19
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
               , "EYEBROW_RIGHT_L": 174
                }

# type aliases {{{

Error = NewType('Error', str)
Cv2Image = numpy.ndarray
S = TypeVar('S')
# }}}


# Coordinates {{{
class Coord():
    """Base class to express Coordinates
        This is made to be converted from dlib.dpoint
    """
    x: float
    y: float

    @staticmethod
    def default(cls):
        """return default coordinate."""
        return cls(0, 0)

class AbsoluteCoord(Coord):
    pass


# }}}


# Each face parts {{{
class Part():
    bottom: Coord
    top: Coord
    leftSide: Coord
    rightSide: Coord

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
    leftEye: Eye
    rightEye: Eye
    mouth: Mouth
    nose: Nose
    leftEyeBrow: EyeBrow
    rightEyeBrow: EyeBrow

    @classmethod
    def default(cls):
        """return default coordinate."""
        return cls(AbsoluteCoord.default(), Eye.default(), Eye.default()
                   Mouth.default(), Nose.default(), EyeBrow.default()
                   EyeBrow.default())



@dataclasses.dataclass(frozen=True)
class RawFaceData:
    eyeDistance: float
#    rightEyeSize: float
#    leftEyeSize: float
    faceHeigh: float
    faceCenter: dlib.dpoint

    @classmethod
    def get(cls: S, landmark: dlib.dpoints) -> S:
        """ Return RawFaceData from dlib.points
        """
        eyeDistance  = abs(landmark[LANDMARK_NUM["RIGHT_EYE_L"]].x
                           - landmark[LANDMARK_NUM["LEFT_EYE_R"]].x)

        _middleForehead = (landmark[LANDMARK_NUM["EYEBROW_LEFT_R"]]
                          + landmark[LANDMARK_NUM["EYEBROW_RIGHT_L"]]) / 2
        _faceHeighVector  = _middleForehead\
                            - landmark[LANDMARK_NUM["TIN_CENTER"]]
        faceHeigh = math.sqrt(_faceHeighVector.x ** 2
                              + _faceHeighVector.y ** 2)
        faceCenter = landmark["49"]
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
    def get(cls: S, landmark: dlib.dpoints, calib: RawFaceData) -> S:
        """ calculate face rotations from calibration data and landmark
        """
        eyeLineVector = landmark[LANDMARK_NUM["RIGHT_EYE_BOTTOM"]] - \
                                landmark[LANDMARK_NUM["LEFT_EYE_BOTTOM"]]
        raw = RawFaceData.get(landmark).thresholded(calib)

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
