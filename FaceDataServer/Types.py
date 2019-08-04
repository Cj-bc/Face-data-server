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


Error = NewType('Error', str)
Cv2Image = numpy.ndarray
S = TypeVar('S')


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

        _faceCenterX = (max(map(lambda p: p.x, landmark))
                        + min(map(lambda p: p.x, landmark))) // 2
        _faceCenterY = (max(map(lambda p: p.y, landmark))
                        + min(map(lambda p: p.y, landmark))) // 2
        faceCenter = dlib.dpoint(_faceCenterX, _faceCenterY)

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


class FaceDetectionError(Exception):
    """Base class for exceptions in this module"""
    pass


class CapHasClosedError(FaceDetectionError):
    """Exception raised for unexpected cv2.VideoCapture close"""
    def __str__(self):
        return "The camera connection has been closed. Please try again"
