from typing import List, NewType, Union, TypeVar
import numpy
import dlib
import dataclasses

Error = NewType('Error', str)
Cv2Image = numpy.ndarray
S = TypeVar('S')


@dataclasses.dataclass(frozen=True)
class RawFaceData:
    eyeDistance: float
#    rightEyeSize: float
#    leftEyeSize: float
    faceHeigh: float
    faceCenter: dlib.point

    def thresholded(self: S, t: S) -> S:
        eD = min(self.eyeDistance, t.eyeDistance)
        fH = min(self.faceHeigh, t.faceHeigh)
        return RawFaceData(eD, fH, self.faceCenter)


class FaceDetectionError(Exception):
    """Base class for exceptions in this module"""
    pass


class CapHasClosedError(FaceDetectionError):
    """Exception raised for unexpected cv2.VideoCapture close"""
    def __str__(self):
        return "The camera connection has been closed. Please try again"
