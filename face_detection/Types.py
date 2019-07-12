from typing import List, NewType, Union
import numpy
import dlib

Error = NewType('Error', str)
Cv2Image = numpy.ndarray


class CalibrationData:
    eyeDistance: float
    rightEyeSize: float
    leftEyeSize: float
    faceHeigh: float
    faceCenter: dlib.point


class FaceDetectionError(Exception):
    """Base class for exceptions in this module"""
    pass


class CapHasClosedError(FaceDetectionError):
    """Exception raised for unexpected cv2.VideoCapture close"""
    def __str__(self):
        return "The camera connection has been closed. Please try again"
