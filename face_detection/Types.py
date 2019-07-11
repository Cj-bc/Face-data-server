from typing import List, NewType, Union, TypeVar

CalibrationData = {"EyeDistance": float}
Landmark = List[int]
Error = NewType('Error', str)


class FaceDetectionError(Exception):
    """Base class for exceptions in this module"""
    pass


class CapHasClosedError(FaceDetectionError):
    """Exception raised for unexpected cv2.VideoCapture close"""
    def __str__(self):
        return "The camera connection has been closed. Please try again"
