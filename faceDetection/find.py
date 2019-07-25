# THIS CODE IS NOT WRITTEN BY ME, BUT BY @kekeho
# Refer to: https://qiita.com/kekeho/items/0b2d4ed5192a4c90a0ac
#
import cv2
import dlib
import math
from .faceDetection import (LANDMARK_NUM, getRawFaceData)
from .Types import RawFaceData, FaceRotations

# type definitions
Coord = (int, int)

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)


# point_abs(a: dlib.point) -> dlib.point {{{
def point_abs(a: dlib.point) -> dlib.point:
    """return 'dlib.point' which x,y are both absolute value """
    return dlib.point(abs(a.x), abs(a.y))
# }}}


# area_rect(a: dlib.point, b: dlib.point c: dlib.point, d: dlib.point) -> int{{{
def area_rect(a: dlib.point, b: dlib.point,
              c: dlib.point, d: dlib.point) -> int:
    """Calculate region of given points

        If we have that points(.) below,

        -----.-------
        .------------
        ------------.
        --------.----

        'area_rect' will returns area of:

        #############
        #############
        #############
        #############

        Not something like:

        -----#-------
        ##########---
        --###########
        ------###----
    """
    upper  = max(map(lambda n: n.y, [a, b, c, d]))
    bottom = min(map(lambda n: n.y, [a, b, c, d]))
    right  = max(map(lambda n: n.x, [a, b, c, d]))
    left   = min(map(lambda n: n.x, [a, b, c, d]))
    return (upper - bottom) * (right - left)
# }}}


# rotates(landmark: dlib.points, calib: RawFaceData) -> FaceRotations
def rotates(landmark: dlib.points, calib: RawFaceData) -> FaceRotations:
    """ calculate face rotations from calibration data and landmark
    """
    eyeLineVector = landmark[LANDMARK_NUM["RIGHT_EYE_BOTTOM"]] - \
                            landmark[LANDMARK_NUM["LEFT_EYE_BOTTOM"]]
    raw = getRawFaceData(landmark).thresholded(calib)

    # TODO: how can I notice which side does face face to?
    #       I can't simply compare eyes sizes, 'cus sometimes
    #       user might wink. In that case, I can't recognize properly.
    degreeY = math.acos(raw.eyeDistance / calib.eyeDistance)
    degreeX = math.acos(raw.faceHeigh / calib.faceHeigh)
    degreeZ = math.atan(abs(eyeLineVector.y / eyeLineVector.x))
    # TODO: ^ This some times got error 'Division by 0'

    rotateX = degreeX if raw.faceCenter.y > calib.faceCenter.y\
                        else -1 * degreeX
    rotateY = degreeY if raw.faceCenter.x > calib.faceCenter.x\
                        else -1 * degreeY
    # v Is this correct code? v
    rotateZ = degreeZ
    return FaceRotations(rotateX, rotateY, rotateZ)
