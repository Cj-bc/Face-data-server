# THIS CODE IS NOT WRITTEN BY ME, BUT BY @kekeho
# Refer to: https://qiita.com/kekeho/items/0b2d4ed5192a4c90a0ac
#
import cv2
import dlib
import math
from math import sqrt
from typing import List, Optional
from faceDetection import (facemark, faceCalibration,
                                         LANDMARK_NUM, getRawFaceData,
                                         waitUntilFaceDetect)
from Types import FaceDetectionError, Cv2Image, RawFaceData, FaceRotations
import sys
import datetime

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


# rotates(landmark: dlib.points) -> (Int, Int, Int)
def rotates(landmark: dlib.points, calib: RawFaceData) -> FaceRotations:
    eyeLineVector = landmark[LANDMARK_NUM["RIGHT_EYE_BOTTOM"]] - \
                            landmark[LANDMARK_NUM["LEFT_EYE_BOTTOM"]]
    print(f"r_e_b: {landmark[LANDMARK_NUM['RIGHT_EYE_BOTTOM']]}, l_e_b: {landmark[LANDMARK_NUM['LEFT_EYE_BOTTOM']]}, Vec: {eyeLineVector}")  # DEBUG
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


def main():
    cap: cv2.VideoCapture = cv2.VideoCapture(0)

    try:
        calibrated: RawFaceData = faceCalibration(cap)
    except FaceDetectionError as e:
        print(f"ERROR: Unexpected things are happened: {e}")
        print("Aborting")
        sys.exit(1)

    while cap.isOpened():

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        rots: FaceRotations = FaceRotations(0, 0, 0)
        _, frame = cap.read()
        landmark: Optional[dlib.points] = facemark(frame)
        # TODO: ^ landmark should never be dlip.points(0) but it does
        print(f"landmark: {landmark}")

        if landmark is not None:
            rots: FaceRotations = rotates(landmark, calibrated)

        print(f"{datetime.datetime.today()}: {rots.x}, {rots.y}, {rots.z}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
