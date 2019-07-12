# THIS CODE IS NOT WRITTEN BY ME, BUT BY @kekeho
# Refer to: https://qiita.com/kekeho/items/0b2d4ed5192a4c90a0ac
#
import cv2
import dlib
import math
from math import sqrt
from typing import List
from faceDetection import facemark, faceCalibration,  LANDMARK_NUM
from Types import FaceDetectionError, Cv2Image, CalibrationData
import sys

# type definitions
Coord = (int, int)

red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)


def point_abs(a: dlib.point) -> dlib.point:
    """return 'dlib.point' which x,y are both absolute value """
    return dlib.point(abs(a.x), abs(a.y))


def area_rect(a: dlib.point, b: dlib.point,
              c: dlib.point, d: dlib.point) -> int:
    """Calculate region of given points """
    upper  = max(map(lambda n: n.y, [a, b, c, d]))
    bottom = min(map(lambda n: n.y, [a, b, c, d]))
    right  = max(map(lambda n: n.x, [a, b, c, d]))
    left   = min(map(lambda n: n.x, [a, b, c, d]))
    return (upper - bottom) * (right - left)


def getMagnitude(p: dlib.point):
    """Get magnitude of given point vector"""
    return sqrt((p.x ^ 2) + (p.y ^ 2))

def main():
    cap: cv2.VideoCapture = cv2.VideoCapture(0)

    try:
        calibrated: CalibrationData = faceCalibration(cap)
    except FaceDetectionError as e:
        print(f"ERROR: Unexpected things are happened: {e}")
        print("Aborting")
        sys.exit(1)

    while cap.isOpened():
        _, frame = cap.read()
        gray: Cv2Image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        landmarks: List[dlib.point] = facemark(gray)
        if landmarks == []:
            continue
        else:
            landmark = landmarks[0]

        eyeDistance  = min(abs(landmark[LANDMARK_NUM["RIGHT_EYE_L"]].x -
                               landmark[LANDMARK_NUM["LEFT_EYE_R"]].x)
                          , calibrated.eyeDistance)
        rightEyeSize = area_rect(landmark[LANDMARK_NUM["RIGHT_EYE_R"]]
                                , landmark[LANDMARK_NUM["RIGHT_EYE_L"]]
                                , landmark[LANDMARK_NUM["RIGHT_EYE_TOP"]]
                                , landmark[LANDMARK_NUM["RIGHT_EYE_BOTTOM"]]
                                 )
        leftEyeSize  = area_rect(landmark[LANDMARK_NUM["LEFT_EYE_R"]]
                                , landmark[LANDMARK_NUM["LEFT_EYE_TOP"]]
                                , landmark[LANDMARK_NUM["LEFT_EYE_L"]]
                                , landmark[LANDMARK_NUM["LEFT_EYE_BOTTOM"]]
                                 )
        eyeLineVector = point_abs(landmark[LANDMARK_NUM["RIGHT_EYE_BOTTOM"]] -
                                  landmark[LANDMARK_NUM["Left_EYE_BOTTOM"]])
        # TODO: how can I notice which side does face face to?
        #       I can't simply compare eyes sizes, 'cus sometimes
        #       user might wink. In that case, I can't recognize properly.
        rotateY = math.acos(eyeDistance / calibrated.eyeDistance)
        # rotateX
        rotateZ = math.atan(eyeLineVector.y / eyeLineVector.x)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
