# THIS CODE IS NOT WRITTEN BY ME, BUT BY @kekeho
# Refer to: https://qiita.com/kekeho/items/0b2d4ed5192a4c90a0ac
#
import cv2
import dlib
from math import sqrt
import math
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
    return dlib.point(abs(a.x), abs(a.y))


def size(leftTop: Coord, rightBottom: Coord) -> float:
    x1, y1 = leftTop
    x2, y2 = rightBottom
    return abs(x2 - x1) * abs(y2 - y1)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

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

        eyeDistance = point_abs(landmark[LANDMARK_NUM["LEFT_EYE_R"]]
                                - landmark[LANDMARK_NUM["RIGHT_EYE_L"]])
        rightEyeSize = size(( landmark[LANDMARK_NUM["RIGHT_EYE_R"]][0]
                            , landmark[LANDMARK_NUM["RIGHT_EYE_TOP"]][1])
                           ,( landmark[LANDMARK_NUM["RIGHT_EYE_L"]][0]
                            , landmark[LANDMARK_NUM["RIGHT_EYE_BOTTOM"]][1])
                           )
        leftEyeSize = size((landmark[LANDMARK_NUM["LEFT_EYE_R"]][0]
                            , landmark[LANDMARK_NUM["LEFT_EYE_TOP"]][1])
                           , (landmark[LANDMARK_NUM["LEFT_EYE_L"]][0]
                            , landmark[LANDMARK_NUM["LEFT_EYE_BOTTOM"]][1])
                           )

        angle = math.cos(eyeDistance)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
