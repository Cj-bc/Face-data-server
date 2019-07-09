# THIS CODE IS NOT WRITTEN BY ME, BUT BY @kekeho
# Refer to: https://qiita.com/kekeho/items/0b2d4ed5192a4c90a0ac
#
import cv2
import dlib
import numpy
import functools
import os
from faceDetection import facemark, faceCalibration, normalization


red = (0, 0, 255)
green = (0, 255, 0)
blue = (255, 0, 0)




if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        landmarks = normalization(facemark(gray))

        for landmark in landmarks:
            functools.reduce(lambda x, points: cv2.drawMarker(frame, (points[0], points[1]), green)
                            , map(lambda n:landmark[n], [ LANDMARK_NUM_RIGHT_EYE_L
                                                        , LANDMARK_NUM_LEFT_EYE_R
                                                        , LANDMARK_NUM_LEFT_EYE_L
                                                        , LANDMARK_NUM_MOUSE_R
                                                        , LANDMARK_NUM_MOUSE_L
                                                        , LANDMARK_NUM_MOUSE_TOP
                                                        , LANDMARK_NUM_MOUSE_BOTTOM
                                                        ])
                            , [])
            cv2.drawMarker(frame,(landmark[LANDMARK_NUM_RIGHT_EYE_R][0], landmark[LANDMARK_NUM_RIGHT_EYE_R][1]),red)

        cv2.imshow("video frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
