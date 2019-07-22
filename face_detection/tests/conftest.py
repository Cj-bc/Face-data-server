import sys
import os
import numpy
import cv2

sys.path.append(os.getcwd())

from Types import Cv2Image

faceFrame: Cv2Image = cv2.imread('tests/src/face.jpg')
noFaceFrame: Cv2Image = cv2.imread('tests/src/noface.png')


class mockedCap():
    opened: bool = True
    frame: Cv2Image = noFaceFrame

    def __init__(self, opened: bool, frame: Cv2Image) -> None:
        self.opened = opened
        self.frame = frame

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> Cv2Image:
        ret = True if self.frame.all() == faceFrame.all() else False
        return (ret, self.frame)
