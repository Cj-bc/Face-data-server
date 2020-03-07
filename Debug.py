import cv2
from typing import (Optional, List, Tuple)

from FaceDataServer.Types import (Face, Cv2Image, Coord
                                 ,
                                  )
import numpy as np


def mapInt(a):
    return tuple(map(int, a))


def lines(img: Cv2Image, center: Tuple[int, int]
         , ends: List[Tuple[Coord, Coord]]):
    if ends == []:
        return img

    def addOffset(a: Tuple[int, int], b: Tuple[int, int]):
        a1, a2 = a
        b1, b2 = b
        return (a1 + b1, a2 + b2)

    s, e = ends[0]
    s_centerized = addOffset(mapInt(s.toTuple()), center)
    e_centerized = addOffset(mapInt(e.toTuple()), center)
    img_ = cv2.line(img, s_centerized, e_centerized, (0, 256, 0))
    tail = ends[1:]
    return lines(img_, center, tail)


def face2Image(size: Tuple[float, float], face: Face
              , frame: Optional[Cv2Image]=None) -> Cv2Image:
    h, w = size
    blank_img = np.zeros((round(h), round(w), 3), np.uint8)
    blank_img[:, :] = (60, 60, 60)
    center = (round(h / 2), round(w / 2))\
             if frame is None\
             else mapInt(face.center.toTuple())
    img = blank_img\
          if frame is None\
          else frame

    lineEnds = [(face.leftEye.bottom, face.leftEye.rightSide)
               , (face.leftEye.rightSide, face.leftEye.top)
               , (face.leftEye.top, face.leftEye.leftSide)
               , (face.leftEye.leftSide, face.leftEye.bottom)
               # right eye
               , (face.rightEye.bottom, face.rightEye.rightSide)
               , (face.rightEye.rightSide, face.rightEye.top)
               , (face.rightEye.top, face.rightEye.leftSide)
               , (face.rightEye.leftSide, face.rightEye.bottom)
               # mouth
               , (face.mouth.bottom, face.mouth.rightSide)
               , (face.mouth.rightSide, face.mouth.top)
               , (face.mouth.top, face.mouth.leftSide)
               , (face.mouth.leftSide, face.mouth.bottom)
                ]
    return lines(img, center, lineEnds)
