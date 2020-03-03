import cv2
import dlib
from contextlib import closing
import secrets
import socket
from concurrent import futures
from typing import (Optional, List, Tuple)

from FaceDataServer.faceDetection import (faceCalibration, facemark)
from FaceDataServer.Types import (RawFaceData,
                                 FaceDetectionError, Face, ExitCode,
                                 FaceData, Cv2Image, Coord)
from logging import getLogger, Logger
import logging.config as loggingConfig
import numpy as np

# Loggers {{{
configuu = {"version": 1
           , "handlers": {"console": {"class": "logging.StreamHandler"}
                        , "file": {"class": "logging.FileHandler"
                            , "filename": "faceDataServer.log"
                            , "formatter": "simpleFormatter"}}
           , "loggers": {"Servicer": {}}
           , "root": {"level": "DEBUG"
                     , "handlers": ["file"]
                     , "formatters": ["simpleFormatter"]}
           , "formatters":
                {"simpleFormatter":
                    {"format": "%(asctime)s - %(name)s "
                               "- %(levelname)s - %(message)s"
                     }
                 }
            }
loggingConfig.dictConfig(configuu)
logger: Logger = getLogger('main')
logger_servicer: Logger = getLogger('Servicer')
# }}}


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


def main():
    # Server setting
    server_address = "0.0.0.0"
    server_port = 5032
    multicast_group = '226.0.0.1'

    # Preparing camera
    cap: cv2.VideoCapture = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
    videoSize = (cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                , cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                 )

    # ========== calibration ==========
    try:
        calib: RawFaceData = faceCalibration(cap)
    except FaceDetectionError as e:
        cap.release()
        logger_servicer.info(f"ERROR: Unexpected things are happened: {e}")
        logger_servicer.info("Aborting")
        return

    logger_servicer.debug("Calibrated.")
    logger_servicer.debug(f"cap: {cap}")

    try:
        # Preparing socket
        with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as sock:
            # Some options. See `man setsockopt'
            sock.setsockopt(socket.SOL_SOCKET
                           , socket.SO_BROADCAST, 1)  # enable BROADCAST
            sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF
                           , socket.inet_aton(server_address))

            # ========== Main loop ==========
            while True:
                if cap.isOpened() is not True:
                    break

                _, frame = cap.read()
                landmark: Optional[dlib.points] = facemark(frame)

                face: Face          = Face.default()\
                                        if landmark is None\
                                        else Face.fromDPoints(landmark)

                data: FaceData = FaceData.default()\
                                        if landmark is None\
                                        else FaceData.get(face, calib)

                sock.sendto(data.toBinary(), (multicast_group, server_port))

                cv2.imshow('face wire test', face2Image(videoSize, face, frame))
                cv2.waitKey(1)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
