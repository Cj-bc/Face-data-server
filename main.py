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


def lines(img: Cv2Image, ends: List[Tuple[Coord, Coord]]):
    if ends == []:
        return img

    def round_(a):
        return tuple(map(round, a))
    s, e = ends[0]
    img_ = cv2.line(img, round_(s.toTuple()), round_(e.toTuple()), (0, 70, 0))
    tail = ends[1:]
    return lines(img_, tail)


def face2Image(size, face: Face) -> Cv2Image:
    h, w = size
    blank_img = np.zeros((round(h), round(w), 3), np.uint8)
    blank_img[:, :] = (60, 60, 60)
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
    lines(blank_img, lineEnds)


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

                cv2.imshow(face2Image(videoSize, face))

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
